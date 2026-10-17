#include "gfx/GpuTimer.h"

#include "gfx/Texture.h"

namespace gfx {

GpuTimer::GpuTimer(Device& device, uint32_t slotCount)
{
    D3D12_QUERY_HEAP_DESC desc{};
    desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    desc.Count = slotCount * kMaxScopes * 2;
    check(device.d3d()->CreateQueryHeap(&desc, IID_PPV_ARGS(&m_heap)), "CreateQueryHeap(timestamp)");
    m_readback = createBuffer(device.d3d(), sizeof(uint64_t) * desc.Count, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST);
    UINT64 frequency = 1;
    check(device.queue()->GetTimestampFrequency(&frequency), "GetTimestampFrequency");
    m_ticksPerMs = static_cast<double>(frequency) / 1000.0;
    m_names.resize(slotCount);
}

void GpuTimer::beginFrame(uint32_t slot)
{
    m_slot = slot;
    m_names[slot].clear();
}

uint32_t GpuTimer::begin(ID3D12GraphicsCommandList* cmd, const char* name)
{
    std::vector<std::string>& names = m_names[m_slot];
    if (names.size() >= kMaxScopes) {
        return kMaxScopes;
    }
    const uint32_t scope = static_cast<uint32_t>(names.size());
    names.emplace_back(name);
    // [POINT] タイムスタンプは前のパスの完了を待たずに書かれることがあり、そのままだと前のパスの残りの時間
    //         （例: Display のピクセルシェーダー）が次の区間に入る。対象リソースなしの UAV バリアで直前の
    //         シェーダーの完了を待ってから測る。パス間の重なりがなくなる分、合計はやや増える。
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    cmd->ResourceBarrier(1, &barrier);
    cmd->EndQuery(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, (m_slot * kMaxScopes + scope) * 2);
    return scope;
}

void GpuTimer::end(ID3D12GraphicsCommandList* cmd, uint32_t scope)
{
    if (scope >= kMaxScopes) {
        return;
    }
    cmd->EndQuery(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, (m_slot * kMaxScopes + scope) * 2 + 1);
}

void GpuTimer::resolve(ID3D12GraphicsCommandList* cmd)
{
    const UINT count = static_cast<UINT>(m_names[m_slot].size()) * 2;
    if (count == 0) {
        return;
    }
    const UINT first = m_slot * kMaxScopes * 2;
    cmd->ResolveQueryData(m_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, first, count, m_readback.Get(),
                          sizeof(uint64_t) * first);
}

std::vector<GpuTimer::Result> GpuTimer::read(uint32_t slot) const
{
    std::vector<Result> results;
    const std::vector<std::string>& names = m_names[slot];
    if (names.empty()) {
        return results;
    }
    const UINT first = slot * kMaxScopes * 2;
    const D3D12_RANGE range{sizeof(uint64_t) * first, sizeof(uint64_t) * (first + names.size() * 2)};
    uint64_t* data = nullptr;
    if (FAILED(m_readback->Map(0, &range, reinterpret_cast<void**>(&data)))) {
        return results;
    }
    for (size_t i = 0; i < names.size(); ++i) {
        const uint64_t t0 = data[first + i * 2];
        const uint64_t t1 = data[first + i * 2 + 1];
        results.push_back({names[i], t1 > t0 ? static_cast<double>(t1 - t0) / m_ticksPerMs : 0.0});
    }
    const D3D12_RANGE none{0, 0};
    m_readback->Unmap(0, &none);
    return results;
}

} // namespace gfx
