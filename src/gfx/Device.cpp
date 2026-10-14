#include "gfx/Device.h"

#include <iostream>

namespace gfx {

void log(const std::string& message)
{
    std::cout << message << std::endl;
    OutputDebugStringA((message + "\n").c_str());
}

std::string narrow(const std::wstring& wide)
{
    if (wide.empty()) {
        return {};
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), out.data(), size, nullptr, nullptr);
    return out;
}

// ---------------------------------------------------------------------------
// UploadRing
// ---------------------------------------------------------------------------

void UploadRing::init(ID3D12Device* device, uint64_t bytesPerSlot, uint32_t slotCount)
{
    m_bytesPerSlot = bytesPerSlot;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytesPerSlot * slotCount;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                          nullptr, IID_PPV_ARGS(&m_buffer)),
          "CreateCommittedResource(upload ring)");
    m_buffer->SetName(L"UploadRing");
    check(m_buffer->Map(0, nullptr, reinterpret_cast<void**>(&m_mapped)), "Map(upload ring)");
}

void UploadRing::beginSlot(uint32_t slot)
{
    m_slotBegin = m_bytesPerSlot * slot;
    m_offset = 0;
}

UploadRing::Allocation UploadRing::allocate(uint64_t bytes)
{
    // [POINT] 定数バッファビューの配置は 256 バイト境界が必須（D3D12 の規則）。
    constexpr uint64_t kAlign = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    const uint64_t aligned = (bytes + kAlign - 1) & ~(kAlign - 1);
    if (m_offset + aligned > m_bytesPerSlot) {
        throw std::runtime_error("UploadRing slot exhausted");
    }
    Allocation a;
    a.cpu = m_mapped + m_slotBegin + m_offset;
    a.gpu = m_buffer->GetGPUVirtualAddress() + m_slotBegin + m_offset;
    m_offset += aligned;
    return a;
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

Device::Device(bool debugLayer)
{
    if (debugLayer) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            log("D3D12 debug layer enabled");
        }
    }
    createFactory();

    // 高性能 GPU を優先してアダプタを選ぶ。
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; m_factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                           IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)))) {
            m_adapterName = narrow(desc.Description);
            break;
        }
    }
    if (!m_device) {
        throw std::runtime_error("No D3D12 device available");
    }
    log("Adapter: " + m_adapterName);

    if (debugLayer) {
        ComPtr<ID3D12InfoQueue> infoQueue;
        if (SUCCEEDED(m_device.As(&infoQueue))) {
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
            infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        }
    }

    // [POINT] Bindless（ResourceDescriptorHeap の直接インデックス）には
    //         Shader Model 6.6 とリソースバインディング Tier 3 が必要。
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_6};
    check(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
          "CheckFeatureSupport(shader model)");
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    check(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)),
          "CheckFeatureSupport(options)");
    if (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
        options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
        throw std::runtime_error("Shader Model 6.6 and resource binding tier 3 are required");
    }

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    check(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_queue)), "CreateCommandQueue");

    for (auto& allocator : m_allocators) {
        check(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
              "CreateCommandAllocator");
    }
    check(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_allocators[0].Get(), nullptr,
                                      IID_PPV_ARGS(&m_cmd)),
          "CreateCommandList");
    m_cmd->Close();
    check(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_immediateAllocator)),
          "CreateCommandAllocator(immediate)");
    check(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_immediateAllocator.Get(), nullptr,
                                      IID_PPV_ARGS(&m_immediateCmd)),
          "CreateCommandList(immediate)");
    m_immediateCmd->Close();

    check(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), "CreateFence");
    m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    m_bindless.init(m_device.Get(), 4096);
    m_rtvPool.init(m_device.Get(), 64);
    m_upload.init(m_device.Get(), 1u << 20, kFramesInFlight);
}

Device::~Device()
{
    if (m_queue) {
        waitIdle();
    }
    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
    }
}

void Device::createFactory()
{
    m_factory.Reset();
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory)), "CreateDXGIFactory2");
}

bool Device::refreshFactoryIfStale()
{
    if (m_factory->IsCurrent()) {
        return false;
    }
    createFactory();
    return true;
}

uint64_t Device::signal()
{
    ++m_fenceValue;
    check(m_queue->Signal(m_fence.Get(), m_fenceValue), "Signal");
    return m_fenceValue;
}

void Device::waitFor(uint64_t value)
{
    if (m_fence->GetCompletedValue() < value) {
        check(m_fence->SetEventOnCompletion(value, m_fenceEvent), "SetEventOnCompletion");
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void Device::waitIdle()
{
    waitFor(signal());
}

ID3D12GraphicsCommandList* Device::beginFrame()
{
    m_frameSlot = static_cast<uint32_t>(m_frameNumber % kFramesInFlight);
    waitFor(m_slotFence[m_frameSlot]);
    check(m_allocators[m_frameSlot]->Reset(), "CommandAllocator::Reset");
    check(m_cmd->Reset(m_allocators[m_frameSlot].Get(), nullptr), "CommandList::Reset");
    m_upload.beginSlot(m_frameSlot);
    ID3D12DescriptorHeap* heaps[] = {m_bindless.heap()};
    m_cmd->SetDescriptorHeaps(1, heaps);
    return m_cmd.Get();
}

void Device::endFrame()
{
    check(m_cmd->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {m_cmd.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    m_slotFence[m_frameSlot] = signal();
    ++m_frameNumber;
}

void Device::executeImmediate(const std::function<void(ID3D12GraphicsCommandList*)>& record)
{
    check(m_immediateAllocator->Reset(), "CommandAllocator::Reset(immediate)");
    check(m_immediateCmd->Reset(m_immediateAllocator.Get(), nullptr), "CommandList::Reset(immediate)");
    ID3D12DescriptorHeap* heaps[] = {m_bindless.heap()};
    m_immediateCmd->SetDescriptorHeaps(1, heaps);
    record(m_immediateCmd.Get());
    check(m_immediateCmd->Close(), "CommandList::Close(immediate)");
    ID3D12CommandList* lists[] = {m_immediateCmd.Get()};
    m_queue->ExecuteCommandLists(1, lists);
    waitIdle();
}

} // namespace gfx
