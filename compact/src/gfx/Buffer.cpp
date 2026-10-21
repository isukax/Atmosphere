#include "gfx/Buffer.h"

#include "gfx/Texture.h"

#include <cstring>

namespace compact::gfx {

Buffer::Buffer(Device& device, const BufferDesc& desc)
    : m_device(&device), m_desc(desc)
{
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = desc.sizeBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (desc.uav) {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    m_state = D3D12_RESOURCE_STATE_COMMON;
    check(device.d3d()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, m_state, nullptr,
                                                IID_PPV_ARGS(&m_resource)),
          "CreateCommittedResource(buffer)");
    m_resource->SetName(desc.name);

    const UINT elementCount = desc.strideBytes > 0 ? static_cast<UINT>(desc.sizeBytes / desc.strideBytes) : 0;
    if (desc.srv && desc.strideBytes > 0) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = elementCount;
        srv.Buffer.StructureByteStride = desc.strideBytes;
        m_srvIndex = device.bindless().allocate();
        device.d3d()->CreateShaderResourceView(m_resource.Get(), &srv, device.bindless().cpu(m_srvIndex));
    }
    if (desc.uav && desc.strideBytes > 0) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = elementCount;
        uav.Buffer.StructureByteStride = desc.strideBytes;
        m_uavIndex = device.bindless().allocate();
        device.d3d()->CreateUnorderedAccessView(m_resource.Get(), nullptr, &uav, device.bindless().cpu(m_uavIndex));
    }
}

Buffer::~Buffer()
{
    release();
}

Buffer::Buffer(Buffer&& other) noexcept
{
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other) {
        release();
        m_device = other.m_device;
        m_desc = other.m_desc;
        m_resource = std::move(other.m_resource);
        m_state = other.m_state;
        m_srvIndex = other.m_srvIndex;
        m_uavIndex = other.m_uavIndex;
        other.m_device = nullptr;
        other.m_srvIndex = kInvalidDescriptorIndex;
        other.m_uavIndex = kInvalidDescriptorIndex;
    }
    return *this;
}

void Buffer::release()
{
    if (!m_device) {
        return;
    }
    // Texture と同じく、作り直す前に Device::waitIdle() を呼ぶのは呼び出し側の責任。
    m_device->bindless().free(m_srvIndex);
    m_device->bindless().free(m_uavIndex);
    m_srvIndex = kInvalidDescriptorIndex;
    m_uavIndex = kInvalidDescriptorIndex;
    m_resource.Reset();
    m_device = nullptr;
}

void Buffer::transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES to)
{
    if (m_state == to) {
        if (to == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER uav{};
            uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource = m_resource.Get();
            cmd->ResourceBarrier(1, &uav);
        }
        return;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_resource.Get();
    barrier.Transition.StateBefore = m_state;
    barrier.Transition.StateAfter = to;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &barrier);
    m_state = to;
}

void Buffer::upload(Device& device, const void* data, uint64_t bytes, D3D12_RESOURCE_STATES finalState)
{
    ComPtr<ID3D12Resource> staging = createBuffer(device.d3d(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    check(staging->Map(0, nullptr, &mapped), "Map(buffer staging)");
    std::memcpy(mapped, data, bytes);
    staging->Unmap(0, nullptr);
    device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
        transition(cmd, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyBufferRegion(m_resource.Get(), 0, staging.Get(), 0, bytes);
        transition(cmd, finalState);
    });
}



} // namespace gfx
