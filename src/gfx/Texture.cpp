#include "gfx/Texture.h"

#include <cstring>

namespace gfx {

ComPtr<ID3D12Resource> createBuffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE heapType,
                                    D3D12_RESOURCE_STATES initialState)
{
    D3D12_HEAP_PROPERTIES heap{heapType};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> buffer;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initialState, nullptr,
                                          IID_PPV_ARGS(&buffer)),
          "CreateCommittedResource(buffer)");
    return buffer;
}

Texture::Texture(Device& device, const TextureDesc& desc)
    : m_device(&device), m_desc(desc)
{
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = desc.width;
    rd.Height = desc.height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = desc.format;
    rd.SampleDesc.Count = 1;
    if (desc.uav) {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if (desc.rtv) {
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    m_state = D3D12_RESOURCE_STATE_COMMON;
    check(device.d3d()->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, m_state, nullptr,
                                                IID_PPV_ARGS(&m_resource)),
          "CreateCommittedResource(texture)");
    m_resource->SetName(desc.name);

    if (desc.srv) {
        m_srvIndex = device.bindless().allocate();
        device.d3d()->CreateShaderResourceView(m_resource.Get(), nullptr, device.bindless().cpu(m_srvIndex));
    }
    if (desc.uav) {
        m_uavIndex = device.bindless().allocate();
        device.d3d()->CreateUnorderedAccessView(m_resource.Get(), nullptr, nullptr, device.bindless().cpu(m_uavIndex));
    }
    if (desc.rtv) {
        m_rtv = device.rtvPool().allocate();
        device.d3d()->CreateRenderTargetView(m_resource.Get(), nullptr, m_rtv);
    }
}

Texture::~Texture()
{
    release();
}

Texture::Texture(Texture&& other) noexcept
{
    *this = std::move(other);
}

Texture& Texture::operator=(Texture&& other) noexcept
{
    if (this != &other) {
        release();
        m_device = other.m_device;
        m_desc = other.m_desc;
        m_resource = std::move(other.m_resource);
        m_state = other.m_state;
        m_srvIndex = other.m_srvIndex;
        m_uavIndex = other.m_uavIndex;
        m_rtv = other.m_rtv;
        other.m_device = nullptr;
        other.m_srvIndex = kInvalidDescriptorIndex;
        other.m_uavIndex = kInvalidDescriptorIndex;
        other.m_rtv = {};
    }
    return *this;
}

void Texture::release()
{
    if (!m_device) {
        return;
    }
    // [POINT] ディスクリプタはここで即座に返却する。GPU が使用中のまま返却すると、
    //         次に確保した別リソースが同じスロットを上書きしてしまうので、
    //         テクスチャを作り直す前に必ず Device::waitIdle() を呼ぶ（呼び出し側の責任）。
    m_device->bindless().free(m_srvIndex);
    m_device->bindless().free(m_uavIndex);
    m_device->rtvPool().free(m_rtv);
    m_srvIndex = kInvalidDescriptorIndex;
    m_uavIndex = kInvalidDescriptorIndex;
    m_rtv = {};
    m_resource.Reset();
    m_device = nullptr;
}

void Texture::transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES to)
{
    if (m_state == to) {
        if (to == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            // 連続する UAV 書き込みの順序を保証する。
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

void Texture::upload(Device& device, const void* data, uint32_t bytesPerPixel)
{
    D3D12_RESOURCE_DESC rd = m_resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 totalBytes = 0;
    device.d3d()->GetCopyableFootprints(&rd, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);
    ComPtr<ID3D12Resource> staging = createBuffer(device.d3d(), totalBytes, D3D12_HEAP_TYPE_UPLOAD,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ);
    uint8_t* mapped = nullptr;
    check(staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Map(staging)");
    const uint32_t rowBytes = m_desc.width * bytesPerPixel;
    for (uint32_t y = 0; y < m_desc.height; ++y) {
        std::memcpy(mapped + footprint.Offset + static_cast<size_t>(y) * footprint.Footprint.RowPitch,
                    static_cast<const uint8_t*>(data) + static_cast<size_t>(y) * rowBytes, rowBytes);
    }
    staging->Unmap(0, nullptr);

    device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
        transition(cmd, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{m_resource.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        src.PlacedFootprint = footprint;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    });
}

std::vector<uint8_t> Texture::readback(Device& device, uint32_t bytesPerPixel)
{
    D3D12_RESOURCE_DESC rd = m_resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 totalBytes = 0;
    device.d3d()->GetCopyableFootprints(&rd, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);
    ComPtr<ID3D12Resource> staging = createBuffer(device.d3d(), totalBytes, D3D12_HEAP_TYPE_READBACK,
                                                  D3D12_RESOURCE_STATE_COPY_DEST);
    device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
        transition(cmd, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{m_resource.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dst{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = footprint;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    });

    std::vector<uint8_t> out(static_cast<size_t>(m_desc.width) * m_desc.height * bytesPerPixel);
    uint8_t* mapped = nullptr;
    check(staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Map(readback)");
    const uint32_t rowBytes = m_desc.width * bytesPerPixel;
    for (uint32_t y = 0; y < m_desc.height; ++y) {
        std::memcpy(out.data() + static_cast<size_t>(y) * rowBytes,
                    mapped + footprint.Offset + static_cast<size_t>(y) * footprint.Footprint.RowPitch, rowBytes);
    }
    staging->Unmap(0, nullptr);
    return out;
}

} // namespace gfx
