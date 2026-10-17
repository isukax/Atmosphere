#pragma once

#include "gfx/Device.h"

#include <vector>

namespace gfx {

struct BufferDesc {
    uint64_t sizeBytes = 0;
    uint32_t strideBytes = 0;  // 構造化バッファの 1 要素のバイト数（SRV/UAV を作るとき必須）
    bool srv = true;
    bool uav = false;
    const wchar_t* name = L"Buffer";
};

/// デフォルトヒープのバッファと、その Bindless SRV/UAV インデックス（構造化バッファ）を所有する。
class Buffer {
public:
    Buffer() = default;
    Buffer(Device& device, const BufferDesc& desc);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    explicit operator bool() const { return m_resource != nullptr; }
    ID3D12Resource* resource() const { return m_resource.Get(); }
    const BufferDesc& desc() const { return m_desc; }
    uint32_t srvIndex() const { return m_srvIndex; }
    uint32_t uavIndex() const { return m_uavIndex; }

    void transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES to);
    /// CPU のデータをアップロードし、`finalState` にする。完了まで待つ。
    void upload(Device& device, const void* data, uint64_t bytes, D3D12_RESOURCE_STATES finalState);
    /// GPU の内容を読み戻す。完了まで待つ。
    std::vector<uint8_t> readback(Device& device);

private:
    void release();

    Device* m_device = nullptr;
    BufferDesc m_desc{};
    ComPtr<ID3D12Resource> m_resource;
    D3D12_RESOURCE_STATES m_state = D3D12_RESOURCE_STATE_COMMON;
    uint32_t m_srvIndex = kInvalidDescriptorIndex;
    uint32_t m_uavIndex = kInvalidDescriptorIndex;
};

} // namespace gfx
