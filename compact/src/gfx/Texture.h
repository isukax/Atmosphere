#pragma once

#include "gfx/Device.h"

#include <vector>

namespace compact::gfx {

struct TextureDesc {
    uint32_t width = 1;
    uint32_t height = 1;
    DXGI_FORMAT format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    bool srv = true;
    bool uav = false;
    bool rtv = false;
    const wchar_t* name = L"Texture";
    uint32_t depth = 1;  // > 1 なら 3D テクスチャ
    bool dsv = false;    // 深度バッファ（format は DXGI_FORMAT_D32_FLOAT のみ対応）
};

/// 2D / 3D テクスチャと、その Bindless SRV/UAV インデックス・RTV/DSV ハンドルを所有する。
/// 状態は 1 つだけ追跡する（サブリソース単位の追跡はしない）。
class Texture {
public:
    Texture() = default;
    Texture(Device& device, const TextureDesc& desc);
    ~Texture();
    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    explicit operator bool() const { return m_resource != nullptr; }
    ID3D12Resource* resource() const { return m_resource.Get(); }
    const TextureDesc& desc() const { return m_desc; }
    uint32_t srvIndex() const { return m_srvIndex; }
    uint32_t uavIndex() const { return m_uavIndex; }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv() const { return m_rtv; }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv() const { return m_dsv; }

    /// 現在の状態から `to` へのバリアを記録する（同じなら何もしない）。
    void transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES to);

    /// CPU のデータ（行は密に詰めたもの）をアップロードする。完了まで待つ。
    void upload(Device& device, const void* data, uint32_t bytesPerPixel);

private:
    void release();

    Device* m_device = nullptr;
    TextureDesc m_desc{};
    ComPtr<ID3D12Resource> m_resource;
    D3D12_RESOURCE_STATES m_state = D3D12_RESOURCE_STATE_COMMON;
    uint32_t m_srvIndex = kInvalidDescriptorIndex;
    uint32_t m_uavIndex = kInvalidDescriptorIndex;
    D3D12_CPU_DESCRIPTOR_HANDLE m_rtv{};
    D3D12_CPU_DESCRIPTOR_HANDLE m_dsv{};
};

/// バッファ 1 本をコマンドリストのバリアなしで作る小さなヘルパー（アップロード・リードバック用）。
ComPtr<ID3D12Resource> createBuffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE heap,
                                    D3D12_RESOURCE_STATES initialState);

} // namespace gfx
