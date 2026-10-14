#pragma once

#include "gfx/Device.h"

namespace display {

/// スワップチェーンの出力モード。design.md D6。
enum class OutputMode {
    SDR,    // R8G8B8A8_UNORM    + DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
    HDR10,  // R10G10B10A2_UNORM + DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
};

const char* toString(OutputMode mode);
DXGI_FORMAT formatOf(OutputMode mode);

/// フリップモデルのスワップチェーン。モード（形式と色空間）の切替とリサイズを行う。
class Swapchain {
public:
    static constexpr uint32_t kBufferCount = 3;

    Swapchain(gfx::Device& device, HWND window, uint32_t width, uint32_t height, OutputMode mode);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    /// モードを変える。GPU が空であること（呼び出し側で waitIdle）。
    /// 戻り値: 要求した色空間が受け入れられたか（false なら SDR に戻している）。
    bool setMode(OutputMode mode);
    /// サイズを変える。GPU が空であること。
    void resize(uint32_t width, uint32_t height);
    /// HDR10 のメタデータ（マスタリング表示の輝度など）を設定する。HDR10 モードでのみ意味がある。
    void setHdrMetadata(float peakNits, float maxFrameAverageNits);

    OutputMode mode() const { return m_mode; }
    DXGI_FORMAT format() const { return formatOf(m_mode); }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }

    ID3D12Resource* backBuffer() const;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv() const;
    /// 現在のバックバッファを `to` へ遷移させる。
    void transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES to);
    void present(bool vsync);

private:
    void createBuffers();
    void releaseBuffers();
    bool applyColorSpace();

    gfx::Device& m_device;
    HWND m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    OutputMode m_mode = OutputMode::SDR;
    ComPtr<IDXGISwapChain4> m_swapchain;
    ComPtr<ID3D12Resource> m_buffers[kBufferCount];
    D3D12_CPU_DESCRIPTOR_HANDLE m_rtvs[kBufferCount]{};
    D3D12_RESOURCE_STATES m_states[kBufferCount]{};
};

} // namespace display
