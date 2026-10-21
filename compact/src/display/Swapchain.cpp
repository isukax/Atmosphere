#include "display/Swapchain.h"

#include <algorithm>

namespace compact::display {

const char* toString(OutputMode mode)
{
    return mode == OutputMode::SDR ? "SDR" : "HDR10";
}

DXGI_FORMAT formatOf(OutputMode mode)
{
    return mode == OutputMode::SDR ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R10G10B10A2_UNORM;
}

namespace {

DXGI_COLOR_SPACE_TYPE colorSpaceOf(OutputMode mode)
{
    return mode == OutputMode::SDR ? DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
                                   : DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
}

} // namespace

Swapchain::Swapchain(gfx::Device& device, HWND window, uint32_t width, uint32_t height, OutputMode mode)
    : m_device(device), m_window(window), m_width(width), m_height(height), m_mode(mode)
{
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = formatOf(mode);
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = kBufferCount;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    ComPtr<IDXGISwapChain1> swapchain1;
    gfx::check(device.factory()->CreateSwapChainForHwnd(device.queue(), window, &desc, nullptr, nullptr, &swapchain1),
               "CreateSwapChainForHwnd");
    gfx::check(swapchain1.As(&m_swapchain), "IDXGISwapChain4");
    device.factory()->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER);
    createBuffers();
    if (!applyColorSpace()) {
        setMode(OutputMode::SDR);
    }
}

Swapchain::~Swapchain()
{
    releaseBuffers();
}

void Swapchain::createBuffers()
{
    for (uint32_t i = 0; i < kBufferCount; ++i) {
        gfx::check(m_swapchain->GetBuffer(i, IID_PPV_ARGS(&m_buffers[i])), "GetBuffer");
        m_rtvs[i] = m_device.rtvPool().allocate();
        m_device.d3d()->CreateRenderTargetView(m_buffers[i].Get(), nullptr, m_rtvs[i]);
        m_states[i] = D3D12_RESOURCE_STATE_PRESENT;
    }
}

void Swapchain::releaseBuffers()
{
    for (uint32_t i = 0; i < kBufferCount; ++i) {
        m_buffers[i].Reset();
        m_device.rtvPool().free(m_rtvs[i]);
        m_rtvs[i] = {};
    }
}

bool Swapchain::applyColorSpace()
{
    // [POINT] SetColorSpace1 はスワップチェーンの内容をどう解釈するかを OS に伝える。
    //         HDR10 のバッファに PQ を書いても、この指定がないと OS は SDR(sRGB) として扱う。
    //         OS の HDR が OFF のディスプレイでは PQ 色空間がサポートされず、ここで失敗する。
    const DXGI_COLOR_SPACE_TYPE colorSpace = colorSpaceOf(m_mode);
    UINT support = 0;
    if (FAILED(m_swapchain->CheckColorSpaceSupport(colorSpace, &support)) ||
        !(support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
        gfx::log(std::string("Color space not supported for present: ") + toString(m_mode));
        return false;
    }
    return SUCCEEDED(m_swapchain->SetColorSpace1(colorSpace));
}

bool Swapchain::setMode(OutputMode mode)
{
    const OutputMode requested = mode;
    m_mode = mode;
    releaseBuffers();
    // [POINT] フリップモデルでは ResizeBuffers で形式も変えられる（スワップチェーンの作り直しは不要）。
    //         バックバッファへの参照はすべて解放してから呼ぶ必要がある。
    gfx::check(m_swapchain->ResizeBuffers(kBufferCount, m_width, m_height, formatOf(mode), 0), "ResizeBuffers");
    createBuffers();
    if (applyColorSpace()) {
        return true;
    }
    if (requested == OutputMode::SDR) {
        return false;
    }
    // HDR10 が使えないので SDR に戻す
    setMode(OutputMode::SDR);
    return false;
}

void Swapchain::resize(uint32_t width, uint32_t height)
{
    m_width = std::max(width, 1u);
    m_height = std::max(height, 1u);
    releaseBuffers();
    gfx::check(m_swapchain->ResizeBuffers(kBufferCount, m_width, m_height, formatOf(m_mode), 0), "ResizeBuffers");
    createBuffers();
    applyColorSpace();
}

void Swapchain::setHdrMetadata(float peakNits, float maxFrameAverageNits)
{
    if (m_mode != OutputMode::HDR10) {
        m_swapchain->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, 0, nullptr);
        return;
    }
    // [POINT] 単位: 原色・白色点の xy は 0.00002 刻み（×50000）、MaxMasteringLuminance と
    //         MaxContentLightLevel は nit、MinMasteringLuminance は 0.0001 nit 刻み。
    //         GT7 の出力はピーク輝度でクリップされるので、コンテンツの最大輝度 = ピーク。
    DXGI_HDR_METADATA_HDR10 metadata{};
    metadata.RedPrimary[0] = static_cast<UINT16>(0.708 * 50000);
    metadata.RedPrimary[1] = static_cast<UINT16>(0.292 * 50000);
    metadata.GreenPrimary[0] = static_cast<UINT16>(0.170 * 50000);
    metadata.GreenPrimary[1] = static_cast<UINT16>(0.797 * 50000);
    metadata.BluePrimary[0] = static_cast<UINT16>(0.131 * 50000);
    metadata.BluePrimary[1] = static_cast<UINT16>(0.046 * 50000);
    metadata.WhitePoint[0] = static_cast<UINT16>(0.3127 * 50000);
    metadata.WhitePoint[1] = static_cast<UINT16>(0.3290 * 50000);
    metadata.MaxMasteringLuminance = static_cast<UINT>(peakNits);
    metadata.MinMasteringLuminance = 0;
    metadata.MaxContentLightLevel = static_cast<UINT16>(std::min(peakNits, 65535.0f));
    metadata.MaxFrameAverageLightLevel = static_cast<UINT16>(std::min(maxFrameAverageNits, 65535.0f));
    m_swapchain->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(metadata), &metadata);
}

ID3D12Resource* Swapchain::backBuffer() const
{
    return m_buffers[m_swapchain->GetCurrentBackBufferIndex()].Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE Swapchain::rtv() const
{
    return m_rtvs[m_swapchain->GetCurrentBackBufferIndex()];
}

void Swapchain::transition(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES to)
{
    const UINT index = m_swapchain->GetCurrentBackBufferIndex();
    if (m_states[index] == to) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_buffers[index].Get();
    barrier.Transition.StateBefore = m_states[index];
    barrier.Transition.StateAfter = to;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &barrier);
    m_states[index] = to;
}

void Swapchain::present(bool vsync)
{
    gfx::check(m_swapchain->Present(vsync ? 1 : 0, 0), "Present");
}

} // namespace display
