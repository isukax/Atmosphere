#include "display/DisplayMonitor.h"

#include <algorithm>

namespace display {

const char* toString(DXGI_COLOR_SPACE_TYPE colorSpace)
{
    switch (colorSpace) {
    case DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709: return "sRGB (G22 P709)";
    case DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709: return "scRGB (G10 P709)";
    case DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020: return "HDR10 (PQ P2020)";
    default: return "other";
    }
}

DisplayState queryDisplayState(IDXGIFactory6* factory, HWND window)
{
    RECT windowRect{};
    GetWindowRect(window, &windowRect);

    DisplayState best;
    long bestArea = -1;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        ComPtr<IDXGIOutput> output;
        for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
            ComPtr<IDXGIOutput6> output6;
            if (FAILED(output.As(&output6))) {
                continue;
            }
            DXGI_OUTPUT_DESC1 desc{};
            if (FAILED(output6->GetDesc1(&desc))) {
                continue;
            }
            const RECT& r = desc.DesktopCoordinates;
            const long w = std::max(0L, std::min(windowRect.right, r.right) - std::max(windowRect.left, r.left));
            const long h = std::max(0L, std::min(windowRect.bottom, r.bottom) - std::max(windowRect.top, r.top));
            if (w * h <= bestArea) {
                continue;
            }
            bestArea = w * h;
            best.valid = true;
            best.deviceName = gfx::narrow(desc.DeviceName);
            best.colorSpace = desc.ColorSpace;
            // [POINT] OS で HDR が有効なとき、出力の色空間は PQ / Rec.2020 と報告される。
            best.hdrEnabled = desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            best.bitsPerColor = desc.BitsPerColor;
            best.minLuminanceNits = desc.MinLuminance;
            best.maxLuminanceNits = desc.MaxLuminance;
            best.maxFullFrameLuminanceNits = desc.MaxFullFrameLuminance;
        }
    }
    return best;
}

} // namespace display
