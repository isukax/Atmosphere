#pragma once

#include "gfx/Common.h"

#include <string>

namespace display {

/// ウィンドウが表示されているディスプレイの状態（OS の報告値）。
struct DisplayState {
    bool valid = false;
    std::string deviceName;          // \\.\DISPLAY1 など
    bool hdrEnabled = false;         // OS の「HDR を使用する」が ON か
    DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    uint32_t bitsPerColor = 0;
    float minLuminanceNits = 0.0f;
    float maxLuminanceNits = 0.0f;           // 小領域のピーク輝度
    float maxFullFrameLuminanceNits = 0.0f;  // 全画面の最大輝度

    bool operator==(const DisplayState&) const = default;
};

/// [POINT] ウィンドウの矩形と最も重なる出力（モニター）を探し、IDXGIOutput6::GetDesc1 で
///         HDR 状態とピーク輝度を読む。スワップチェーンの GetContainingOutput は
///         スワップチェーン作成前や古いファクトリでは使えないため、この方法で統一する
///         （Microsoft D3D12HDR サンプルと同じ方式）。
///         `factory` は IsCurrent() == true の最新のものを渡すこと（Device::refreshFactoryIfStale）。
DisplayState queryDisplayState(IDXGIFactory6* factory, HWND window);

const char* toString(DXGI_COLOR_SPACE_TYPE colorSpace);

} // namespace display
