// Bindless 共通定義。
//
// [POINT] ルートシグネチャ（src/gfx/Pipeline.cpp）は
//           b0 … パスごとの定数バッファ（ルート CBV）
//           s0 … 線形/クランプ、s1 … ポイント/クランプ（静的サンプラー）
//         だけを持つ。テクスチャは定数バッファで受け取ったインデックスで
//         `ResourceDescriptorHeap[index]` から取り出す（SM 6.6）。
#ifndef BINDLESS_HLSLI
#define BINDLESS_HLSLI

static const uint kInvalidDescriptorIndex = 0xFFFFFFFFu;

SamplerState gLinearClamp : register(s0);
SamplerState gPointClamp : register(s1);

#endif // BINDLESS_HLSLI
