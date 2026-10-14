// Phase 1 のテストパターン: 物理輝度 [cd/m²] の対数ランプを露出して fb でシーンカラーに書く。
//
// 帯（上から）: 無彩色 / 作業色空間の R 原色 / G 原色 / B 原色。
// 各帯の左端→右端で輝度 L を 10^log10MinNits 〜 10^log10MaxNits まで対数で変える。
// 帯の上部 15% には 1 桁（10 倍）ごとに目盛り線（黒）を描く。
#include "common/Bindless.hlsli"

cbuffer TestPatternConstants : register(b0)
{
    uint   gOutputIndex      : packoffset(c0.x);
    uint   gWidth            : packoffset(c0.y);
    uint   gHeight           : packoffset(c0.z);
    float  gExposureScale    : packoffset(c1.x);  // 1/(1.2·2^EV100) [1/(cd/m²)]
    float  gLog10MinNits     : packoffset(c1.y);
    float  gLog10MaxNits     : packoffset(c1.z);
    float3 gLuminanceWeights : packoffset(c2);    // 作業色空間の Y 係数
};

// [POINT] FP16 (R16G16B16A16_FLOAT) の最大値は 65504。露出後でも太陽ディスクなどは超えうるので、
//         書き込み前に明示的にクランプして inf を作らない（inf は後段で NaN を生む）。
static const float kMaxSceneColorFb = 65000.0;

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputIndex];

    const float t = float(id.x) / float(max(gWidth - 1, 1));
    const float log10Nits = lerp(gLog10MinNits, gLog10MaxNits, t);
    // [POINT] これが物理輝度 L [cd/m²]（= nit）。シーン側の値はすべてこの単位で作る。
    const float luminanceNits = pow(10.0, log10Nits);

    const uint band = min(id.y * 4 / gHeight, 3u);
    const float bandHeight = float(gHeight) / 4.0;
    const float yInBand = float(id.y) - float(band) * bandHeight;

    // 帯ごとの色。原色の帯は「その原色だけで輝度が L になる」ように Y 係数で割る。
    float3 rgbNits;
    if (band == 0) {
        rgbNits = luminanceNits.xxx;  // Y 係数の和は 1 なので、無彩色は (L, L, L) で輝度 L
    } else {
        rgbNits = 0.0;
        rgbNits[band - 1] = luminanceNits / gLuminanceWeights[band - 1];
    }

    // [POINT] 露出: fb = L [cd/m²] × k [1/(cd/m²)]、k = 1/(1.2·2^EV100)。
    //         1.2·2^EV100 [cd/m²] の輝度が fb 1.0（= 表示 100 nit、GT7 の線形区間）になる。
    float3 fb = min(rgbNits * gExposureScale, kMaxSceneColorFb);

    // 1 桁（10 倍）ごとの目盛り線（帯の上部のみ）。背景が暗ければ白（fb 1.0）、明るければ黒。
    const float pixelsPerDecade = float(gWidth) / max(gLog10MaxNits - gLog10MinNits, 1e-6);
    const float distanceToDecadePx = abs(log10Nits - round(log10Nits)) * pixelsPerDecade;
    if (yInBand < bandHeight * 0.15 && distanceToDecadePx < 1.0) {
        fb = dot(fb, gLuminanceWeights) < 0.5 ? 1.0 : 0.0;
    }
    // 帯の境界に 2px の黒線
    if (yInBand < 2.0) {
        fb = 0.0;
    }
    output[id.xy] = float4(fb, 1.0);
}
