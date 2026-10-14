// 表示パス: シーンカラー（fb）→ GT7 Tone Mapping → SDR / HDR10 エンコード → UI 合成。
// design.md §7.1。
//
// 単位の流れ:
//   シーンカラー     : fb（露出済み、1.0 = 100 cd/m²）、linear 作業色空間
//   → Rec.2020       : fb、linear Rec.2020
//   → GT7            : SDR [0,1]（1 = SDR 白）/ HDR fb（≤ peakNits/100）
//   → SDR            : Rec.709 にクリップ → sRGB OETF → ディザ → 8bit
//   → HDR10          : ×100 で nit → PQ → 10bit
#include "common/Bindless.hlsli"
#include "common/Noise.hlsli"
#include "display/GT7ToneMap.hlsli"
#include "display/TransferFunctions.hlsli"

#define OUTPUT_SDR_SRGB 0
#define OUTPUT_HDR10_PQ 1
#define OUTPUT_LINEAR_REC2020 2   // 検証用: GT7 の生出力をそのまま float RT に書く（UI なし）

#define TONEMAP_GT7 0
#define TONEMAP_NONE 1            // 検証用: トーンマップを通さない（PQ 単体の検証に使う）

// [POINT] cbuffer のメンバーは packoffset で位置を固定し、C++ 側（DisplayPass.cpp）の
//         static_assert(offsetof) と 1 対 1 で対応させる。float3x3 の行の余り 4 バイトに
//         次のスカラーが詰められる HLSL の規則で、暗黙のレイアウトは C++ と簡単にずれる
//         （GraphicsTest/HDR で実際に起きた不具合。症状は「色が違う」だけで検出されない）。
cbuffer DisplayConstants : register(b0)
{
    row_major float3x3 gWorkingToRec2020 : packoffset(c0);
    row_major float3x3 gRec2020ToRec709  : packoffset(c3);
    row_major float3x3 gRec709ToRec2020  : packoffset(c6);
    uint  gSceneColorIndex    : packoffset(c9.x);
    uint  gUiIndex            : packoffset(c9.y);
    uint  gOutputEncode       : packoffset(c9.z);
    uint  gToneMapOperator    : packoffset(c9.w);
    float gExposureRescale    : packoffset(c10.x);
    float gUiWhiteNits        : packoffset(c10.y);
    uint  gFrameIndex         : packoffset(c10.z);
    float gGT7TargetFb        : packoffset(c11.x);
    float gGT7SdrCorrection   : packoffset(c11.y);
    float gGT7MidPoint        : packoffset(c11.z);
    float gGT7LinearSection   : packoffset(c11.w);
    float gGT7ToeStrength     : packoffset(c12.x);
    float gGT7K               : packoffset(c12.y);
    float gGT7KA              : packoffset(c12.z);
    float gGT7TargetUcsI      : packoffset(c12.w);
    float gGT7BlendRatio      : packoffset(c13.x);
    float gGT7FadeStart       : packoffset(c13.y);
    float gGT7FadeEnd         : packoffset(c13.z);
};

GT7Params LoadGT7Params()
{
    GT7Params p;
    p.targetFb = gGT7TargetFb;
    p.sdrCorrection = gGT7SdrCorrection;
    p.midPoint = gGT7MidPoint;
    p.linearSection = gGT7LinearSection;
    p.toeStrength = gGT7ToeStrength;
    p.k = gGT7K;
    p.kA = gGT7KA;
    p.targetUcsI = gGT7TargetUcsI;
    p.blendRatio = gGT7BlendRatio;
    p.fadeStart = gGT7FadeStart;
    p.fadeEnd = gGT7FadeEnd;
    return p;
}

struct VSOut
{
    float4 pos : SV_Position;
};

// 頂点バッファなしの全画面三角形
VSOut VSMain(uint id : SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    VSOut o;
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    const int2 pixel = int2(i.pos.xy);
    Texture2D<float4> sceneColor = ResourceDescriptorHeap[gSceneColorIndex];

    // [POINT] シーンカラーは露出済み（プリエクスポーズ済み）の fb 値。手動露出では
    //         gExposureRescale = 1。将来の自動露出では「今の露出 / 書き込み時の露出」を掛ける。
    float3 sceneFb = sceneColor.Load(int3(pixel, 0)).rgb * gExposureRescale;

    // 作業色空間 → Rec.2020（Rec.2020 作業時は単位行列）。GT7 は Rec.2020 で処理する。
    float3 rec2020Fb = mul(gWorkingToRec2020, sceneFb);

    float3 mapped = gToneMapOperator == TONEMAP_GT7 ? GT7ToneMap(rec2020Fb, LoadGT7Params()) : rec2020Fb;

    if (gOutputEncode == OUTPUT_LINEAR_REC2020) {
        return float4(mapped, 1.0);
    }

    // UI テクスチャ: ImGui が R8G8B8A8_UNORM に sRGB 値で描いたもの。
    // [POINT] ImGui の描画は (SrcAlpha, InvSrcAlpha) ブレンドで透明黒の上に重ねるので、
    //         RGB は「アルファ乗算済み」、A は被覆率になっている。
    float4 ui = float4(0.0, 0.0, 0.0, 0.0);
    if (gUiIndex != kInvalidDescriptorIndex) {
        Texture2D<float4> uiTexture = ResourceDescriptorHeap[gUiIndex];
        ui = uiTexture.Load(int3(pixel, 0));
    }

    if (gOutputEncode == OUTPUT_HDR10_PQ) {
        // [POINT] GT7 の HDR 出力は fb（1.0 = 100 nit）なので ×100 で表示輝度 [nit] にする。
        float3 sceneNits = mapped * 100.0;
        // UI は sRGB 値 → linear → 「UI の白 = gUiWhiteNits」として nit にし、Rec.2020 で合成する。
        float3 uiLinear709 = ui.a > 0.0 ? SrgbDecode(ui.rgb / ui.a) * ui.a : 0.0;
        float3 uiNits = mul(gRec709ToRec2020, uiLinear709) * gUiWhiteNits;
        float3 nits = sceneNits * (1.0 - ui.a) + uiNits;
        return float4(PQEncodeNits(nits), 1.0);
    }

    // SDR: GT7 の出力（[0,1]、Rec.2020）→ Rec.709 → 範囲外をクリップ → sRGB
    // [POINT] Rec.2020 でしか表せない高彩度は Rec.709 の SDR では表示できないのでクリップされる。
    //         これは表示側の制約で、作業色空間やトーンマップの不具合ではない。
    float3 linear709 = saturate(mul(gRec2020ToRec709, mapped));
    float3 encoded = SrgbEncode(linear709);
    // [POINT] 8bit 量子化の直前に ±1 LSB の三角分布ディザを加える（空のグラデーションの縞対策）。
    const float lsb = 1.0 / 255.0;
    encoded += lsb * float3(TriangularNoise(uint2(pixel), gFrameIndex, 0),
                            TriangularNoise(uint2(pixel), gFrameIndex, 1),
                            TriangularNoise(uint2(pixel), gFrameIndex, 2));
    // UI はアルファ乗算済みの sRGB 値なので、sRGB 値のまま重ねる（ImGui の想定どおりの見た目）。
    encoded = ui.rgb + encoded * (1.0 - ui.a);
    return float4(saturate(encoded), 1.0);
}
