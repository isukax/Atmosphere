// 空を描くパス（Sky-View LUT 生成・LUT からの描画・リファレンスのレイマーチ・パストレーサー）共通の定数。
// C++ 側は src/atmosphere/SkyPassConstants.h（static_assert で packoffset と照合）。
#ifndef SKY_PASS_CONSTANTS_HLSLI
#define SKY_PASS_CONSTANTS_HLSLI

#include "atmosphere/AtmosphereIntegrate.hlsli"

#define OUTPUT_CAMERA_FB 0     // カメラ視点。露出済み fb をシーンカラーに書く
#define OUTPUT_TEST_PER_LUX 1  // 検証用。正距円筒の方向グリッドで、単位照度あたりの輝度 [sr⁻¹] を書く

#define SKY_FLAG_MULTIPLE_SCATTERING 1u
#define SKY_FLAG_SUN_DISK 2u

cbuffer SkyPassConstants : register(b0)
{
    AtmosphereParameters gAtmosphere   : packoffset(c0);   // c0-c5
    row_major float4x4 gInvViewProjRot : packoffset(c6);   // 回転のみのビュー × 射影 の逆行列（Reverse-Z）
    float3 gCameraPosKm                : packoffset(c10);  // 惑星座標 [km]
    uint   gOutputUav                  : packoffset(c10.w);
    float3 gSunDirection               : packoffset(c11);  // 太陽へ向かう単位ベクトル
    uint   gTransmittanceLutSrv        : packoffset(c11.w);
    float3 gSunIlluminanceLux          : packoffset(c12);  // E_sun × 太陽の色（RGB、Y = E_sun）[lx]
    float  gExposureScale              : packoffset(c12.w); // 1/(1.2·2^EV100) [1/(cd/m²)]
    uint   gWidth                      : packoffset(c13.x);
    uint   gHeight                     : packoffset(c13.y);
    uint   gStepCount                  : packoffset(c13.z);
    uint   gSunTransmittanceMode       : packoffset(c13.w);
    uint   gNestedSteps                : packoffset(c14.x);
    uint   gOutputMode                 : packoffset(c14.y);
    uint   gDirectionUav               : packoffset(c14.z);
    uint   gMultiScatteringLutSrv      : packoffset(c14.w);
    float2 gTransmittanceLutSize       : packoffset(c15.x);
    float2 gMultiScatteringLutSize     : packoffset(c15.z);
    uint   gSkyViewLutSrv              : packoffset(c16.x);
    uint   gFlags                      : packoffset(c16.y); // SKY_FLAG_*
    float2 gSkyViewLutSize             : packoffset(c16.z);
    uint   gFrameIndex                 : packoffset(c17.x); // パストレーサー: 乱数の種・累積の振り分け
    uint   gMaxDepth                   : packoffset(c17.y); // パストレーサー: 最大散乱回数
    uint   gSamplesPerDispatch         : packoffset(c17.z); // パストレーサー: 1 回のディスパッチで追加するサンプル数
    uint   gAccumulationUavA           : packoffset(c17.w); // パストレーサー: 累積（偶数フレーム）
    uint   gAccumulationUavB           : packoffset(c18.x); // パストレーサー: 累積（奇数フレーム）
    uint   gSkyViewMieLutUav           : packoffset(c18.y); // Sky-View LUT の生成: Mie の単散乱（位相関数なし）の出力
    uint   gSkyViewMieLutSrv           : packoffset(c18.z); // LUT からの描画: 同じ LUT を読む
};

AtmosphereLuts LoadAtmosphereLuts()
{
    AtmosphereLuts luts;
    luts.transmittanceSrv = gTransmittanceLutSrv;
    luts.transmittanceSize = gTransmittanceLutSize;
    luts.multiScatteringSrv = gMultiScatteringLutSrv;
    luts.multiScatteringSize = gMultiScatteringLutSize;
    return luts;
}

// カメラの視線方向: ピクセル中心の NDC を、回転のみの (View·Proj)⁻¹ でワールド方向に戻す。
// [POINT] Reverse-Z では近平面が z = 1。平行移動を含めない行列なので、戻した点がそのまま方向になる
//         （カメラ位置 ~6360 km との差を取らずに済み、精度が落ちない）。
float3 CameraViewDirection(uint2 pixel)
{
    const float2 ndc = float2((float(pixel.x) + 0.5) / float(gWidth) * 2.0 - 1.0,
                              1.0 - (float(pixel.y) + 0.5) / float(gHeight) * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, 1.0, 1.0));
    return normalize(h.xyz / h.w);
}

// 出力ピクセルの視線方向（カメラ視点 or 検証用の正距円筒グリッド）
float3 OutputViewDirection(uint2 pixel)
{
    return gOutputMode == OUTPUT_TEST_PER_LUX ? EquirectDirection(pixel, uint2(gWidth, gHeight))
                                              : CameraViewDirection(pixel);
}

// [POINT] FP16 のシーンカラーに書く前のクランプ（太陽ディスクは露出後も 65504 を超えうる）
static const float kMaxSceneColorFb = 65000.0;

// 単位照度あたりの輝度 [sr⁻¹] を出力する。カメラ視点なら L = (L/E)·E_sun を露出して fb で書く。
void WriteSkyOutput(uint2 pixel, float3 luminancePerLux, float3 viewDir)
{
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    if (gOutputMode == OUTPUT_TEST_PER_LUX) {
        output[pixel] = float4(luminancePerLux, 1.0);
        if (gDirectionUav != kInvalidDescriptorIndex) {
            RWTexture2D<float4> directions = ResourceDescriptorHeap[gDirectionUav];
            directions[pixel] = float4(viewDir, 0.0);
        }
        return;
    }
    // [POINT] L [cd/m²] = (L/E) [sr⁻¹] × E_sun [lx]、fb = L × k [1/(cd/m²)]
    output[pixel] = float4(min(luminancePerLux * gSunIlluminanceLux * gExposureScale, kMaxSceneColorFb), 1.0);
}

// 太陽ディスク: 視線が太陽の視半径の内側なら、L/E = T_view / Ω_sun [sr⁻¹]（Ω_sun = 2π(1 − cos α)）。
// [POINT] 太陽の放射輝度は E_sun / Ω_sun（大気上端、周縁減光なし）。大気を通った分だけ T_view で減衰する。
//         Sky-View LUT は解像度が低く、太陽（直径 0.54°）を表せないので、ここで解析的に足す。
float3 SunDiskPerLux(float3 cameraPosKm, float3 viewDir)
{
    const float cosAlpha = cos(gAtmosphere.sunAngularRadiusRad);
    if (dot(viewDir, gSunDirection) < cosAlpha) {
        return 0.0;
    }
    const float r = length(cameraPosKm);
    const float mu = dot(cameraPosKm, viewDir) / r;
    if (RayIntersectsGround(gAtmosphere, r, mu)) {
        return 0.0;
    }
    Texture2D<float4> lut = ResourceDescriptorHeap[gTransmittanceLutSrv];
    const float3 transmittance = SampleTransmittanceLut(gAtmosphere, lut, gLinearClamp, r, mu, gTransmittanceLutSize);
    const float solidAngleSr = 2.0 * kPi * (1.0 - cosAlpha);
    return transmittance / solidAngleSr;
}

#endif // SKY_PASS_CONSTANTS_HLSLI
