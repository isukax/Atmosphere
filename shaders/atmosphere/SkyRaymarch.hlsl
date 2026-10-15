// 空の単散乱をピクセルごとにレイマーチする（Phase 2 のリファレンス描画）。
//
// L_scat(視線) = ∫ T_view(0→t) · Σ_{R,M} σs(t)·p(θ) · T_sun(t) dt × E_sun
//   T_view : カメラ → サンプル点の透過率（視線に沿って光学的厚さを積算）
//   T_sun  : サンプル点 → 太陽（大気上端）の透過率。Transmittance LUT か、入れ子レイマーチで求める
//   p(θ)   : 位相関数 [sr⁻¹]、θ は視線方向と太陽方向のなす角
// 多重散乱・地表の反射・太陽ディスクは Phase 3 以降。
//
// 出力モード:
//   0: カメラ視点。L = (L/E)·E_sun を露出して fb でシーンカラーに書く
//   1: 検証用。正距円筒の方向グリッドで、単位照度あたりの輝度 L/E [sr⁻¹] を float32 で書き、
//      使った方向も別テクスチャに書く（Python が同じ方向で積分できるように）
#include "common/Bindless.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"
#include "atmosphere/Phase.hlsli"

#define SUN_TRANSMITTANCE_LUT 0
#define SUN_TRANSMITTANCE_NESTED 1

#define OUTPUT_CAMERA_FB 0
#define OUTPUT_TEST_PER_LUX 1

cbuffer SkyRaymarchConstants : register(b0)
{
    AtmosphereParameters gAtmosphere   : packoffset(c0);
    row_major float4x4 gInvViewProjRot : packoffset(c6);   // 回転のみのビュー × 射影 の逆行列
    float3 gCameraPosKm                : packoffset(c10);  // 惑星座標 [km]
    uint   gOutputUav                  : packoffset(c10.w);
    float3 gSunDirection               : packoffset(c11);  // 太陽へ向かう単位ベクトル
    uint   gTransmittanceLutSrv        : packoffset(c11.w);
    float3 gSunIlluminanceLux          : packoffset(c12);  // E_sun × 太陽の色（RGB、Y = E_sun）[lx]
    float  gExposureScale              : packoffset(c12.w); // 1/(1.2·2^EV100) [1/(cd/m²)]
    uint   gWidth                      : packoffset(c13.x);
    uint   gHeight                     : packoffset(c13.y);
    uint   gPrimarySteps               : packoffset(c13.z);
    uint   gSunTransmittanceMode       : packoffset(c13.w);
    uint   gNestedSteps                : packoffset(c14.x);
    uint   gOutputMode                 : packoffset(c14.y);
    uint   gDirectionUav               : packoffset(c14.z);
    float2 gTransmittanceLutSize       : packoffset(c15.x);
};

static const float kMaxSceneColorFb = 65000.0;

// 点 p（惑星座標 [km]）から太陽方向の透過率。太陽が地平線の下（惑星の影）なら 0。
float3 SunTransmittance(float3 p)
{
    const float r = length(p);
    const float muSun = dot(p, gSunDirection) / r;
    // [POINT] 惑星の影: 太陽へ向かう光線が地表に当たる点には直達光が届かない。
    //         太陽は点光源として扱う（視半径 0.27° の半影は無視）。
    if (RayIntersectsGround(gAtmosphere, r, muSun)) {
        return 0.0;
    }
    if (gSunTransmittanceMode == SUN_TRANSMITTANCE_LUT) {
        Texture2D<float4> lut = ResourceDescriptorHeap[gTransmittanceLutSrv];
        return SampleTransmittanceLut(gAtmosphere, lut, gLinearClamp, r, muSun, gTransmittanceLutSize);
    }
    return ComputeTransmittanceToTop(gAtmosphere, r, muSun, gNestedSteps);
}

// 単位照度あたりの単散乱輝度 L/E [sr⁻¹]
float3 IntegrateSingleScattering(float3 origin, float3 viewDir)
{
    // 視線の積分区間: 大気上端の内側から、地表に当たればそこまで。
    float tTop0;
    float tTop1;
    if (!RaySphere(origin, viewDir, gAtmosphere.topRadiusKm, tTop0, tTop1) || tTop1 <= 0.0) {
        return 0.0;  // 大気に入らない
    }
    const float tStart = max(tTop0, 0.0);
    float tEnd = tTop1;
    float tGround0;
    float tGround1;
    if (RaySphere(origin, viewDir, gAtmosphere.bottomRadiusKm, tGround0, tGround1) && tGround0 > 0.0) {
        tEnd = min(tEnd, tGround0);
    }

    const float cosTheta = dot(viewDir, gSunDirection);
    const float phaseRayleigh = RayleighPhase(cosTheta);
    const float phaseMie = CornetteShanksPhase(gAtmosphere.miePhaseG, cosTheta);

    // [POINT] 中点則: 区間を N 等分し、各区間の中点で被積分関数を評価する。
    //         カメラからサンプル点までの透過率は、前の区間までの光学的厚さ + この区間の半分。
    const float dt = (tEnd - tStart) / float(gPrimarySteps);
    float3 opticalDepth = 0.0;
    float3 luminance = 0.0;
    for (uint i = 0; i < gPrimarySteps; ++i) {
        const float t = tStart + (float(i) + 0.5) * dt;
        const float3 p = origin + t * viewDir;
        const MediumSample m = SampleMedium(gAtmosphere, length(p) - gAtmosphere.bottomRadiusKm);
        const float3 viewTransmittance = exp(-(opticalDepth + 0.5 * m.extinction * dt));
        // [POINT] 単位の確認: σs [km⁻¹] × p [sr⁻¹] × dt [km] = sr⁻¹（照度 1 lx あたりの輝度 cd/m²）
        const float3 scattering = m.rayleighScattering * phaseRayleigh + m.mieScattering * phaseMie;
        luminance += viewTransmittance * scattering * SunTransmittance(p) * dt;
        opticalDepth += m.extinction * dt;
    }
    return luminance;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];

    if (gOutputMode == OUTPUT_TEST_PER_LUX) {
        // 正距円筒の方向: 方位 φ = 2π·(x+0.5)/W、仰角 e = π/2 − π·(y+0.5)/H。+Y が天頂。
        const float phi = 2.0 * kPi * (float(id.x) + 0.5) / float(gWidth);
        const float elevation = 0.5 * kPi - kPi * (float(id.y) + 0.5) / float(gHeight);
        const float3 dir = float3(cos(elevation) * cos(phi), sin(elevation), cos(elevation) * sin(phi));
        output[id.xy] = float4(IntegrateSingleScattering(gCameraPosKm, dir), 1.0);
        RWTexture2D<float4> directions = ResourceDescriptorHeap[gDirectionUav];
        directions[id.xy] = float4(dir, 0.0);
        return;
    }

    // カメラの視線方向: ピクセル中心の NDC を、回転のみの (View·Proj)⁻¹ でワールド方向に戻す。
    // [POINT] Reverse-Z では近平面が z = 1。平行移動を含めない行列なので、
    //         戻した点がそのまま方向になる（カメラ位置 ~6360 km との差を取らずに済み、精度が落ちない）。
    const float2 ndc = float2((float(id.x) + 0.5) / float(gWidth) * 2.0 - 1.0,
                              1.0 - (float(id.y) + 0.5) / float(gHeight) * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, 1.0, 1.0));
    const float3 viewDir = normalize(h.xyz / h.w);

    const float3 luminancePerLux = IntegrateSingleScattering(gCameraPosKm, viewDir);
    // [POINT] L [cd/m²] = (L/E) [sr⁻¹] × E_sun [lx]、fb = L × k [1/(cd/m²)]。
    const float3 fb = min(luminancePerLux * gSunIlluminanceLux * gExposureScale, kMaxSceneColorFb);
    output[id.xy] = float4(fb, 1.0);
}
