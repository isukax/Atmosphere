// 空を描くパス（Sky-View LUT 生成・LUT からの描画・リファレンスのレイマーチ・パストレーサー）共通の定数。
// C++ 側は src/atmosphere/SkyPassConstants.h（static_assert で packoffset と照合）。
#ifndef SKY_PASS_CONSTANTS_HLSLI
#define SKY_PASS_CONSTANTS_HLSLI

#include "atmosphere/AtmosphereIntegrate.hlsli"

#define OUTPUT_CAMERA_FB 0     // カメラ視点。露出済み fb をシーンカラーに書く
#define OUTPUT_TEST_PER_LUX 1  // 検証用。正距円筒の方向グリッドで、単位照度あたりの輝度 [sr⁻¹] を書く

#define SKY_FLAG_MULTIPLE_SCATTERING 1u
#define SKY_FLAG_SUN_DISK 2u
#define SKY_FLAG_WRITE_AP_TRANSMITTANCE 4u  // 検証用: 地形のピクセルに合成の代わりに透過率（RGB）を書く（V3h）

// 空気遠近法の透過率の求め方（design.md §5.7）
#define AP_TRANSMITTANCE_VOLUME_MEAN 0  // 0: ボリュームの A（RGB の平均。論文・参照実装と同じ近似）
#define AP_TRANSMITTANCE_LUT_RATIO 1    // 1: Transmittance LUT の比（RGB）
#define AP_TRANSMITTANCE_VOLUME_RGB 2   // 2: ボリュームの RGB の透過率（もう 1 枚の 3D テクスチャ）

// 空気遠近法の影の方式（design.md §5.6）
#define AP_SHADOW_VOLUME_POINT 0     // 0: ボリュームの積分で 1 回の比較
#define AP_SHADOW_VOLUME_FILTERED 1  // 1: ボリュームの積分で、ボクセルの横幅の円盤をぼかして引く
#define AP_SHADOW_PER_PIXEL 2        // 5: ボリュームは影なし。合成でピクセルごとに影の区間を差し引く

cbuffer SkyPassConstants : register(b0)
{
    AtmosphereParameters gAtmosphere   : packoffset(c0);   // c0-c9
    row_major float4x4 gInvViewProjRot : packoffset(c10);   // 回転のみのビュー × 射影 の逆行列（Reverse-Z）
    float3 gCameraPosKm                : packoffset(c14);  // 惑星座標 [km]
    uint   gOutputUav                  : packoffset(c14.w);
    float3 gSunDirection               : packoffset(c15);  // 太陽へ向かう単位ベクトル
    uint   gTransmittanceLutSrv        : packoffset(c15.w);
    float3 gSunIlluminanceLux          : packoffset(c16);  // E_sun × 太陽の色（RGB、Y = E_sun）[lx]
    float  gExposureScale              : packoffset(c16.w); // 1/(1.2·2^EV100) [1/(cd/m²)]
    uint   gWidth                      : packoffset(c17.x);
    uint   gHeight                     : packoffset(c17.y);
    uint   gStepCount                  : packoffset(c17.z);
    uint   gSunTransmittanceMode       : packoffset(c17.w);
    uint   gNestedSteps                : packoffset(c18.x);
    uint   gOutputMode                 : packoffset(c18.y);
    uint   gDirectionUav               : packoffset(c18.z);
    uint   gMultiScatteringLutSrv      : packoffset(c18.w);
    float2 gTransmittanceLutSize       : packoffset(c19.x);
    float2 gMultiScatteringLutSize     : packoffset(c19.z);
    uint   gSkyViewLutSrv              : packoffset(c20.x);
    uint   gFlags                      : packoffset(c20.y); // SKY_FLAG_*
    float2 gSkyViewLutSize             : packoffset(c20.z);
    uint   gFrameIndex                 : packoffset(c21.x); // パストレーサー: 乱数の種・累積の振り分け
    uint   gMaxDepth                   : packoffset(c21.y); // パストレーサー: 最大散乱回数
    uint   gSamplesPerDispatch         : packoffset(c21.z); // パストレーサー: 1 回のディスパッチで追加するサンプル数
    uint   gAccumulationUavA           : packoffset(c21.w); // パストレーサー: 累積（偶数フレーム）
    uint   gAccumulationUavB           : packoffset(c22.x); // パストレーサー: 累積（奇数フレーム）
    uint   gSkyViewMieLutUav           : packoffset(c22.y); // Sky-View LUT の生成: Mie の単散乱（位相関数なし）の出力
    uint   gSkyViewMieLutSrv           : packoffset(c22.z); // LUT からの描画: 同じ LUT を読む
    uint   gApTransmittanceMode        : packoffset(c22.w); // AP_TRANSMITTANCE_*
    uint   gAerialPerspectiveTransmittanceUav : packoffset(c23.x); // 空気遠近法のボリューム（透過率 RGB、§5.7 の方式 2）
    uint   gAerialPerspectiveTransmittanceSrv : packoffset(c23.y);
    uint   gPathTraceMediumSrv         : packoffset(c23.z); // パストレーサーの表の媒質（§5.8）。無効値なら解析式
    uint   gSkyIrradianceUav           : packoffset(c23.w); // 空からの照度の SH 係数（9 × float4）の出力
    uint   gAerialPerspectiveUav       : packoffset(c24.x); // 空気遠近法のボリューム（Rayleigh + 多重散乱、A = 平均透過率）
    uint   gAerialPerspectiveMieUav    : packoffset(c24.y); // 空気遠近法のボリューム（Mie の単散乱、位相関数なし）
    uint   gAerialPerspectiveSrv       : packoffset(c24.z);
    uint   gAerialPerspectiveMieSrv    : packoffset(c24.w);
    uint   gDepthSrv                   : packoffset(c25.x); // 深度バッファ（Reverse-Z）。無効値なら地形なし
    float  gApKmPerSlice               : packoffset(c25.y); // 空気遠近法のスライスの距離単位 [km]（参照実装と同じ 4）
    float  gApSliceCount               : packoffset(c25.z); // 32
    uint   gRaymarchMeanTransmittance  : packoffset(c25.w); // レイマーチで地形に掛ける透過率を RGB 平均にする（V3c の比較用）
    row_major float4x4 gShadowFarToUvz : packoffset(c26);   // カメラ相対の位置 [m] → 全体のカスケードの (u, v, 深度)
    uint   gShadowFarSrv               : packoffset(c30.x); // 地形の影（全体のカスケード）。無効値なら影なし（design.md §5.5）
    uint   gShadowNearSrv              : packoffset(c30.y); // 近景のカスケード。無効値なら全体だけ
    uint   gApShadowMethod             : packoffset(c30.z); // AP_SHADOW_*
    uint   gApPerPixelSteps            : packoffset(c30.w); // 方式 5: 視線上で影を調べる点の数
    row_major float4x4 gShadowNearToUvz : packoffset(c31);  // カメラ相対の位置 [m] → 近景のカスケードの (u, v, 深度)
    float  gApShadowFilterScale        : packoffset(c35.x); // 方式 1: 円盤の半径の倍率（1 = ボクセルの半幅）
    uint   gApPerPixelBisectSteps      : packoffset(c35.y); // 方式 5: 影の境界の二分探索の回数
    uint   gAerialPerspectiveMsUav     : packoffset(c35.z); // 空気遠近法のボリューム（多重散乱）
    uint   gAerialPerspectiveMsSrv     : packoffset(c35.w);
    // Mie の成分 1（2 成分のプリセットだけ。design.md §5.9）: Sky-View の Mie LUT・空気遠近法の Mie ボリュームの 2 枚目
    uint   gSkyViewMie1LutUav          : packoffset(c36.x);
    uint   gSkyViewMie1LutSrv          : packoffset(c36.y);
    uint   gAerialPerspectiveMie1Uav   : packoffset(c36.z);
    uint   gAerialPerspectiveMie1Srv   : packoffset(c36.w);
    // 多重散乱の SH 近似（MS_SH、design.md §5.11）: Mie の成分 0 の χ_0〜χ_2（c37〜c39）、成分 1 の χ_0〜χ_2（c40〜c42）
    float4 gMsPhaseMoments[6]          : packoffset(c37);
};

// 定数の地形の影（2 段のカスケード）
OpaqueShadowParameters LoadOpaqueShadow()
{
    OpaqueShadowParameters p;
    p.farSrv = gShadowFarSrv;
    p.nearSrv = gShadowNearSrv;
    p.farToUvz = gShadowFarToUvz;
    p.nearToUvz = gShadowNearToUvz;
    return p;
}

// 空気遠近法・地形のピクセルのレイマーチ: 定数の地形の影を積分に使う（無効値なら影なし。1 回の比較）
void SetOpaqueShadowFromConstants(inout IntegrationSettings s)
{
    s.opaqueShadow = LoadOpaqueShadow();
    s.opaqueShadowFilterRadiusPerM = 0.0;
}

AtmosphereLuts LoadAtmosphereLuts()
{
    AtmosphereLuts luts;
    luts.transmittanceSrv = gTransmittanceLutSrv;
    luts.transmittanceSize = gTransmittanceLutSize;
    luts.multiScatteringSrv = gMultiScatteringLutSrv;
    luts.multiScatteringSize = gMultiScatteringLutSize;
#if MS_SH
    [unroll] for (uint l = 0; l < 3; ++l) {
        luts.mieChi[l] = gMsPhaseMoments[l].rgb;
        luts.mie1Chi[l] = gMsPhaseMoments[3 + l].rgb;
    }
#endif
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

// Sky-View LUT（2 枚、2 成分なら 3 枚）を、カメラ位置から見た方向 viewDir で引く。
//   rayleighAndMs [sr⁻¹]  : Rayleigh の単散乱 + 多重散乱（単位照度あたり）
//   mieNoPhase    [無次元]: Mie の単散乱から位相関数を除いたもの（× p_M(θ) で輝度）
//   mie1NoPhase   [無次元]: Mie の成分 1 の同じ量（1 成分なら 0）
void SampleSkyViewLuts(float3 cameraPosKm, float3 viewDir, out float3 rayleighAndMs, out float3 mieNoPhase,
                       out float3 mie1NoPhase)
{
    const float r = length(cameraPosKm);
    const float3 up = cameraPosKm / r;
    const float viewZenithCos = dot(viewDir, up);

    // [POINT] 視線と太陽の「方位差の余弦」を、カメラの水平面上で求める（参照実装と同じ）。
    //         視線が真上・真下だと方位が定まらないが、その方向では値が方位に依存しないので 1 とする。
    float lightViewCos = 1.0;
    const float3 side = cross(up, viewDir);
    if (dot(side, side) > 1e-12) {
        const float3 sideN = normalize(side);
        const float3 forward = normalize(cross(sideN, up));  // 視線の水平成分の向き
        const float2 lightOnPlane = float2(dot(gSunDirection, forward), dot(gSunDirection, sideN));
        const float len = length(lightOnPlane);
        lightViewCos = len > 1e-6 ? lightOnPlane.x / len : 1.0;
    }

    float t0;
    float t1;
    const bool intersectGround = RaySphere(cameraPosKm, viewDir, gAtmosphere.bottomRadiusKm, t0, t1) && t0 > 0.0;
    const float lutR = max(r, gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);
    const float2 uv = SkyViewViewParamsToUv(gAtmosphere, intersectGround, viewZenithCos, lightViewCos, lutR,
                                            gSkyViewLutSize);
    Texture2D<float4> skyView = ResourceDescriptorHeap[gSkyViewLutSrv];
    Texture2D<float4> skyViewMie = ResourceDescriptorHeap[gSkyViewMieLutSrv];
    rayleighAndMs = skyView.SampleLevel(gLinearClamp, uv, 0.0).rgb;
    mieNoPhase = skyViewMie.SampleLevel(gLinearClamp, uv, 0.0).rgb;
    mie1NoPhase = 0.0;
#if MIE_GENERAL
    if (gAtmosphere.mieComponentCount > 1) {
        Texture2D<float4> skyViewMie1 = ResourceDescriptorHeap[gSkyViewMie1LutSrv];
        mie1NoPhase = skyViewMie1.SampleLevel(gLinearClamp, uv, 0.0).rgb;
    }
#endif
}

// 深度バッファの値（Reverse-Z、0 = 無限遠）から、カメラ → その点の距離 [km]。
// [POINT] 回転のみの (View·Proj)⁻¹ で戻すと、カメラ相対の位置 [m] がそのまま得られる（平行移動を含めないため）。
float DepthToDistanceKm(uint2 pixel, float depth)
{
    const float2 ndc = float2((float(pixel.x) + 0.5) / float(gWidth) * 2.0 - 1.0,
                              1.0 - (float(pixel.y) + 0.5) / float(gHeight) * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, depth, 1.0));
    return length(h.xyz / h.w) * 0.001;
}

// 地形のピクセルの空気遠近法: シーンカラー（fb）に透過率を掛け、内散乱を足す。
//   inscatterPerLux [sr⁻¹]、transmittance [無次元]
void ApplyAerialPerspective(uint2 pixel, float3 inscatterPerLux, float3 transmittance)
{
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    if ((gFlags & SKY_FLAG_WRITE_AP_TRANSMITTANCE) != 0) {
        output[pixel] = float4(transmittance, 1.0);  // 検証用（V3h）: 内散乱と混ぜずに透過率だけを比べる
        return;
    }
    const float3 geometryFb = output[pixel].rgb;
    // [POINT] 内散乱は単位照度あたりなので、E_sun × 太陽の色を掛けて輝度にし、露出して fb にする。
    //         地形の色はすでに露出済みの fb なので、透過率をそのまま掛ける。
    const float3 fb = geometryFb * transmittance + inscatterPerLux * gSunIlluminanceLux * gExposureScale;
    output[pixel] = float4(min(fb, kMaxSceneColorFb), 1.0);
}

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
