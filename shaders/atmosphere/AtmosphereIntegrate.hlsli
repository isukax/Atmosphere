// 視線に沿った散乱の積分（Hillaire 2020 §4 式 (1)〜(4)、§5.5 式 (11)）。
// Multi-scattering LUT・Sky-View LUT・リファレンスのレイマーチで共通に使う。
//
// 単位: 照度 1 lx あたりの量で計算する（輝度なら sr⁻¹ = cd/m² per lx）。
//   L_scat = ∫ T_view(t) · σs(t) · [ p(θ) · T_sun(t) + Ψ_ms(t) ] dt  (+ 地表の反射)
#ifndef ATMOSPHERE_INTEGRATE_HLSLI
#define ATMOSPHERE_INTEGRATE_HLSLI

#include "common/Bindless.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"
#include "atmosphere/Phase.hlsli"

#define SUN_TRANSMITTANCE_LUT 0
#define SUN_TRANSMITTANCE_NESTED 1

struct AtmosphereLuts
{
    uint   transmittanceSrv;
    float2 transmittanceSize;
    uint   multiScatteringSrv;
    float2 multiScatteringSize;
};

struct IntegrationSettings
{
    uint stepCount;
    uint sunTransmittanceMode;  // SUN_TRANSMITTANCE_*
    uint nestedSteps;           // SUN_TRANSMITTANCE_NESTED のときの太陽方向の区間数
    bool multipleScattering;    // σs·Ψ_ms（Multi-scattering LUT）を加える
    bool isotropicPhase;        // 位相関数を 1/(4π) にする（Multi-scattering LUT の生成用）
    bool groundRadiance;        // 地表に当たったら地表のランバート反射を加える（Multi-scattering LUT の生成用）
    bool uniformSteps;          // true: 区間を等間隔に置く（Multi-scattering LUT）/ false: 2 次配置（Sky-View LUT・レイマーチ）
    bool separateMiePhase;      // true: Mie の単散乱を luminance に入れず、位相関数なしで mieSingleNoPhase に分ける（Sky-View LUT 用）
};

struct ScatteringResult
{
    float3 luminance;         // 単位照度あたりの輝度 [sr⁻¹]（separateMiePhase なら Mie の単散乱を除く）
    float3 mieSingleNoPhase;  // ∫ T_view·σs_M·T_sun dt [無次元]。× p_M(θ) [sr⁻¹] で Mie の単散乱の輝度（単位照度あたり）
    float3 multiScatAs1;      // f_ms の被積分量: ∫ T_view·σs dt（等方な単位入射光が散乱して戻る割合）
    float3 transmittance;     // カメラ → 積分区間の終端の透過率
};

// (1 − e^(−x)) / x を桁落ちなく計算する（x = σt·dt ≥ 0）。
// [POINT] x が小さいとき 1 − e^(−x) は「1 に近い数どうしの引き算」になり、float32 では相対誤差が
//         ~6×10⁻⁸ / x まで膨らむ（x = 10⁻⁵ で 0.6%）。GPU の exp の近似誤差は向きが揃うので平均で消えず、
//         細かいステップ（リファレンスの 2048 区間）で空の輝度が一律 1% ずれた（Phase 3 の V2 回帰で発見）。
//         x < 10⁻² ではテイラー展開 1 − x/2 + x²/6 − x³/24（打ち切り誤差 x⁴/120 < 10⁻¹⁰）を使う。
float3 OneMinusExpOverX(float3 x)
{
    const float3 series = 1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0)));
    return select(x < 1e-2, series, (1.0 - exp(-x)) / max(x, 1e-30));
}

// 点 p（惑星座標 [km]）から太陽方向の透過率。太陽が地平線の下（惑星の影）なら 0。
float3 SunTransmittance(AtmosphereParameters a, AtmosphereLuts luts, IntegrationSettings s, float3 p, float3 sunDir)
{
    const float r = length(p);
    const float muSun = dot(p, sunDir) / r;
    // [POINT] 惑星の影: 太陽へ向かう光線が地表に当たる点には直達光が届かない。
    //         太陽は点光源として扱う（視半径 0.27° による半影は無視）。
    if (RayIntersectsGround(a, r, muSun)) {
        return 0.0;
    }
    if (s.sunTransmittanceMode == SUN_TRANSMITTANCE_LUT) {
        Texture2D<float4> lut = ResourceDescriptorHeap[luts.transmittanceSrv];
        return SampleTransmittanceLut(a, lut, gLinearClamp, r, muSun, luts.transmittanceSize);
    }
    return ComputeTransmittanceToTop(a, r, muSun, s.nestedSteps);
}

// 点 p での多重散乱の寄与 Ψ_ms [sr⁻¹]（Multi-scattering LUT を高度と太陽天頂角で引く）
float3 SampleMultipleScattering(AtmosphereParameters a, AtmosphereLuts luts, float3 p, float3 sunDir)
{
    const float r = length(p);
    const float2 uv = MultiScatteringRMuSunToUv(a, r, dot(p, sunDir) / r, luts.multiScatteringSize);
    Texture2D<float4> lut = ResourceDescriptorHeap[luts.multiScatteringSrv];
    return lut.SampleLevel(gLinearClamp, uv, 0.0).rgb;
}

ScatteringResult IntegrateScatteredLuminance(AtmosphereParameters a, AtmosphereLuts luts, IntegrationSettings s,
                                             float3 origin, float3 dir, float3 sunDir)
{
    ScatteringResult result;
    result.luminance = 0.0;
    result.mieSingleNoPhase = 0.0;
    result.multiScatAs1 = 0.0;
    result.transmittance = 1.0;

    // 積分区間: 大気上端の内側から、地表に当たればそこまで
    float tTop0;
    float tTop1;
    if (!RaySphere(origin, dir, a.topRadiusKm, tTop0, tTop1) || tTop1 <= 0.0) {
        return result;
    }
    const float tStart = max(tTop0, 0.0);
    float tEnd = tTop1;
    bool hitGround = false;
    float tGround0;
    float tGround1;
    if (RaySphere(origin, dir, a.bottomRadiusKm, tGround0, tGround1) && tGround0 > 0.0 && tGround0 < tEnd) {
        tEnd = tGround0;
        hitGround = true;
    }

    const float cosTheta = dot(dir, sunDir);
    const float phaseRayleigh = s.isotropicPhase ? kIsotropicPhase : RayleighPhase(cosTheta);
    const float phaseMie = s.isotropicPhase ? kIsotropicPhase : CornetteShanksPhase(a.miePhaseG, cosTheta);

    // [POINT] 光線上のサンプル配置（区間端）。LUT のマッピング（テクセル ↔ 方向）とは別の話。
    //   2 次配置 t_k = (k/N)²（Sky-View LUT・レイマーチ）: カメラの近くを細かくする。地表付近の Mie は
    //     スケールハイト 1.2 km で急減するので、等間隔（Sky-View LUT の 30 区間で 1 区間 ~5 km）だと最初の区間の
    //     中点で密度を過小評価し、太陽の周りの光輪が 10〜19% 暗くなった（Phase 3 の V3 で発見）。
    //   等間隔 t_k = k/N（Multi-scattering LUT）: 全高度の点から下向きを含む全方向に積分するので、2 次配置では
    //     かえって誤差が増えた（20 区間で最大 2.7% → 4.9%、design.md §14）。
    // 参照実装（sebh/UnrealEngineSkyAtmosphere）も 2 次配置は Sky-View LUT と画面のレイマーチだけで、そこでは
    // 区間数も光線の長さで変える。本実装は区間数を固定し、区間内のサンプル位置は中点（参照実装は 0.3）。
    const float range = tEnd - tStart;
    const float invSteps = 1.0 / float(s.stepCount);
    float3 throughput = 1.0;
    for (uint i = 0; i < s.stepCount; ++i) {
        const float u0 = float(i) * invSteps;
        const float u1 = float(i + 1) * invSteps;
        const float t0 = tStart + range * (s.uniformSteps ? u0 : u0 * u0);
        const float t1 = tStart + range * (s.uniformSteps ? u1 : u1 * u1);
        const float dt = t1 - t0;
        const float t = 0.5 * (t0 + t1);
        const float3 p = origin + t * dir;
        const MediumSample m = SampleMedium(a, length(p) - a.bottomRadiusKm);
        const float3 scattering = m.rayleighScattering + m.mieScattering;
        const float3 segmentTransmittance = exp(-m.extinction * dt);

        // 区間内の光源項 S [km⁻¹·sr⁻¹]: 単散乱（太陽の直達光）+ 多重散乱（式 (11)）
        const float3 sunTransmittance = SunTransmittance(a, luts, s, p, sunDir);
        const float3 mieNoPhase = m.mieScattering * sunTransmittance;  // [km⁻¹]
        float3 source = m.rayleighScattering * phaseRayleigh * sunTransmittance;
        if (!s.separateMiePhase) {
            source += mieNoPhase * phaseMie;
        }
        if (s.multipleScattering) {
            source += scattering * SampleMultipleScattering(a, luts, p, sunDir);
        }
        // [POINT] 区間内で媒質が一定とみなした解析積分（Hillaire の "energy-conserving" 積分）:
        //           ∫₀^dt S·e^(−σt·x) dx = S·(1 − e^(−σt·dt)) / σt = S·dt·(1 − e^(−x))/x、x = σt·dt
        //         少ないステップ数（Sky-View LUT は 30）でも、区間内の減衰を正しく扱える。
        //         参照実装の (S − S·T)/σt は x が小さいと桁落ちするので、OneMinusExpOverX を使う。
        const float3 segmentWeight = dt * OneMinusExpOverX(m.extinction * dt);
        result.luminance += throughput * source * segmentWeight;
        result.mieSingleNoPhase += throughput * mieNoPhase * segmentWeight;
        result.multiScatAs1 += throughput * scattering * segmentWeight;
        throughput *= segmentTransmittance;
    }

    if (hitGround && s.groundRadiance) {
        // [POINT] 地表のランバート反射: 輝度 = ρ/π × 照度。照度 = E_sun·T_sun·cos(太陽天頂角)（単位照度あたり）。
        const float3 groundPos = origin + tEnd * dir;
        const float3 normal = normalize(groundPos);
        const float3 sunT = SunTransmittance(a, luts, s, groundPos, sunDir);
        result.luminance += throughput * sunT * saturate(dot(normal, sunDir)) * a.groundAlbedo / kPi;
    }
    result.transmittance = throughput;
    return result;
}

#endif // ATMOSPHERE_INTEGRATE_HLSLI
