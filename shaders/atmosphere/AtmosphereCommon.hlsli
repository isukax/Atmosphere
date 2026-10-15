// 大気の媒質モデル・幾何・Transmittance LUT のパラメタ化（Hillaire 2020 / Bruneton 2017）。
//
// 単位（docs/design.md §2）:
//   距離・高度・半径 … km（惑星中心が原点の「惑星座標」）
//   散乱・吸収・消散係数 … km⁻¹
//   透過率 … 無次元
#ifndef ATMOSPHERE_COMMON_HLSLI
#define ATMOSPHERE_COMMON_HLSLI

#include "common/Math.hlsli"

// [POINT] C++ の atmosphere::AtmosphereGpu（src/atmosphere/AtmosphereParams.h）と同じ並び。
//         float3 + float の組で 16 バイトの行をちょうど埋めるので、HLSL と C++ の
//         パッキングが一致する（static_assert で照合している）。
struct AtmosphereParameters
{
    float  bottomRadiusKm;           // 地表の半径（地球 6360）
    float  topRadiusKm;              // 大気上端の半径（地球 6460）
    float  rayleighScaleHeightKm;    // 8
    float  mieScaleHeightKm;         // 1.2
    float3 rayleighScatteringPerKm;  // 海面高度での値
    float  miePhaseG;                // 0.8
    float3 mieScatteringPerKm;       // 海面高度での値
    float  ozoneCenterKm;            // 25
    float3 mieExtinctionPerKm;       // 散乱 + 吸収
    float  ozoneHalfWidthKm;         // 15（テント型分布の半幅）
    float3 ozoneAbsorptionPerKm;     // 最大密度（25 km）での値
    float  sunAngularRadiusRad;      // 0.004675
    float3 groundAlbedo;             // 地表の拡散反射率
    float  pad0;
};

// ---------------------------------------------------------------------------
// 媒質
// ---------------------------------------------------------------------------

struct MediumSample
{
    float3 rayleighScattering;  // σs_R(h) [km⁻¹]
    float3 mieScattering;       // σs_M(h) [km⁻¹]
    float3 extinction;          // σt(h) = σs_R + σt_M + σa_O3 [km⁻¹]
};

// 高度 altitudeKm [km] での媒質の係数。
MediumSample SampleMedium(AtmosphereParameters a, float altitudeKm)
{
    // [POINT] 高度は 0 以上にクランプする。惑星座標の float32 は地表付近（6360 km）で精度が ~0.5 m しかなく、
    //         地表すれすれの点がわずかに負の高度になる。負の高度で密度が 1 を超えると、パストレーサーの
    //         デルタトラッキングの前提（σt ≤ σ̄）が崩れ、惑星内部の点では exp が溢れて NaN になる。
    altitudeKm = max(altitudeKm, 0.0);
    // [POINT] 密度分布: Rayleigh と Mie は指数（スケールハイト H で 1/e）、
    //         オゾンは 25 km を中心とした幅 30 km のテント関数（吸収のみ、散乱しない）。
    const float rayleighDensity = exp(-altitudeKm / a.rayleighScaleHeightKm);
    const float mieDensity = exp(-altitudeKm / a.mieScaleHeightKm);
    const float ozoneDensity = max(0.0, 1.0 - abs(altitudeKm - a.ozoneCenterKm) / a.ozoneHalfWidthKm);

    MediumSample m;
    m.rayleighScattering = a.rayleighScatteringPerKm * rayleighDensity;
    m.mieScattering = a.mieScatteringPerKm * mieDensity;
    m.extinction = m.rayleighScattering + a.mieExtinctionPerKm * mieDensity + a.ozoneAbsorptionPerKm * ozoneDensity;
    return m;
}

// ---------------------------------------------------------------------------
// 幾何
// ---------------------------------------------------------------------------

// 光線 origin + t·dir（|dir| = 1）と、原点中心・半径 radius の球との交点 t0 <= t1。
// 交差しなければ false。
bool RaySphere(float3 origin, float3 dir, float radius, out float t0, out float t1)
{
    const float r = length(origin);
    const float b = dot(origin, dir);
    // [POINT] c = |o|² − R² を (|o|−R)(|o|+R) で計算する。|o| ≈ R ≈ 6360 km のとき
    //         |o|² と R² はどちらも 4×10⁷ 程度で、float の差は桁落ちして地平線付近が壊れる。
    const float c = (r - radius) * (r + radius);
    const float discriminant = b * b - c;
    t0 = 0.0;
    t1 = 0.0;
    if (discriminant < 0.0) {
        return false;
    }
    // [POINT] 2 次方程式の解の安定な形: q = −(b + sign(b)·√D)、解は q と c/q。
    //         −b ± √D をそのまま使うと、片方の解で大きな数どうしの引き算になる。
    const float s = sqrt(discriminant);
    const float q = -(b + (b >= 0.0 ? s : -s));
    const float ta = q;
    const float tb = q != 0.0 ? c / q : -b;
    t0 = min(ta, tb);
    t1 = max(ta, tb);
    return true;
}

// 半径 r [km] の点から天頂角の余弦 mu の方向に、大気上端までの距離 [km]（Bruneton 2017）。
float DistanceToTopAtmosphere(AtmosphereParameters a, float r, float mu)
{
    const float discriminant = r * r * (mu * mu - 1.0) + a.topRadiusKm * a.topRadiusKm;
    return max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
}

// 半径 r の点から mu の方向の光線が地表に当たるか（Bruneton 2017）。
bool RayIntersectsGround(AtmosphereParameters a, float r, float mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + a.bottomRadiusKm * a.bottomRadiusKm >= 0.0;
}

// ---------------------------------------------------------------------------
// 透過率
// ---------------------------------------------------------------------------

// 半径 r、方向 mu から大気上端までの光学的厚さ τ を台形則（stepCount 区間）で数値積分した透過率 exp(−τ)。
// Transmittance LUT の生成と、LUT を使わないリファレンス（太陽方向の入れ子レイマーチ）で使う。
float3 ComputeTransmittanceToTop(AtmosphereParameters a, float r, float mu, uint stepCount)
{
    const float distanceKm = DistanceToTopAtmosphere(a, r, mu);
    const float dt = distanceKm / float(stepCount);
    float3 opticalDepth = 0.0;
    for (uint i = 0; i <= stepCount; ++i) {
        const float t = float(i) * dt;
        // 光線上の点の半径（余弦定理）。sqrt の中は常に正。
        const float ri = sqrt(t * t + 2.0 * r * mu * t + r * r);
        const float weight = (i == 0 || i == stepCount) ? 0.5 : 1.0;  // 台形則
        opticalDepth += SampleMedium(a, ri - a.bottomRadiusKm).extinction * (weight * dt);
    }
    return exp(-opticalDepth);
}

// ---------------------------------------------------------------------------
// Transmittance LUT のパラメタ化（Bruneton 2017 の §"Precomputation"）
// ---------------------------------------------------------------------------

// [POINT] サブテクセル補正: [0,1] のパラメータを「端のテクセル中心」に合わせる。
//         u = 0 / 1 がテクスチャの端（テクセルの外縁）ではなく端のテクセル中心に来るので、
//         双線形補間で端の値がそのまま得られる（Hillaire の参照実装と同じ）。
float FromUnitToSubUv(float u, float resolution)
{
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

float FromSubUvToUnit(float u, float resolution)
{
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

// (r, mu) → LUT の UV。
// [POINT] u は「大気上端までの距離」を、その (r) での最小値（真上）〜最大値（地平線）で
//         正規化したもの。地平線付近の急変する領域にテクセルが集中する。
//         v は地表の接線距離 ρ = √(r² − R_bottom²) を最大値 H で正規化したもの。
float2 TransmittanceRMuToUv(AtmosphereParameters a, float r, float mu, float2 lutSize)
{
    const float H = sqrt(a.topRadiusKm * a.topRadiusKm - a.bottomRadiusKm * a.bottomRadiusKm);
    const float rho = sqrt(max(r * r - a.bottomRadiusKm * a.bottomRadiusKm, 0.0));
    const float d = DistanceToTopAtmosphere(a, r, mu);
    const float dMin = a.topRadiusKm - r;
    const float dMax = rho + H;
    const float xMu = (d - dMin) / (dMax - dMin);
    const float xR = rho / H;
    return float2(FromUnitToSubUv(xMu, lutSize.x), FromUnitToSubUv(xR, lutSize.y));
}

// LUT の UV → (r, mu)。LUT 生成時に各テクセルが表す光線を求める。
void TransmittanceUvToRMu(AtmosphereParameters a, float2 uv, float2 lutSize, out float r, out float mu)
{
    const float xMu = FromSubUvToUnit(uv.x, lutSize.x);
    const float xR = FromSubUvToUnit(uv.y, lutSize.y);
    const float H = sqrt(a.topRadiusKm * a.topRadiusKm - a.bottomRadiusKm * a.bottomRadiusKm);
    const float rho = H * xR;
    r = sqrt(rho * rho + a.bottomRadiusKm * a.bottomRadiusKm);
    const float dMin = a.topRadiusKm - r;
    const float dMax = rho + H;
    const float d = dMin + xMu * (dMax - dMin);
    mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

// LUT から透過率（半径 r の点から mu の方向に大気上端まで）を引く。
// 地表に当たる方向（太陽が地平線の下）では LUT の定義域外なので、呼び出し側で 0 にすること。
float3 SampleTransmittanceLut(AtmosphereParameters a, Texture2D<float4> lut, SamplerState s, float r, float mu,
                              float2 lutSize)
{
    return lut.SampleLevel(s, TransmittanceRMuToUv(a, r, mu, lutSize), 0.0).rgb;
}

// ---------------------------------------------------------------------------
// Multi-scattering LUT のパラメタ化（Hillaire 2020 §5.5）
//   u = cos(太陽天頂角) を [-1, 1] → [0, 1]、v = 高度を [地表, 大気上端] → [0, 1]
// ---------------------------------------------------------------------------

// [POINT] 地表・大気上端のちょうど上の点は光線の判定が不安定になるので、内側へ少しずらす。
static const float kSurfaceOffsetKm = 0.001;

float2 MultiScatteringRMuSunToUv(AtmosphereParameters a, float r, float muSun, float2 lutSize)
{
    const float2 unit = float2(muSun * 0.5 + 0.5,
                               saturate((r - a.bottomRadiusKm) / (a.topRadiusKm - a.bottomRadiusKm)));
    return float2(FromUnitToSubUv(unit.x, lutSize.x), FromUnitToSubUv(unit.y, lutSize.y));
}

void MultiScatteringUvToRMuSun(AtmosphereParameters a, float2 uv, float2 lutSize, out float r, out float muSun)
{
    const float2 unit = float2(FromSubUvToUnit(uv.x, lutSize.x), FromSubUvToUnit(uv.y, lutSize.y));
    muSun = clamp(unit.x * 2.0 - 1.0, -1.0, 1.0);
    r = clamp(a.bottomRadiusKm + saturate(unit.y) * (a.topRadiusKm - a.bottomRadiusKm),
              a.bottomRadiusKm + kSurfaceOffsetKm, a.topRadiusKm - kSurfaceOffsetKm);
}

// ---------------------------------------------------------------------------
// Sky-View LUT のパラメタ化（Hillaire 2020 §5.3、参照実装の UvToSkyViewLutParams）
//   u = 視線と太陽の方位差の余弦（太陽側 = 0、反対側 = 1）を平方根で非線形化
//   v = 視線の天頂角。v = 0.5 がちょうど地平線（高度に応じた見かけの地平線）で、
//       上半分が空、下半分が地面方向。どちらも地平線側に 2 次でテクセルを集める。
// ---------------------------------------------------------------------------

// [POINT] 見かけの地平線の天頂角は高度で変わる: cos β = √(r² − R²) / r（β は「真下」から地平線までの角度）。
//         地平線を常に v = 0.5 に置くことで、カメラの高度が変わっても地平線付近の解像度が保たれる。
void SkyViewUvToViewParams(AtmosphereParameters a, float2 uv, float2 lutSize, float r,
                           out float viewZenithCos, out float lightViewCos)
{
    uv = float2(FromSubUvToUnit(uv.x, lutSize.x), FromSubUvToUnit(uv.y, lutSize.y));
    const float horizonDistance = sqrt(max(r * r - a.bottomRadiusKm * a.bottomRadiusKm, 0.0));
    const float beta = acos(horizonDistance / r);
    const float zenithHorizonAngle = kPi - beta;
    float viewZenithAngle;
    if (uv.y < 0.5) {
        float coord = 1.0 - 2.0 * uv.y;
        coord = 1.0 - coord * coord;
        viewZenithAngle = zenithHorizonAngle * coord;
    } else {
        float coord = uv.y * 2.0 - 1.0;
        viewZenithAngle = zenithHorizonAngle + beta * coord * coord;
    }
    viewZenithCos = cos(viewZenithAngle);
    lightViewCos = -(uv.x * uv.x * 2.0 - 1.0);
}

float2 SkyViewViewParamsToUv(AtmosphereParameters a, bool intersectGround, float viewZenithCos, float lightViewCos,
                             float r, float2 lutSize)
{
    const float horizonDistance = sqrt(max(r * r - a.bottomRadiusKm * a.bottomRadiusKm, 0.0));
    const float beta = acos(horizonDistance / r);
    const float zenithHorizonAngle = kPi - beta;
    const float viewZenithAngle = acos(clamp(viewZenithCos, -1.0, 1.0));
    float2 uv;
    if (!intersectGround) {
        const float coord = 1.0 - sqrt(saturate(1.0 - viewZenithAngle / zenithHorizonAngle));
        uv.y = coord * 0.5;
    } else {
        const float coord = sqrt(saturate((viewZenithAngle - zenithHorizonAngle) / beta));
        uv.y = coord * 0.5 + 0.5;
    }
    uv.x = sqrt(saturate(-lightViewCos * 0.5 + 0.5));
    return float2(FromUnitToSubUv(uv.x, lutSize.x), FromUnitToSubUv(uv.y, lutSize.y));
}

// ---------------------------------------------------------------------------
// 検証用の方向グリッド（正距円筒）
// ---------------------------------------------------------------------------

// 方位 φ = 2π·(x+0.5)/W、仰角 e = π/2 − π·(y+0.5)/H。+Y が天頂。
float3 EquirectDirection(uint2 pixel, uint2 size)
{
    const float phi = 2.0 * kPi * (float(pixel.x) + 0.5) / float(size.x);
    const float elevation = 0.5 * kPi - kPi * (float(pixel.y) + 0.5) / float(size.y);
    return float3(cos(elevation) * cos(phi), sin(elevation), cos(elevation) * sin(phi));
}

#endif // ATMOSPHERE_COMMON_HLSLI
