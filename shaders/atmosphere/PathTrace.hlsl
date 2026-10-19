// 大気のパストレーサー（多重散乱の真値。V4 の比較対象）。
//
// 方式（Hillaire 2020 のリファレンスと同様）:
//   - 自由行程: デルタトラッキング（媒質が不均一なので、最大消散係数 σ̄ で仮の衝突を起こし、
//     確率 σt(x)/σ̄ で本当の衝突とする）
//   - 太陽への次イベント推定（NEE）: 透過率は比率トラッキング（Π(1 − σt/σ̄)）
//   - 散乱方向: Rayleigh（一様サンプリング）と Mie（Henyey-Greenstein で重点サンプリングし
//     Mie の位相関数（Cornette-Shanks、またはエアロゾルのプリセットの表。design.md §5.9）との比で補正）の混合を
//     1 サンプル MIS で選ぶ。表のときの g は平均の余弦
//   - 吸収（Mie の一部・オゾン）はスループットに σs/σt を掛けて扱う
//   - 表の媒質（design.md §5.8）: Mie を最大 2 成分・鉛直分布と位相関数を表で与える（V4b 用。TraceChannelTable）
// [POINT] RGB の各チャンネルを独立したスカラーのランダムウォークとして追跡する
//         （係数がチャンネルごとに違うので、1 本の経路を共有すると重み付けが複雑になる。3 倍の計算で単純さを取る）。
// 太陽は点光源（Sky-View LUT・レイマーチと同じ条件）。視線が直接太陽を見る寄与（太陽ディスク）は含めない。
#include "atmosphere/SkyPassConstants.hlsli"
#include "common/Noise.hlsli"

struct Rng
{
    uint state;
};

float NextFloat(inout Rng rng)
{
    rng.state = rng.state * 747796405u + 2891336453u;
    uint word = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8) * (1.0 / 16777216.0);
}

// 指数分布の自由行程 [km]（1 − u ∈ (0, 1] なので log は有限）
float SampleFreeFlight(inout Rng rng, float majorantPerKm)
{
    return -log(1.0 - NextFloat(rng)) / majorantPerKm;
}

float3 SampleUniformSphere(inout Rng rng)
{
    const float z = 1.0 - 2.0 * NextFloat(rng);
    const float phi = 2.0 * kPi * NextFloat(rng);
    const float s = sqrt(saturate(1.0 - z * z));
    return float3(s * cos(phi), s * sin(phi), z);
}

float HenyeyGreensteinPhase(float g, float cosTheta)
{
    const float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * kPi * denom * sqrt(denom));
}

// 進行方向 d のまわりに HG 分布で散乱方向を選ぶ（cosθ は逆関数法）
float3 SampleHenyeyGreenstein(inout Rng rng, float3 d, float g)
{
    const float u = NextFloat(rng);
    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u;
    } else {
        const float sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
        cosTheta = (1.0 + g * g - sq * sq) / (2.0 * g);
    }
    const float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    const float phi = 2.0 * kPi * NextFloat(rng);
    float3 b1;
    float3 b2;
    OrthonormalBasis(d, b1, b2);
    return normalize(sinTheta * cos(phi) * b1 + sinTheta * sin(phi) * b2 + cosTheta * d);
}

float3 SampleCosineHemisphere(inout Rng rng, float3 n)
{
    const float r = sqrt(NextFloat(rng));
    const float phi = 2.0 * kPi * NextFloat(rng);
    float3 b1;
    float3 b2;
    OrthonormalBasis(n, b1, b2);
    return normalize(r * cos(phi) * b1 + r * sin(phi) * b2 + sqrt(saturate(1.0 - r * r)) * n);
}

// ---------------------------------------------------------------------------
// 表の媒質（design.md §5.8、V4b）。reference/aerosol_experiment.py が書き出す StructuredBuffer<float>。
//   [0] Mie の成分の数（最大 2）、[1..15] 予備
//   成分 k のブロック（kMediumHeaderFloats + k·kMediumComponentFloats から）:
//     [0..2] σs RGB [km⁻¹]、[3] 鉛直分布の最大値、[4..6] σa RGB [km⁻¹]、[7] 予備、
//     [8 ..] 鉛直分布 256 点（高度 h_i = 100·(i/255)² km）、
//     位相関数 3 × 2049 点（θ_j = π·(j/2048)²）[sr⁻¹]、累積分布 3 × 2049 点（θ ≤ θ_j の確率）
//   Rayleigh とオゾンは解析式の媒質（gAtmosphere）と同じ。
// ---------------------------------------------------------------------------
static const uint kMediumHeaderFloats = 16;
static const uint kMediumProfileCount = 256;
static const uint kMediumPhaseCount = 2049;
static const uint kMediumMaxComponents = 2;
static const uint kMediumComponentFloats = 8 + kMediumProfileCount + 6 * kMediumPhaseCount;

bool UseTableMedium()
{
    return gPathTraceMediumSrv != kInvalidDescriptorIndex;
}

uint MediumBase(uint k)
{
    return kMediumHeaderFloats + k * kMediumComponentFloats;
}

uint MediumComponentCount(StructuredBuffer<float> m)
{
    return min((uint)m[0], kMediumMaxComponents);
}

// 成分 k の鉛直分布（表を v = √(h/100) で線形補間）
float MediumProfile(StructuredBuffer<float> m, uint k, float altitudeKm)
{
    const float f = sqrt(saturate(altitudeKm / 100.0)) * float(kMediumProfileCount - 1);
    const uint i = min((uint)f, kMediumProfileCount - 2);
    const uint b = MediumBase(k) + 8;
    return lerp(m[b + i], m[b + i + 1], f - float(i));
}

// 位相関数の表の区間 j の端の x = 1 − cosθ（θ = π·(j/2048)²）。1 − cosθ = 2·sin²(θ/2) で、θ が小さくても精度を保つ。
float MediumBinX(uint j)
{
    const float u = float(j) / float(kMediumPhaseCount - 1);
    const float s = sin(0.5 * kPi * u * u);
    return 2.0 * s * s;
}

// 散乱角の余弦から、表の位置 f = √(θ/π)·2048（整数部が区間）
float MediumPhaseCoord(float cosTheta)
{
    return sqrt(acos(clamp(cosTheta, -1.0, 1.0)) / kPi) * float(kMediumPhaseCount - 1);
}

float MediumPhase(StructuredBuffer<float> m, uint k, uint c, float f)
{
    const uint i = min((uint)f, kMediumPhaseCount - 2);
    const uint b = MediumBase(k) + 8 + kMediumProfileCount + c * kMediumPhaseCount;
    return lerp(m[b + i], m[b + i + 1], f - float(i));
}

// 区間の中で cosθ を一様に選ぶサンプリングの pdf [sr⁻¹]: 区間の確率 / (2π·Δx)
float MediumPhasePdf(StructuredBuffer<float> m, uint k, uint c, float f)
{
    const uint i = min((uint)f, kMediumPhaseCount - 2);
    const uint b = MediumBase(k) + 8 + kMediumProfileCount + 3 * kMediumPhaseCount + c * kMediumPhaseCount;
    const float probability = m[b + i + 1] - m[b + i];
    return probability / (2.0 * kPi * max(MediumBinX(i + 1) - MediumBinX(i), 1e-30));
}

// 成分 k の位相関数の表から散乱方向を選ぶ: 累積分布で区間を選び（二分探索）、区間の中で cosθ を一様に選ぶ
float3 SampleMediumPhase(StructuredBuffer<float> m, uint k, uint c, float3 d, inout Rng rng)
{
    const uint b = MediumBase(k) + 8 + kMediumProfileCount + 3 * kMediumPhaseCount + c * kMediumPhaseCount;
    const float xi = NextFloat(rng);
    uint lo = 0;
    uint hi = kMediumPhaseCount - 1;
    while (hi - lo > 1) {
        const uint mid = (lo + hi) / 2;
        if (m[b + mid] <= xi) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const float x = lerp(MediumBinX(lo), MediumBinX(lo + 1), NextFloat(rng));
    const float cosTheta = 1.0 - x;
    const float sinTheta = sqrt(saturate(x * (2.0 - x)));
    const float phi = 2.0 * kPi * NextFloat(rng);
    float3 b1;
    float3 b2;
    OrthonormalBasis(d, b1, b2);
    return normalize(sinTheta * cos(phi) * b1 + sinTheta * sin(phi) * b2 + cosTheta * d);
}

struct TableMediumSample
{
    float scatteringR;                          // σs_R [km⁻¹]（チャンネル c）
    float scatteringM[kMediumMaxComponents];    // σs_Mk [km⁻¹]
    float extinction;                           // σt [km⁻¹]
};

TableMediumSample SampleTableMedium(StructuredBuffer<float> m, float altitudeKm, uint c)
{
    altitudeKm = max(altitudeKm, 0.0);
    TableMediumSample s;
    s.scatteringR = gAtmosphere.rayleighScatteringPerKm[c] * exp(-altitudeKm / gAtmosphere.rayleighScaleHeightKm);
    const float ozone = max(0.0, 1.0 - abs(altitudeKm - gAtmosphere.ozoneCenterKm) / gAtmosphere.ozoneHalfWidthKm);
    s.extinction = s.scatteringR + gAtmosphere.ozoneAbsorptionPerKm[c] * ozone;
    const uint count = MediumComponentCount(m);
    [unroll] for (uint k = 0; k < kMediumMaxComponents; ++k) {
        s.scatteringM[k] = 0.0;
        if (k < count) {
            const float rho = MediumProfile(m, k, altitudeKm);
            const uint b = MediumBase(k);
            s.scatteringM[k] = rho * m[b + c];
            s.extinction += rho * (m[b + c] + m[b + 4 + c]);
        }
    }
    return s;
}

// チャンネル c の最大消散係数 σ̄ [km⁻¹]: 各成分の密度の最大値 × 係数の和が上限になる
// （解析式の媒質は Rayleigh・オゾンが 1 以下、Mie の成分 k が 1 + a_k 以下（地表、自由大気の背景の分。design.md §5.9）。
//  表の媒質は分布の最大値を表に持つ）
float MajorantPerKm(uint c)
{
    if (UseTableMedium()) {
        StructuredBuffer<float> m = ResourceDescriptorHeap[gPathTraceMediumSrv];
        float majorant = gAtmosphere.rayleighScatteringPerKm[c] + gAtmosphere.ozoneAbsorptionPerKm[c];
        const uint count = MediumComponentCount(m);
        for (uint k = 0; k < count; ++k) {
            const uint b = MediumBase(k);
            majorant += m[b + 3] * (m[b + c] + m[b + 4 + c]);
        }
        return majorant;
    }
    // [POINT] 背景も成分 1 もないとき（Phase 4.8 までの媒質）は、前と同じ式（パストレースの回帰を画素で比べるため）
    float majorant = gAtmosphere.rayleighScatteringPerKm[c] + gAtmosphere.mieExtinctionPerKm[c] +
                     gAtmosphere.ozoneAbsorptionPerKm[c];
#if MIE_GENERAL
    majorant += gAtmosphere.mieFreeTropDensity * gAtmosphere.mieExtinctionPerKm[c];
    if (gAtmosphere.mieComponentCount > 1) {
        majorant += (1.0 + gAtmosphere.mie1FreeTropDensity) * gAtmosphere.mie1ExtinctionPerKm[c];
    }
#endif
    return majorant;
}

// 点 x のチャンネル c の消散係数（解析式 / 表の媒質）
float ExtinctionAt(float3 x, uint c)
{
    const float altitudeKm = length(x) - gAtmosphere.bottomRadiusKm;
    if (UseTableMedium()) {
        StructuredBuffer<float> m = ResourceDescriptorHeap[gPathTraceMediumSrv];
        return SampleTableMedium(m, altitudeKm, c).extinction;
    }
    return SampleMedium(gAtmosphere, altitudeKm).extinction[c];
}

// p から太陽方向の透過率（比率トラッキング）。惑星の影なら 0。
float SunTransmittanceRatioTracking(float3 p, uint c, float majorant, inout Rng rng)
{
    const float r = length(p);
    if (RayIntersectsGround(gAtmosphere, r, dot(p, gSunDirection) / r)) {
        return 0.0;
    }
    float t0;
    float t1;
    if (!RaySphere(p, gSunDirection, gAtmosphere.topRadiusKm, t0, t1)) {
        return 1.0;
    }
    float transmittance = 1.0;
    float t = 0.0;
    for (uint k = 0; k < 4096; ++k) {
        t += SampleFreeFlight(rng, majorant);
        if (t >= t1) {
            break;
        }
        transmittance *= 1.0 - ExtinctionAt(p + t * gSunDirection, c) / majorant;
    }
    return transmittance;
}

// チャンネル c の 1 本の経路。単位照度あたりの輝度 [sr⁻¹] の推定値を返す（Mie が 1 成分の解析式の媒質）。
float TraceChannel(uint c, float3 origin, float3 dir, inout Rng rng)
{
    const float majorant = MajorantPerKm(c);
    const float g = gAtmosphere.miePhaseG;
    float throughput = 1.0;
    float luminance = 0.0;
    float3 p = origin;
    float3 d = dir;
    for (uint depth = 0; depth < gMaxDepth; ++depth) {
        float tTop0;
        float tTop1;
        if (!RaySphere(p, d, gAtmosphere.topRadiusKm, tTop0, tTop1) || tTop1 <= 0.0) {
            break;
        }
        float tMax = tTop1;
        bool hitGround = false;
        float tGround0;
        float tGround1;
        if (RaySphere(p, d, gAtmosphere.bottomRadiusKm, tGround0, tGround1) && tGround0 > 0.0 && tGround0 < tMax) {
            tMax = tGround0;
            hitGround = true;
        }

        // デルタトラッキング
        float t = 0.0;
        bool collided = false;
        MediumSample m = (MediumSample)0;
        for (uint k = 0; k < 65536; ++k) {
            t += SampleFreeFlight(rng, majorant);
            if (t >= tMax) {
                break;
            }
            m = SampleMedium(gAtmosphere, length(p + t * d) - gAtmosphere.bottomRadiusKm);
            if (NextFloat(rng) * majorant < m.extinction[c]) {
                collided = true;
                break;
            }
        }

        if (!collided) {
            if (!hitGround) {
                break;  // 大気の外へ抜けた
            }
            // 地表（ランバート）: 太陽への NEE と、余弦分布での反射
            const float albedo = gAtmosphere.groundAlbedo[c];
            if (albedo <= 0.0) {
                break;
            }
            const float3 n = normalize(p + tMax * d);
            p = n * (gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);
            const float cosSun = dot(n, gSunDirection);
            if (cosSun > 0.0) {
                luminance += throughput * albedo / kPi * cosSun * SunTransmittanceRatioTracking(p, c, majorant, rng);
            }
            throughput *= albedo;
            d = SampleCosineHemisphere(rng, n);
            continue;
        }

        p += t * d;
        // [POINT] 衝突点が float32 の誤差で地表の下に入ると、次の光線が惑星の内部を通ってしまう
        //         （地表との交点が負の t になり「地表に当たらない」と判定される）。地表の 1 m 上に戻す。
        const float minRadius = gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm;
        if (dot(p, p) < minRadius * minRadius) {
            p = normalize(p) * minRadius;
        }
        const float scatteringR = m.rayleighScattering[c];
        const float scatteringM = m.mieScattering[c];
        const float scattering = scatteringR + scatteringM;
        // [POINT] 吸収の扱い: 本当の衝突のうち散乱の割合 σs/σt をスループットに掛ける（吸収で経路を終えない）
        throughput *= scattering / m.extinction[c];

        // 太陽への NEE: 位相関数は Rayleigh と Mie を σs の比で混ぜたもの
        const float cosSun = dot(d, gSunDirection);
        const float phaseSun = (scatteringR * RayleighPhase(cosSun) +
                                scatteringM * MiePhase(gAtmosphere, 0, cosSun)[c]) / scattering;
        luminance += throughput * phaseSun * SunTransmittanceRatioTracking(p, c, majorant, rng);

        // 次の方向: 混合分布 pdf = w_R/(4π) + w_M·p_HG、重み = 位相関数 / pdf（どちらの比も有界）
        const float weightR = scatteringR / scattering;
        const float3 newDir = NextFloat(rng) < weightR ? SampleUniformSphere(rng) : SampleHenyeyGreenstein(rng, d, g);
        const float cosNew = dot(newDir, d);
        const float pdf = weightR * kIsotropicPhase + (1.0 - weightR) * HenyeyGreensteinPhase(g, cosNew);
        const float phase = (scatteringR * RayleighPhase(cosNew) + scatteringM * MiePhase(gAtmosphere, 0, cosNew)[c]) /
                            scattering;
        throughput *= phase / pdf;
        d = newDir;
    }
    return luminance;
}

// Mie が 2 成分の解析式の媒質（design.md §5.9）でのチャンネル c の 1 本の経路。TraceChannel と同じ手順で、散乱・NEE・
// 方向のサンプリングを成分ごとに混ぜたもの（表の媒質の TraceChannelTable と同じ作り）。
// [POINT] 1 成分の TraceChannel は Phase 4.8 までと同じコードのまま残す（式の形を変えるとコンパイラの演算の並びが
//         変わり、パストレースの回帰が画素で一致しなくなった）
float TraceChannelTwoMie(uint c, float3 origin, float3 dir, inout Rng rng)
{
    const float majorant = MajorantPerKm(c);
    const float g0 = gAtmosphere.miePhaseG;
    const float g1 = gAtmosphere.mie1PhaseG;
    float throughput = 1.0;
    float luminance = 0.0;
    float3 p = origin;
    float3 d = dir;
    for (uint depth = 0; depth < gMaxDepth; ++depth) {
        float tTop0;
        float tTop1;
        if (!RaySphere(p, d, gAtmosphere.topRadiusKm, tTop0, tTop1) || tTop1 <= 0.0) {
            break;
        }
        float tMax = tTop1;
        bool hitGround = false;
        float tGround0;
        float tGround1;
        if (RaySphere(p, d, gAtmosphere.bottomRadiusKm, tGround0, tGround1) && tGround0 > 0.0 && tGround0 < tMax) {
            tMax = tGround0;
            hitGround = true;
        }

        // デルタトラッキング
        float t = 0.0;
        bool collided = false;
        MediumSample m = (MediumSample)0;
        for (uint k = 0; k < 65536; ++k) {
            t += SampleFreeFlight(rng, majorant);
            if (t >= tMax) {
                break;
            }
            m = SampleMedium(gAtmosphere, length(p + t * d) - gAtmosphere.bottomRadiusKm);
            if (NextFloat(rng) * majorant < m.extinction[c]) {
                collided = true;
                break;
            }
        }

        if (!collided) {
            if (!hitGround) {
                break;
            }
            const float albedo = gAtmosphere.groundAlbedo[c];
            if (albedo <= 0.0) {
                break;
            }
            const float3 n = normalize(p + tMax * d);
            p = n * (gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);
            const float cosSun = dot(n, gSunDirection);
            if (cosSun > 0.0) {
                luminance += throughput * albedo / kPi * cosSun * SunTransmittanceRatioTracking(p, c, majorant, rng);
            }
            throughput *= albedo;
            d = SampleCosineHemisphere(rng, n);
            continue;
        }

        p += t * d;
        const float minRadius = gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm;
        if (dot(p, p) < minRadius * minRadius) {
            p = normalize(p) * minRadius;
        }
        const float scatteringR = m.rayleighScattering[c];
        const float scatteringM0 = m.mieScattering[c];
        const float scatteringM1 = m.mie1Scattering[c];
        const float scattering = scatteringR + scatteringM0 + scatteringM1;
        throughput *= scattering / m.extinction[c];

        // 太陽への NEE: 位相関数は Rayleigh と Mie の各成分を σs の比で混ぜたもの
        const float cosSun = dot(d, gSunDirection);
        const float phaseSun = (scatteringR * RayleighPhase(cosSun) + scatteringM0 * MiePhase(gAtmosphere, 0, cosSun)[c] +
                                scatteringM1 * MiePhase(gAtmosphere, 1, cosSun)[c]) / scattering;
        luminance += throughput * phaseSun * SunTransmittanceRatioTracking(p, c, majorant, rng);

        // 次の方向: σs の比で成分を選び、その成分の分布（Rayleigh は一様、Mie は g_k の Henyey-Greenstein）で
        // サンプリング。重み = 混ぜた位相関数 / 混ぜた pdf（σs の重みは共通なので和で割らなくてよい）
        const float pick = NextFloat(rng) * scattering;
        float3 newDir;
        if (pick < scatteringR) {
            newDir = SampleUniformSphere(rng);
        } else {
            newDir = SampleHenyeyGreenstein(rng, d, pick - scatteringR < scatteringM0 ? g0 : g1);
        }
        const float cosNew = dot(newDir, d);
        const float pdf = scatteringR * kIsotropicPhase + scatteringM0 * HenyeyGreensteinPhase(g0, cosNew) +
                          scatteringM1 * HenyeyGreensteinPhase(g1, cosNew);
        const float phase = scatteringR * RayleighPhase(cosNew) + scatteringM0 * MiePhase(gAtmosphere, 0, cosNew)[c] +
                            scatteringM1 * MiePhase(gAtmosphere, 1, cosNew)[c];
        throughput *= phase / max(pdf, 1e-30);
        d = newDir;
    }
    return luminance;
}

// 表の媒質（§5.8）でのチャンネル c の 1 本の経路。TraceChannel と同じ手順で、Mie を最大 2 成分・位相関数を表にしたもの。
float TraceChannelTable(uint c, float3 origin, float3 dir, inout Rng rng)
{
    StructuredBuffer<float> medium = ResourceDescriptorHeap[gPathTraceMediumSrv];
    const uint count = MediumComponentCount(medium);
    const float majorant = MajorantPerKm(c);
    float throughput = 1.0;
    float luminance = 0.0;
    float3 p = origin;
    float3 d = dir;
    for (uint depth = 0; depth < gMaxDepth; ++depth) {
        float tTop0;
        float tTop1;
        if (!RaySphere(p, d, gAtmosphere.topRadiusKm, tTop0, tTop1) || tTop1 <= 0.0) {
            break;
        }
        float tMax = tTop1;
        bool hitGround = false;
        float tGround0;
        float tGround1;
        if (RaySphere(p, d, gAtmosphere.bottomRadiusKm, tGround0, tGround1) && tGround0 > 0.0 && tGround0 < tMax) {
            tMax = tGround0;
            hitGround = true;
        }

        // デルタトラッキング
        float t = 0.0;
        bool collided = false;
        TableMediumSample m = (TableMediumSample)0;
        for (uint k = 0; k < 65536; ++k) {
            t += SampleFreeFlight(rng, majorant);
            if (t >= tMax) {
                break;
            }
            m = SampleTableMedium(medium, length(p + t * d) - gAtmosphere.bottomRadiusKm, c);
            if (NextFloat(rng) * majorant < m.extinction) {
                collided = true;
                break;
            }
        }

        if (!collided) {
            if (!hitGround) {
                break;
            }
            const float albedo = gAtmosphere.groundAlbedo[c];
            if (albedo <= 0.0) {
                break;
            }
            const float3 n = normalize(p + tMax * d);
            p = n * (gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);
            const float cosSun = dot(n, gSunDirection);
            if (cosSun > 0.0) {
                luminance += throughput * albedo / kPi * cosSun * SunTransmittanceRatioTracking(p, c, majorant, rng);
            }
            throughput *= albedo;
            d = SampleCosineHemisphere(rng, n);
            continue;
        }

        p += t * d;
        const float minRadius = gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm;
        if (dot(p, p) < minRadius * minRadius) {
            p = normalize(p) * minRadius;
        }
        float scattering = m.scatteringR;
        [unroll] for (uint k2 = 0; k2 < kMediumMaxComponents; ++k2) {
            scattering += m.scatteringM[k2];
        }
        throughput *= scattering / m.extinction;

        // 太陽への NEE: 位相関数は Rayleigh と Mie の各成分を σs の比で混ぜたもの
        const float fSun = MediumPhaseCoord(dot(d, gSunDirection));
        float phaseSun = m.scatteringR * RayleighPhase(dot(d, gSunDirection));
        for (uint k3 = 0; k3 < count; ++k3) {
            phaseSun += m.scatteringM[k3] * MediumPhase(medium, k3, c, fSun);
        }
        luminance += throughput * (phaseSun / scattering) * SunTransmittanceRatioTracking(p, c, majorant, rng);

        // 次の方向: σs の比で成分を選び、その成分の分布でサンプリング。重み = 混ぜた位相関数 / 混ぜた pdf
        float pick = NextFloat(rng) * scattering;
        float3 newDir;
        if (pick < m.scatteringR || count == 0) {
            newDir = SampleUniformSphere(rng);
        } else {
            pick -= m.scatteringR;
            const uint k4 = (count > 1 && pick >= m.scatteringM[0]) ? 1u : 0u;
            newDir = SampleMediumPhase(medium, k4, c, d, rng);
        }
        const float cosNew = dot(newDir, d);
        const float fNew = MediumPhaseCoord(cosNew);
        float pdf = m.scatteringR * kIsotropicPhase;
        float phase = m.scatteringR * RayleighPhase(cosNew);
        for (uint k5 = 0; k5 < count; ++k5) {
            pdf += m.scatteringM[k5] * MediumPhasePdf(medium, k5, c, fNew);
            phase += m.scatteringM[k5] * MediumPhase(medium, k5, c, fNew);
        }
        // [POINT] pdf と phase は同じ σs の重みなので、比をとれば σs の和で割る必要はない
        throughput *= phase / max(pdf, 1e-30);
        d = newDir;
    }
    return luminance;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float3 viewDir = OutputViewDirection(id.xy);
    Rng rng;
    rng.state = PcgHash(id.x + PcgHash(id.y + PcgHash(gFrameIndex)));

    float3 sum = 0.0;
    for (uint s = 0; s < gSamplesPerDispatch; ++s) {
        [unroll] for (uint c = 0; c < 3; ++c) {
#if MIE_GENERAL
            // 解析式の媒質が一般形（2 成分・背景）。1 成分でも背景があれば TraceChannelTwoMie（成分 1 の σs は 0）
            sum[c] += UseTableMedium() ? TraceChannelTable(c, gCameraPosKm, viewDir, rng)
                                       : TraceChannelTwoMie(c, gCameraPosKm, viewDir, rng);
#else
            sum[c] += UseTableMedium() ? TraceChannelTable(c, gCameraPosKm, viewDir, rng)
                                       : TraceChannel(c, gCameraPosKm, viewDir, rng);
#endif
        }
    }

    // [POINT] 偶数フレームと奇数フレームを別の累積テクスチャに足す。2 つの平均の差から
    //         モンテカルロのノイズを見積もれる（検証で「真値の誤差」が十分小さいことを確認するため）。
    //         frameIndex 0 と 1 はそれぞれの最初の書き込みなので、前の値を捨てる（累積のリセット）。
    RWTexture2D<float4> accumulation = ResourceDescriptorHeap[(gFrameIndex & 1) ? gAccumulationUavB
                                                                                 : gAccumulationUavA];
    RWTexture2D<float4> other = ResourceDescriptorHeap[(gFrameIndex & 1) ? gAccumulationUavA : gAccumulationUavB];
    float4 acc = gFrameIndex < 2 ? 0.0 : accumulation[id.xy];
    acc += float4(sum, float(gSamplesPerDispatch));
    accumulation[id.xy] = acc;
    const float4 otherAcc = gFrameIndex < 1 ? 0.0 : other[id.xy];
    const float4 total = acc + otherAcc;

    float3 luminance = total.rgb / max(total.a, 1.0);
    if ((gFlags & SKY_FLAG_SUN_DISK) != 0) {
        luminance += SunDiskPerLux(gCameraPosKm, viewDir);
    }
    WriteSkyOutput(id.xy, luminance, viewDir);
}
