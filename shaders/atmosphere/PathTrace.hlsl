// 大気のパストレーサー（多重散乱の真値。V4 の比較対象）。
//
// 方式（Hillaire 2020 のリファレンスと同様）:
//   - 自由行程: デルタトラッキング（媒質が不均一なので、最大消散係数 σ̄ で仮の衝突を起こし、
//     確率 σt(x)/σ̄ で本当の衝突とする）
//   - 太陽への次イベント推定（NEE）: 透過率は比率トラッキング（Π(1 − σt/σ̄)）
//   - 散乱方向: Rayleigh（一様サンプリング）と Mie（Henyey-Greenstein で重点サンプリングし
//     Cornette-Shanks との比で補正）の混合を 1 サンプル MIS で選ぶ
//   - 吸収（Mie の一部・オゾン）はスループットに σs/σt を掛けて扱う
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

// n を z 軸とする正規直交基底（Duff et al. 2017）
void OrthonormalBasis(float3 n, out float3 b1, out float3 b2)
{
    const float s = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    b2 = float3(b, s + n.y * n.y * a, -n.y);
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

// チャンネル c の最大消散係数 σ̄ [km⁻¹]: 各成分の密度は 1 以下なので、係数の和が上限になる
float MajorantPerKm(uint c)
{
    return gAtmosphere.rayleighScatteringPerKm[c] + gAtmosphere.mieExtinctionPerKm[c] +
           gAtmosphere.ozoneAbsorptionPerKm[c];
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
        const float3 x = p + t * gSunDirection;
        const MediumSample m = SampleMedium(gAtmosphere, length(x) - gAtmosphere.bottomRadiusKm);
        transmittance *= 1.0 - m.extinction[c] / majorant;
    }
    return transmittance;
}

// チャンネル c の 1 本の経路。単位照度あたりの輝度 [sr⁻¹] の推定値を返す。
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
                                scatteringM * CornetteShanksPhase(g, cosSun)) / scattering;
        luminance += throughput * phaseSun * SunTransmittanceRatioTracking(p, c, majorant, rng);

        // 次の方向: 混合分布 pdf = w_R/(4π) + w_M·p_HG、重み = 位相関数 / pdf（どちらの比も有界）
        const float weightR = scatteringR / scattering;
        const float3 newDir = NextFloat(rng) < weightR ? SampleUniformSphere(rng) : SampleHenyeyGreenstein(rng, d, g);
        const float cosNew = dot(newDir, d);
        const float pdf = weightR * kIsotropicPhase + (1.0 - weightR) * HenyeyGreensteinPhase(g, cosNew);
        const float phase = (scatteringR * RayleighPhase(cosNew) + scatteringM * CornetteShanksPhase(g, cosNew)) /
                            scattering;
        throughput *= phase / pdf;
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
            sum[c] += TraceChannel(c, gCameraPosKm, viewDir, rng);
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
