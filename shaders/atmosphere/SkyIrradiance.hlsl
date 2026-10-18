// 空からの拡散照度（地形の環境光）を L2 球面調和で求める。design.md §5.3。
//
// カメラ位置で見た空の輝度（Sky-View LUT 2 枚）を全球で SH に射影し、余弦の畳み込みで照度にする。
// 出力: 照度の SH 係数 9 個（単位照度あたり。E(n) = Σ c_i·Y_i(n) で、太陽照度に対する比 [無次元]）
//
// [POINT] 輝度を 2 つの項に分け、それぞれに合った方向の格子で求積する:
//           Rayleigh + 多重散乱  … 地平線の帯を除けば滑らか → 方位 × 仰角の格子（128×64、仰角は地平線に集める）
//           Mie の単散乱 p_M(θ)·B(ω) … p_M のピーク（g = 0.99 で幅 0.88°）が正距円筒の間隔（~2.8°）より細い
//             → 太陽を極にした極座標の格子（Mie の項だけ別に）で、θ = π·u² と変数変換して太陽の近くを細かく刻む（128×16）。
//               B（位相関数なしの Mie、Sky-View Mie LUT）と Y_lm は滑らかなので方位方向は粗くてよい。
//         p_M を l ≤ 2 の Legendre 展開に置き換える方法は、B が方向で大きく変わる（地平線で明るい）ため
//         壁面の照度で最大 20% ずれた（design.md §14 Phase 4）。
#include "atmosphere/Phase.hlsli"
#include "atmosphere/SkyPassConstants.hlsli"
#include "common/SphericalHarmonics.hlsli"

static const uint kThreads = 256;
static const uint kGridWidth = 128;  // Rayleigh + 多重散乱: 方位方向
static const uint kGridHeight = 64;  // Rayleigh + 多重散乱: 仰角方向
static const uint kMieThetaCount = 128;  // Mie: 太陽からの角度方向（θ = π·u²）
static const uint kMiePhiCount = 16;     // Mie: 太陽の周りの方位方向

groupshared float3 gsSum[kShCount][kThreads];

void Accumulate(inout float3 sum[kShCount], float3 dir, float3 radianceTimesSolidAngle)
{
    float b[9];
    ShBasis(dir, b);
    [unroll] for (uint k = 0; k < kShCount; ++k) {
        sum[k] += radianceTimesSolidAngle * b[k];
    }
}

[numthreads(kThreads, 1, 1)]
void CSMain(uint3 thread : SV_GroupThreadID)
{
    float3 sum[kShCount];
    [unroll] for (uint k = 0; k < kShCount; ++k) {
        sum[k] = 0.0;
    }
    float3 rayleighAndMs;
    float3 mieNoPhase;
    // Rayleigh + 多重散乱: 方位は等間隔、仰角は地平線（俯角 acos(R/r)）を中心に二次で刻む:
    //   仰角 e = e_h ± (π/2 ∓ e_h)·s²（s ∈ (−1, 1) の等間隔の中点）、dω = cos(e)·|de/ds|·ds·dφ
    // [POINT] 空は地平線のすぐ上が最も明るく、その下（地面の手前の大気）で急に暗くなる。仰角を等間隔に
    //         64 分割（~2.8°）すると、この帯を取りこぼして壁面の照度が最大 7% ずれる。Sky-View LUT の
    //         緯度の割り付けと同じ考え方で、地平線の近くにサンプルを集める（64 行で 0.6% 以内）。
    const float horizonElevation = -acos(saturate(gAtmosphere.bottomRadiusKm / length(gCameraPosKm)));
    for (uint i = thread.x; i < kGridWidth * kGridHeight; i += kThreads) {
        const float s = -1.0 + 2.0 * (float(i / kGridWidth) + 0.5) / kGridHeight;
        const float range = s > 0.0 ? 0.5 * kPi - horizonElevation : 0.5 * kPi + horizonElevation;
        const float elevation = horizonElevation + sign(s) * range * s * s;
        const float azimuth = 2.0 * kPi * (float(i % kGridWidth) + 0.5) / kGridWidth;
        const float3 dir = float3(cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth));
        const float solidAngle = cos(elevation) * (range * 2.0 * abs(s) * (2.0 / kGridHeight)) * (2.0 * kPi / kGridWidth);
        SampleSkyViewLuts(gCameraPosKm, dir, rayleighAndMs, mieNoPhase);
        Accumulate(sum, dir, rayleighAndMs * solidAngle);
    }
    // Mie の単散乱: 太陽方向 s を極にした極座標。θ = π·u²（u は等間隔の中点）、dθ = 2π·u·du、
    // dω = sinθ·dθ·dφ。g = 0.99 でも p_M の全立体角の積分が 1 に 1e-4 で一致する刻み。
    float3 tangent;
    float3 bitangent;
    OrthonormalBasis(gSunDirection, tangent, bitangent);
    for (uint j = thread.x; j < kMieThetaCount * kMiePhiCount; j += kThreads) {
        const float u = (float(j % kMieThetaCount) + 0.5) / kMieThetaCount;
        const float phi = 2.0 * kPi * (float(j / kMieThetaCount) + 0.5) / kMiePhiCount;
        const float theta = kPi * u * u;
        const float solidAngle = sin(theta) * (2.0 * kPi * u / kMieThetaCount) * (2.0 * kPi / kMiePhiCount);
        const float3 dir = cos(theta) * gSunDirection + sin(theta) * (cos(phi) * tangent + sin(phi) * bitangent);
        SampleSkyViewLuts(gCameraPosKm, dir, rayleighAndMs, mieNoPhase);
        Accumulate(sum, dir, mieNoPhase * (MiePhase(gAtmosphere, cos(theta)) * solidAngle));
    }
    [unroll] for (uint k3 = 0; k3 < kShCount; ++k3) {
        gsSum[k3][thread.x] = sum[k3];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kThreads / 2; stride > 0; stride /= 2) {
        if (thread.x < stride) {
            [unroll] for (uint k4 = 0; k4 < kShCount; ++k4) {
                gsSum[k4][thread.x] += gsSum[k4][thread.x + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread.x < kShCount) {
        RWStructuredBuffer<float4> output = ResourceDescriptorHeap[gSkyIrradianceUav];
        // 輝度の SH 係数 × Â_l = 照度の SH 係数
        output[thread.x] = float4(gsSum[thread.x][0] * kShCosineLobe[thread.x], 0.0);
    }
}
