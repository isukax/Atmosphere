// Multi-scattering LUT の生成（Hillaire 2020 §5.5、式 (5)〜(10)）。
//
// 各テクセル = 高度 r・太陽天頂角の余弦 μs の点。値は多重散乱の寄与 Ψ_ms [sr⁻¹]（単位照度あたり）。
// 描画時に σs(x)·Ψ_ms を光源項に足すだけで、2 次以降の散乱（無限次まで）を近似できる。
//
// [POINT] 近似の考え方:
//   - 2 次以降の散乱は等方的とみなす（位相関数を 1/(4π) にする）。
//   - 点 x の周りの全方向（64 方向）から、単散乱で x に届く 2 次散乱の輝度 L₂ を集める（地表の反射も含む）。
//   - 同じ方向から、等方な単位入射光が媒質で散乱して x に戻る割合 f_ms を集める。
//   - 近傍も同じ照度を受けていると仮定すると、散乱を n 回繰り返した寄与は L₂·f_ms^(n−1) で、
//     その和は等比級数 L₂·(1 + f_ms + f_ms² + …) = L₂ / (1 − f_ms)。
#include "atmosphere/AtmosphereIntegrate.hlsli"

cbuffer MultiScatteringLutConstants : register(b0)
{
    AtmosphereParameters gAtmosphere : packoffset(c0);   // c0-c9
    uint   gOutputUav                : packoffset(c10.x);
    uint   gWidth                    : packoffset(c10.y);
    uint   gHeight                   : packoffset(c10.z);
    uint   gStepCount                : packoffset(c10.w);
    uint   gTransmittanceLutSrv      : packoffset(c11.x);
    uint   gSunTransmittanceMode     : packoffset(c11.y);
    uint   gNestedSteps              : packoffset(c11.z);
    uint   gDebugUav                 : packoffset(c11.w);  // 検証用: 各テクセルの (r [km], μs)。不要なら無効値
    float2 gTransmittanceLutSize     : packoffset(c12.x);
    uint   gUniformSteps             : packoffset(c12.z);  // 1: 区間を等間隔に置く（比較用）/ 0: 2 次配置（既定、design.md §12 Q8）
};

static const uint kSqrtDirectionCount = 8;
static const uint kDirectionCount = kSqrtDirectionCount * kSqrtDirectionCount;  // 64 方向

groupshared float3 gsLuminance[kDirectionCount];
groupshared float3 gsMultiScatAs1[kDirectionCount];

// 1 テクセル = 1 スレッドグループ。グループ内の 64 スレッドがそれぞれ 1 方向を積分し、共有メモリで合計する。
[numthreads(1, 1, kDirectionCount)]
void CSMain(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint2 texel = group.xy;
    const uint directionIndex = thread.z;
    const float2 lutSize = float2(gWidth, gHeight);

    float r;
    float muSun;
    MultiScatteringUvToRMuSun(gAtmosphere, (float2(texel) + 0.5) / lutSize, lutSize, r, muSun);
    // 局所座標: 点は +Y 軸上、太陽は XY 平面内
    const float3 origin = float3(0.0, r, 0.0);
    const float3 sunDir = float3(sqrt(saturate(1.0 - muSun * muSun)), muSun, 0.0);

    // [POINT] 球面上の層化一様サンプリング: 方位 θ = 2π·a、極角 φ = acos(1 − 2b)（cos φ が一様 → 立体角が一様）
    const float a = (float(directionIndex % kSqrtDirectionCount) + 0.5) / float(kSqrtDirectionCount);
    const float b = (float(directionIndex / kSqrtDirectionCount) + 0.5) / float(kSqrtDirectionCount);
    const float theta = 2.0 * kPi * a;
    const float phi = acos(1.0 - 2.0 * b);
    const float3 dir = float3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));

    AtmosphereLuts luts;
    luts.transmittanceSrv = gTransmittanceLutSrv;
    luts.transmittanceSize = gTransmittanceLutSize;
    luts.multiScatteringSrv = kInvalidDescriptorIndex;
    luts.multiScatteringSize = 0.0;
    IntegrationSettings settings;
    settings.stepCount = gStepCount;
    settings.sunTransmittanceMode = gSunTransmittanceMode;
    settings.nestedSteps = gNestedSteps;
    settings.multipleScattering = false;
    settings.isotropicPhase = true;
    settings.groundRadiance = true;
    settings.uniformSteps = gUniformSteps != 0;
    settings.separateMiePhase = false;
    settings.maxDistanceKm = kNoDistanceLimitKm;
    DisableOpaqueShadow(settings);
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, luts, settings, origin, dir, sunDir);

    gsLuminance[directionIndex] = result.luminance;
    gsMultiScatAs1[directionIndex] = result.multiScatAs1;
    GroupMemoryBarrierWithGroupSync();

    if (directionIndex != 0) {
        return;
    }
    float3 luminanceSum = 0.0;
    float3 multiScatAs1Sum = 0.0;
    for (uint i = 0; i < kDirectionCount; ++i) {
        luminanceSum += gsLuminance[i];
        multiScatAs1Sum += gsMultiScatAs1[i];
    }
    // [POINT] 全球の積分 ∫ X·p_u dω（p_u = 1/(4π)）をモンテカルロで: Σ X · (4π/N) · (1/(4π)) = 平均。
    //   L₂   [sr⁻¹] : 等方位相で点に向かって散乱される 2 次散乱の輝度
    //   f_ms [無次元]: 単位の等方入射光のうち、散乱されて点に戻る割合（< 1）
    const float3 secondOrder = luminanceSum / float(kDirectionCount);
    const float3 fms = multiScatAs1Sum / float(kDirectionCount);
    const float3 psi = secondOrder / (1.0 - fms);  // 式 (9)(10): Ψ_ms = L₂ · 1/(1 − f_ms)

    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    output[texel] = float4(psi, 1.0);
    if (gDebugUav != kInvalidDescriptorIndex) {
        RWTexture2D<float4> debugOutput = ResourceDescriptorHeap[gDebugUav];
        debugOutput[texel] = float4(r, muSun, 0.0, 0.0);
    }
}
