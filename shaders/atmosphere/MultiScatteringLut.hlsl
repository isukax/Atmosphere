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
//
// MS_SH = 1 の版（design.md §5.11）は、等方の Ψ_ms の代わりに、入ってくる散乱光の場の球面調和の係数（l ≤ 2、6 個）を
// 書く。閉じ方の f_ms を伝達行列 F に広げ、地表との往復も含める。下の #else 以降。F の表を作る CSTransfer（行ごと）を
// 先に、係数を作る CSMain を後にディスパッチする。
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
    // 以下は SH の版（MS_SH）だけが使う（design.md §5.11）
    uint   gRowOffset                : packoffset(c12.w);  // このディスパッチの最初の行（高度）。検証用の重い設定は行ごとに分ける
    float4 gPhaseMoments[6]          : packoffset(c13);    // Mie の成分 0 の χ_0〜χ_2、成分 1 の χ_0〜χ_2（RGB）
    uint   gTransferUav              : packoffset(c19.x);  // F の表（行 = 高度、17 個）。CSTransfer が書く
    uint   gTransferSrv              : packoffset(c19.y);  // 同じ表を CSMain が読む
};

#if !MS_SH

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

#else // MS_SH

// [POINT] SH の版の考え方（design.md §5.11）:
//   - 各テクセル（高度 r・μs）の点 x に入ってくる散乱光の場 L(ω)（散乱 1 回以上）を、局所の枠（z = 天頂、x = 太陽の
//     水平の向き）の SH の係数 c_b で持つ。
//   - 1 次の場 c₁: 太陽光が光線上で 1 回散乱して x に届く輝度（実際の位相関数）+ 地表での直達光の反射を射影する。
//     前方散乱のピークを拾うため、方向は太陽中心の格子（θ = π·u²、32 環 × 方位 32 = 1,024 方向。立体角の重み）。
//     鏡映対称なので半分（512 方向）だけを積む。
//   - 閉じ方: 近傍の場も x と同じ係数（局所性の仮定。Hillaire 2020 の f_ms と同じ）とすると、もう 1 回散乱した場は
//     F·c。F_ab = ∫ Y_a·Y_b·Ã_{l(b)} dω（Ã_l = Σ_k χ_{k,l}·∫T·σs,k dt）に、地面に当たる方向の (ρ/π)·T_g·e_b を足す
//     （e_b = ∫_{z>0} Y_b·z dω、地表の照度）。c = c₁ + F·c₁ + F²·c₁ + … = (I − F)⁻¹·c₁。
//   - 大気が天頂の周りに回転対称なので F は m ごとのブロック（m = 0: b = 0, 1, 3 の 3×3、m = 1: b = 2, 4 の 2×2、
//     m = 2: b = 5 の 1×1）に分かれ、μs によらない。F は高度（行）ごとに CSTransfer が天頂角の 1 次元の求積で作り、
//     CSMain（テクセルごと）は c₁ だけを積む。
static const uint kShRings = 32;
static const uint kShAzimuths = 32;
// [POINT] 場は太陽と天頂を含む面について鏡映対称（基底は y² だけを使う）なので、太陽中心の格子の方位 φ と −φ は
//         同じ値になる。φ ∈ (0, π) の半分（方位 16、512 方向）だけを積み、重みを 2 倍にする（格子の方位は
//         φ_i = 2π(i + 0.5)/32 で、i と 31 − i が対になる）。結果は 1,024 方向とほぼ同じ（差 < 10⁻⁵）で、時間は半分
static const uint kShAzimuthsIntegrated = kShAzimuths / 2;
static const float kShAzimuthWeight = 2.0 * 2.0 * kPi / float(kShAzimuths);
// 128 スレッド × 4 方向。集計は WaveActiveSum → ウェーブごとの和を groupshared に置き、スレッド 0 が足す
// （64 スレッド × 8 方向・groupshared の総当たりの集計より 2 割速い。V8、design.md §14 Phase 4.11）
static const uint kShThreads = 128;
static const uint kShDirectionsPerThread = kShRings * kShAzimuthsIntegrated / kShThreads;  // 4
// F の表（CSTransfer → CSMain、行 = 高度）の並び: F の m = 0 のブロック（3×3、行優先）、m = 1（2×2）、m = 2（1）、
// 地表の G（m = 0 の 3）
static const uint kShTransferF0 = 0;
static const uint kShTransferF1 = 9;
static const uint kShTransferF2 = 13;
static const uint kShTransferG = 14;
static const uint kShTransferCount = 17;
// F の天頂角の求積: 地平線で分けて、地面側・空側にそれぞれ 64 本（1 スレッド 1 本）
static const uint kShZenithGround = 64;
static const uint kShZenithSky = 64;
static const uint kShTransferThreads = kShZenithGround + kShZenithSky;
// ブロックの係数の番号と帯 l
static const uint kShBlock0Band[3] = {0, 1, 2};
static const uint kShBlock1Band[2] = {1, 2};
// e_b = ∫_{z>0} Y_b·z dω（m = 0 のブロック b = 0, 1, 3。m > 0 は 0）
static const float kShHemisphereCosine[3] = {0.886226925, 1.023326708, 0.495415912};
static const float kShDeterminantFloor = 1e-4;

AtmosphereLuts MakeShLuts()
{
    AtmosphereLuts luts;
    luts.transmittanceSrv = gTransmittanceLutSrv;
    luts.transmittanceSize = gTransmittanceLutSize;
    luts.multiScatteringSrv = kInvalidDescriptorIndex;
    luts.multiScatteringSize = 0.0;
    [unroll] for (uint l = 0; l < 3; ++l) {
        luts.mieChi[l] = gPhaseMoments[l].rgb;
        luts.mie1Chi[l] = gPhaseMoments[3 + l].rgb;
    }
    return luts;
}

IntegrationSettings MakeShSettings()
{
    IntegrationSettings settings;
    settings.stepCount = gStepCount;
    settings.sunTransmittanceMode = gSunTransmittanceMode;
    settings.nestedSteps = gNestedSteps;
    settings.multipleScattering = false;
    settings.isotropicPhase = false;  // 1 回目の散乱は実際の位相関数
    settings.groundRadiance = true;
    settings.uniformSteps = gUniformSteps != 0;
    settings.separateMiePhase = false;
    settings.maxDistanceKm = kNoDistanceLimitKm;
    DisableOpaqueShadow(settings);
    return settings;
}

// ---------------------------------------------------------------------------
// F の表: 1 グループ = 1 行（高度）、128 スレッド = 天頂角の 128 本
// ---------------------------------------------------------------------------
groupshared float3 gsShTransfer[kShTransferCount][kShTransferThreads];

// [POINT] F は光線の天頂角だけで決まる量（Ã、T_g）の角度積分なので、方位 φ の積分を解析的に済ませ、天頂角の余弦 z
//         だけで積む。局所の枠で x = s·cosφ、y = s·sinφ（s² = 1 − z²）とすると ∫Y_a·Y_b dφ は m の違う組で 0、
//         m = 0 は 2π·Y_a·Y_b、m = 1 は π·s²·f_a·f_b（Y = f·x）、m = 2 は π·Y22²·s⁴。Ã は地平線 z_h = −√(1 − (R/r)²)
//         で不連続（地面に当たる光線は短い）で、近くで急に変わるので、地平線で分けて z = z_h ∓ (1 ± z_h)·s²（s は
//         中点則）で寄せる。太陽中心の格子で積むと、μs ごとに地平線付近の刻みが変わり、F が同じ行の中でばらついた
//         （μs = ±1 の列で最大。Phase 4.11 の最初の版）。
[numthreads(kShTransferThreads, 1, 1)]
void CSTransfer(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint row = group.x;
    const uint t = thread.x;
    const float2 lutSize = float2(gWidth, gHeight);
    float r;
    float muSun;
    MultiScatteringUvToRMuSun(gAtmosphere, float2(0.5, float(row) + 0.5) / lutSize, lutSize, r, muSun);
    const float bottom = gAtmosphere.bottomRadiusKm;
    // 1 − (R/r)² を桁落ちなく（地表の行は r − R = 1 m）
    const float zh = -sqrt(saturate((r - bottom) * (r + bottom) / (r * r)));
    const bool groundSide = t < kShZenithGround;
    const float s = groundSide ? (float(t) + 0.5) / float(kShZenithGround)
                               : (float(t - kShZenithGround) + 0.5) / float(kShZenithSky);
    const float span = groundSide ? 1.0 + zh : 1.0 - zh;
    const float z = groundSide ? zh - span * s * s : zh + span * s * s;
    const float dz = span * 2.0 * s / float(groundSide ? kShZenithGround : kShZenithSky);
    const float s2 = saturate(1.0 - z * z);

    const AtmosphereLuts luts = MakeShLuts();
    IntegrationSettings settings = MakeShSettings();
    // Ã と T_g は太陽によらない。太陽方向の透過率（輝度にだけ効く）は LUT にして、入れ子の設定でも軽くする
    settings.sunTransmittanceMode = SUN_TRANSMITTANCE_LUT;
    const float3 origin = float3(0.0, r, 0.0);
    const float3 dir = float3(sqrt(s2), z, 0.0);
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, luts, settings, origin, dir,
                                                                float3(0.0, 1.0, 0.0));
    // Ã_l = Σ_k χ_{k,l}·∫T·σs,k dt（Rayleigh の χ は (1, 0, 0.1)）
    float3 bandAs1[3];
    bandAs1[0] = result.rayleighAs1 + luts.mieChi[0] * result.mieAs1;
    bandAs1[1] = luts.mieChi[1] * result.mieAs1;
    bandAs1[2] = 0.1 * result.rayleighAs1 + luts.mieChi[2] * result.mieAs1;
#if MIE_GENERAL
    [unroll] for (uint l = 0; l < 3; ++l) {
        bandAs1[l] += luts.mie1Chi[l] * result.mie1As1;
    }
#endif
    // 方位 φ = 0 での基底の値（m = 0 は φ によらない、m = 1 は ×cosφ、m = 2 は ×cos2φ）
    const float f0[3] = {kMsShY00, kMsShY1 * z, kMsShY20 * (3.0 * z * z - 1.0)};
    const float f1[2] = {kMsShY1, kMsShY21 * z};
    const float w0 = 2.0 * kPi * dz;
    const float w1 = kPi * s2 * dz;
    [unroll] for (uint i0 = 0; i0 < 3; ++i0) {
        [unroll] for (uint j0 = 0; j0 < 3; ++j0) {
            gsShTransfer[kShTransferF0 + i0 * 3 + j0][t] = w0 * f0[i0] * f0[j0] * bandAs1[kShBlock0Band[j0]];
        }
        gsShTransfer[kShTransferG + i0][t] = result.hitGround ? w0 * f0[i0] * result.transmittance : 0.0;
    }
    [unroll] for (uint i1 = 0; i1 < 2; ++i1) {
        [unroll] for (uint j1 = 0; j1 < 2; ++j1) {
            gsShTransfer[kShTransferF1 + i1 * 2 + j1][t] = w1 * f1[i1] * f1[j1] * bandAs1[kShBlock1Band[j1]];
        }
    }
    gsShTransfer[kShTransferF2][t] = kPi * kMsShY22 * kMsShY22 * s2 * s2 * dz * bandAs1[2];
    GroupMemoryBarrierWithGroupSync();
    if (t < kShTransferCount) {
        float3 sum = 0.0;
        for (uint i = 0; i < kShTransferThreads; ++i) {
            sum += gsShTransfer[t][i];
        }
        RWTexture2D<float4> transfer = ResourceDescriptorHeap[gTransferUav];
        transfer[uint2(t, row)] = float4(sum, 0.0);
    }
}

// ---------------------------------------------------------------------------
// 係数: 1 テクセル = 1 スレッドグループ（128 スレッド × 4 方向）。group.y + gRowOffset が高度の行
// ---------------------------------------------------------------------------
groupshared float3 gsShWave[kMsShCount][kShThreads / 4];  // ウェーブごとの和（ウェーブの幅は 4 以上）

[numthreads(kShThreads, 1, 1)]
void CSMain(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint2 texel = uint2(group.x, group.y + gRowOffset);
    const uint t = thread.x;
    const float2 lutSize = float2(gWidth, gHeight);

    float r;
    float muSun;
    MultiScatteringUvToRMuSun(gAtmosphere, (float2(texel) + 0.5) / lutSize, lutSize, r, muSun);
    // 局所座標: 点は +Y 軸上、太陽は XY 平面内（x ≥ 0）。SH の枠は z = +Y、x = +X（μs = ±1 でも退化しない）
    const float3 origin = float3(0.0, r, 0.0);
    const float sinSun = sqrt(saturate(1.0 - muSun * muSun));
    const float3 sunDir = float3(sinSun, muSun, 0.0);
    // 方向の格子の枠: z' = 太陽、x' = (−μs, sinθs, 0)、y' = +Z（reference/ms_sh.py の sun_grid と同じ）
    const float3 gridX = float3(-muSun, sinSun, 0.0);
    const float3 gridY = float3(0.0, 0.0, 1.0);

    const AtmosphereLuts luts = MakeShLuts();
    const IntegrationSettings settings = MakeShSettings();

    float3 acc[kMsShCount];
    [unroll] for (uint k = 0; k < kMsShCount; ++k) {
        acc[k] = 0.0;
    }
    for (uint j = 0; j < kShDirectionsPerThread; ++j) {
        const uint d = t + j * kShThreads;
        const uint ring = d / kShAzimuthsIntegrated;
        const uint azimuth = d % kShAzimuthsIntegrated;
        const float u0 = float(ring) / float(kShRings);
        const float u1 = float(ring + 1) / float(kShRings);
        const float uc = (float(ring) + 0.5) / float(kShRings);
        const float theta = kPi * uc * uc;
        const float phi = 2.0 * kPi * (float(azimuth) + 0.5) / float(kShAzimuths);
        const float weight = (cos(kPi * u0 * u0) - cos(kPi * u1 * u1)) * kShAzimuthWeight;
        const float3 local = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        const float3 dir = local.x * gridX + local.y * gridY + local.z * sunDir;

        const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, luts, settings, origin, dir, sunDir);
        // SH の枠: z = +Y、x = +X、y = z × x = −Z（基底は y² だけを使う）
        const MsShBasis basis = EvalMsShBasis(dir.x, dir.y, dir.z * dir.z);
        [unroll] for (uint b = 0; b < kMsShCount; ++b) {
            acc[b] += weight * basis.y[b] * result.luminance;
        }
    }

    float3 c1[kMsShCount];
    const uint lanes = WaveGetLaneCount();
    [unroll] for (uint k2 = 0; k2 < kMsShCount; ++k2) {
        const float3 waveSum = WaveActiveSum(acc[k2]);
        if (WaveIsFirstLane()) {
            gsShWave[k2][t / lanes] = waveSum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t != 0) {
        return;
    }
    [unroll] for (uint b = 0; b < kMsShCount; ++b) {
        c1[b] = 0.0;
        for (uint w = 0; w < kShThreads / lanes; ++w) {
            c1[b] += gsShWave[b][w];
        }
    }
    Texture2D<float4> transferTable = ResourceDescriptorHeap[gTransferSrv];
    float3 transfer[kShTransferCount];
    [unroll] for (uint k3 = 0; k3 < kShTransferCount; ++k3) {
        transfer[k3] = transferTable.Load(int3(k3, texel.y, 0)).rgb;
    }
    // m = 0 のブロック: A = I − F0 − (ρ/π)·G ⊗ e（チャンネルごとの 3×3 を float3 のまま解く）
    float3 a[3][3];
    [unroll] for (uint i = 0; i < 3; ++i) {
        [unroll] for (uint j = 0; j < 3; ++j) {
            const float3 ground = gAtmosphere.groundAlbedo / kPi * transfer[kShTransferG + i] * kShHemisphereCosine[j];
            a[i][j] = (i == j ? 1.0 : 0.0) - transfer[kShTransferF0 + i * 3 + j] - ground;
        }
    }
    const float3 rhs0[3] = {c1[0], c1[1], c1[3]};
    const float3 cof00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    const float3 cof01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    const float3 cof02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    const float3 det0 = a[0][0] * cof00 + a[0][1] * cof01 + a[0][2] * cof02;
    // 余因子行列（adj(A) = 余因子の転置）で x = adj(A)·b / det
    const float3 x0 = (cof00 * rhs0[0] + (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * rhs0[1] +
                       (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * rhs0[2]) / det0;
    const float3 x1 = (cof01 * rhs0[0] + (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * rhs0[1] +
                       (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * rhs0[2]) / det0;
    const float3 x3 = (cof02 * rhs0[0] + (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * rhs0[1] +
                       (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * rhs0[2]) / det0;
    // m = 1 のブロック（2×2）と m = 2（1×1）
    const float3 b00 = 1.0 - transfer[kShTransferF1 + 0];
    const float3 b01 = -transfer[kShTransferF1 + 1];
    const float3 b10 = -transfer[kShTransferF1 + 2];
    const float3 b11 = 1.0 - transfer[kShTransferF1 + 3];
    const float3 det1 = b00 * b11 - b01 * b10;
    const float3 x2 = (b11 * c1[2] - b01 * c1[4]) / det1;
    const float3 x4 = (b00 * c1[4] - b10 * c1[2]) / det1;
    const float3 x5 = c1[5] / max(1.0 - transfer[kShTransferF2], 1e-3);
    // [POINT] I − F がほぼ特異なとき（光学的に非常に厚い大気・アルベド 1 など）は、対角だけで閉じる
    //         （各係数を独立な等比級数とみなす。Hillaire 2020 の 1/(1 − f_ms) と同じ形）
    const bool3 singular0 = abs(det0) < kShDeterminantFloor;
    const bool3 singular1 = abs(det1) < kShDeterminantFloor;
    float3 c[kMsShCount];
    c[0] = select(singular0, c1[0] / max(a[0][0], 1e-3), x0);
    c[1] = select(singular0, c1[1] / max(a[1][1], 1e-3), x1);
    c[3] = select(singular0, c1[3] / max(a[2][2], 1e-3), x3);
    c[2] = select(singular1, c1[2] / max(b00, 1e-3), x2);
    c[4] = select(singular1, c1[4] / max(b11, 1e-3), x4);
    c[5] = x5;

    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    [unroll] for (uint b2 = 0; b2 < kMsShCount; ++b2) {
        output[uint2(texel.x + b2 * gWidth, texel.y)] = float4(c[b2], 1.0);
    }
    if (gDebugUav != kInvalidDescriptorIndex) {
        // 検証用: (r [km], μs, 3 チャンネルの m = 0 のブロックの行列式の最小, F0_00 の G（地表の項を除く）)。
        // F は行ごとの表なので .w は行の中で同じ値
        RWTexture2D<float4> debugOutput = ResourceDescriptorHeap[gDebugUav];
        debugOutput[texel] = float4(r, muSun, min(det0.x, min(det0.y, det0.z)), transfer[kShTransferF0].g);
    }
}

#endif // MS_SH
