// 視線に沿った散乱の積分（Hillaire 2020 §4 式 (1)〜(4)、§5.5 式 (11)）。
// Multi-scattering LUT・Sky-View LUT・リファレンスのレイマーチで共通に使う。
//
// 単位: 照度 1 lx あたりの量で計算する（輝度なら sr⁻¹ = cd/m² per lx）。
//   L_scat = ∫ T_view(t) · σs(t) · [ p(θ) · T_sun(t) + Ψ_ms(t) ] dt  (+ 地表の反射)
#ifndef ATMOSPHERE_INTEGRATE_HLSLI
#define ATMOSPHERE_INTEGRATE_HLSLI

#include "common/Bindless.hlsli"
#include "common/Shadow.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"
#include "atmosphere/Phase.hlsli"
#if MS_SH
#include "atmosphere/MultiScatteringSh.hlsli"
#endif

#define SUN_TRANSMITTANCE_LUT 0
#define SUN_TRANSMITTANCE_NESTED 1

// Mie の位相関数の表の点の数（θ = π·u²、u は 0〜1 の等間隔。reference/aerosol_models.py の PHASE_TABLE_COUNT）
static const uint kMiePhaseTableCount = 2049;

// Mie の成分 component（0 / 1）の位相関数 [sr⁻¹]（RGB）。表があれば表（design.md §5.9、エアロゾルのプリセット）、
// なければ Cornette-Shanks。
// [POINT] 表は θ = π·u² で刻む（前方のピークを細かく）。u = √(θ/π) で線形補間する（パストレーサーの表の媒質・
//         Python の PhaseTable と同じ補間）。
float3 MiePhase(AtmosphereParameters a, uint component, float cosTheta)
{
    const uint tableSrv = component == 0 ? a.miePhaseTableSrv : a.mie1PhaseTableSrv;
    if (tableSrv == kInvalidDescriptorIndex) {
        return CornetteShanksPhase(component == 0 ? a.miePhaseG : a.mie1PhaseG, cosTheta).xxx;
    }
    StructuredBuffer<float4> table = ResourceDescriptorHeap[tableSrv];
    const float u = sqrt(acos(clamp(cosTheta, -1.0, 1.0)) / kPi);
    const float f = u * float(kMiePhaseTableCount - 1);
    const uint i = min(uint(f), kMiePhaseTableCount - 2);
    return lerp(table[i].rgb, table[i + 1].rgb, f - float(i));
}

struct AtmosphereLuts
{
    uint   transmittanceSrv;
    float2 transmittanceSize;
    uint   multiScatteringSrv;    // MS_SH なら SH の係数のアトラス（32×32 のタイル 6 枚、design.md §5.11）
    float2 multiScatteringSize;   // 1 枚のタイルの大きさ（32×32）
#if MS_SH
    // Mie の成分 0・1 の位相関数の Legendre のモーメント χ_0〜χ_2（RGB）。Rayleigh は (1, 0, 0.1) で固定
    float3 mieChi[3];
    float3 mie1Chi[3];
#endif
};

// 不透明物（地形・球）の太陽の影。2 段のカスケード（design.md §5.5、§5.6）
struct OpaqueShadowParameters
{
    uint farSrv;          // 全体のカスケード。無効値なら影なし
    uint nearSrv;         // 近景のカスケード。無効値なら全体だけ
    float4x4 farToUvz;    // カメラ相対の位置 [m] → (u, v, 深度)
    float4x4 nearToUvz;
};

static const float kGoldenAngle = 2.39996323;  // 黄金角 [rad]（円盤の点の並び）
static const uint kShadowFilterTaps = 12;

// 点（カメラ相対 [m]）で太陽が見える割合 [0, 1]。
// filterRadiusM > 0 なら、光源から見た半径 filterRadiusM [m] の円盤を 12 点の PCF で平均する（design.md §5.6 の
// 方式 1）。rotation: 円盤の点の並びを回す角度 [rad]（ステップごとに変えて、点の並びの模様を目立たなくする）
float SampleOpaqueShadow(OpaqueShadowParameters p, float3 cameraRelativeM, float filterRadiusM, float rotation)
{
    if (p.farSrv == kInvalidDescriptorIndex) {
        return 1.0;
    }
    // [POINT] 近景のカスケードは、点が u, v, 深度とも範囲の内側のときだけ使う。近景の範囲より光源から遠い点は、
    //         近景の奥行きの範囲の外（さらに遠い側）の遮蔽物を取りこぼすので、全体のカスケードを使う。
    uint srv = p.farSrv;
    float4x4 toUvz = p.farToUvz;
    if (p.nearSrv != kInvalidDescriptorIndex) {
        const float3 uvzNear = mul(p.nearToUvz, float4(cameraRelativeM, 1.0)).xyz;
        if (all(uvzNear > 0.01) && all(uvzNear < 0.99)) {
            srv = p.nearSrv;
            toUvz = p.nearToUvz;
        }
    }
    const float3 uvz = mul(toUvz, float4(cameraRelativeM, 1.0)).xyz;
    if (filterRadiusM <= 0.0) {
        return SampleShadowMap(srv, uvz);
    }
    // 光源空間の 1 m あたりの (u, v) = 行列の 1 行目・2 行目の長さ（軸 / 範囲の幅）
    const float2 radiusUv = filterRadiusM * float2(length(toUvz[0].xyz), length(toUvz[1].xyz));
    Texture2D<float> map = ResourceDescriptorHeap[NonUniformResourceIndex(srv)];
    float width;
    float height;
    map.GetDimensions(width, height);
    if (max(radiusUv.x * width, radiusUv.y * height) < 0.5) {
        return SampleShadowMap(srv, uvz);  // 円盤が半テクセルより小さい（近い所）: 1 回で同じ
    }
    float sum = 0.0;
    [unroll] for (uint k = 0; k < kShadowFilterTaps; ++k) {
        // Vogel の螺旋: 半径 √((k + 0.5)/N)、角度 k × 黄金角（円盤の中に一様に並ぶ）
        const float r = sqrt((float(k) + 0.5) / float(kShadowFilterTaps));
        const float angle = float(k) * kGoldenAngle + rotation;
        sum += SampleShadowMap(srv, float3(uvz.xy + r * float2(cos(angle), sin(angle)) * radiusUv, uvz.z));
    }
    return sum / float(kShadowFilterTaps);
}

struct IntegrationSettings
{
    uint stepCount;
    uint sunTransmittanceMode;  // SUN_TRANSMITTANCE_*
    uint nestedSteps;           // SUN_TRANSMITTANCE_NESTED のときの太陽方向の区間数
    bool multipleScattering;    // σs·Ψ_ms（Multi-scattering LUT）を加える
    bool isotropicPhase;        // 位相関数を 1/(4π) にする（Multi-scattering LUT の生成用）
    bool groundRadiance;        // 地表に当たったら地表のランバート反射を加える（Multi-scattering LUT の生成用）
    bool uniformSteps;          // true: 区間を等間隔に置く（比較用）/ false: 2 次配置（既定。全パス）
    bool separateMiePhase;      // true: Mie の単散乱を luminance に入れず、位相関数なしで成分ごとに mie*SingleNoPhase に分ける（Sky-View LUT・空気遠近法用）
    float maxDistanceKm;        // 積分する距離の上限 [km]（地形・空気遠近法のボクセルまで）。上限なしは kNoDistanceLimitKm
    // 不透明物（地形）の太陽の影（design.md §5.5）。farSrv が無効値なら影なし。光線の原点がカメラのときだけ使える
    OpaqueShadowParameters opaqueShadow;
    // 影をぼかす円盤の半径 / カメラからの距離 [m/m]（0 なら 1 回の比較。design.md §5.6 の方式 1）
    float opaqueShadowFilterRadiusPerM;
};

// 影を使わない積分の設定（opaqueShadow* を埋める）
void DisableOpaqueShadow(inout IntegrationSettings s)
{
    s.opaqueShadow.farSrv = kInvalidDescriptorIndex;
    s.opaqueShadow.nearSrv = kInvalidDescriptorIndex;
    s.opaqueShadow.farToUvz = (float4x4)0;
    s.opaqueShadow.nearToUvz = (float4x4)0;
    s.opaqueShadowFilterRadiusPerM = 0.0;
}

static const float kNoDistanceLimitKm = 1e9;

struct ScatteringResult
{
    float3 luminance;         // 単位照度あたりの輝度 [sr⁻¹]（separateMiePhase なら Mie の単散乱を除く）
    float3 mieSingleNoPhase;  // ∫ T_view·σs_M·T_sun dt [無次元]。× p_M(θ) [sr⁻¹] で Mie の単散乱の輝度（単位照度あたり）
    // Mie の成分 1 の同じ量（1 成分なら 0）。成分ごとに鉛直分布が違うと視線上で位相関数の混ざり方が変わるので、
    // 位相関数を除いた積分は成分ごとに持つ（design.md §5.9）
    float3 mie1SingleNoPhase;
    float3 multiScatAs1;      // f_ms の被積分量: ∫ T_view·σs dt（等方な単位入射光が散乱して戻る割合）
    float3 multipleScattering;  // luminance のうち多重散乱の分 [sr⁻¹]（影の対象外。空気遠近法で単散乱と分けて持つ）
    float3 transmittance;     // カメラ → 積分区間の終端の透過率
    float  opaqueShadowMean;  // 視線上の点（各区間の中点）で太陽が見える割合の平均（影なしなら 1。検証用）
#if MS_SH
    // SH の Multi-scattering LUT の閉じ方（伝達行列 F）用: 成分ごとの ∫ T_view·σs dt と、地表に当たったか
    float3 rayleighAs1;
    float3 mieAs1;
    float3 mie1As1;
    bool   hitGround;
#endif
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

#if MS_SH
// 点 p で視線 dir（見る向き）へ散乱される多重散乱の光源 [km⁻¹·sr⁻¹]（単位照度あたり。design.md §5.11）。
//   J = Σ_k σs,k · Σ_b χ_{k,l(b)} · c_b(r, μs) · Y_b(dir)
// c_b は入ってくる散乱光の場（散乱 1 回以上）の SH の係数、χ_{k,l} は成分 k の位相関数の Legendre のモーメント
// （Funk–Hecke: 位相関数との畳み込みは帯 l ごとに χ_l 倍）。Rayleigh は χ = (1, 0, 0.1)。
float3 MultipleScatteringSourceSh(AtmosphereParameters a, AtmosphereLuts luts, float3 p, float3 dir, float3 sunDir,
                                  MediumSample m)
{
    const float r = length(p);
    const float3 up = p / r;
    const float muSun = dot(up, sunDir);
    const float2 uv = MultiScatteringRMuSunToUv(a, r, muSun, luts.multiScatteringSize);
    Texture2D<float4> lut = ResourceDescriptorHeap[luts.multiScatteringSrv];
    float3 c[kMsShCount];
    [unroll] for (uint b = 0; b < kMsShCount; ++b) {
        c[b] = lut.SampleLevel(gLinearClamp, MsShAtlasUv(uv, b), 0.0).rgb;
    }
    // [POINT] 局所の枠の x 軸 = 太陽の水平の向き。太陽が真上・真下だと定まらないが、そこでは場が天頂の周りに
    //         対称で m > 0 の係数が 0 なので、m > 0 の項を落とす
    const float3 sunHorizontal = sunDir - muSun * up;
    const float len2 = dot(sunHorizontal, sunHorizontal);
    const float azimuthValid = len2 > 1e-12 ? 1.0 : 0.0;
    const float3 xAxis = sunHorizontal * (azimuthValid * rsqrt(max(len2, 1e-30)));
    const float z = dot(dir, up);
    const float x = dot(dir, xAxis);
    const MsShBasis basis = EvalMsShBasis(x, z, max(1.0 - x * x - z * z, 0.0));
    const float3 band0 = c[0] * basis.y[0];
    const float3 band1 = c[1] * basis.y[1] + c[2] * basis.y[2];
    const float3 band2 = c[3] * basis.y[3] + c[4] * basis.y[4] + azimuthValid * c[5] * basis.y[5];
    float3 source = m.rayleighScattering * (band0 + 0.1 * band2) +
                    m.mieScattering * (luts.mieChi[0] * band0 + luts.mieChi[1] * band1 + luts.mieChi[2] * band2);
#if MIE_GENERAL
    source += m.mie1Scattering * (luts.mie1Chi[0] * band0 + luts.mie1Chi[1] * band1 + luts.mie1Chi[2] * band2);
#endif
    return source;
}
#endif

ScatteringResult IntegrateScatteredLuminance(AtmosphereParameters a, AtmosphereLuts luts, IntegrationSettings s,
                                             float3 origin, float3 dir, float3 sunDir)
{
    ScatteringResult result;
    result.luminance = 0.0;
    result.mieSingleNoPhase = 0.0;
    result.mie1SingleNoPhase = 0.0;
    result.multiScatAs1 = 0.0;
    result.multipleScattering = 0.0;
    result.transmittance = 1.0;
    result.opaqueShadowMean = 1.0;
#if MS_SH
    result.rayleighAs1 = 0.0;
    result.mieAs1 = 0.0;
    result.mie1As1 = 0.0;
    result.hitGround = false;
#endif

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
    // 地形・ボクセルまでの距離で打ち切る（その手前で終わるなら、惑星の地表には当たらない）
    if (s.maxDistanceKm < tEnd) {
        tEnd = max(s.maxDistanceKm, tStart);
        hitGround = false;
    }
#if MS_SH
    result.hitGround = hitGround;
#endif

    const float cosTheta = dot(dir, sunDir);
    const float phaseRayleigh = s.isotropicPhase ? kIsotropicPhase : RayleighPhase(cosTheta);
    const float3 phaseMie = s.isotropicPhase ? kIsotropicPhase.xxx : MiePhase(a, 0, cosTheta);
#if MIE_GENERAL
    // [POINT] HLSL の三項演算子は両辺を評価するので、成分 1 の位相関数は if で求める
    float3 phaseMie1 = kIsotropicPhase.xxx;
    if (!s.isotropicPhase && a.mieComponentCount > 1) {
        phaseMie1 = MiePhase(a, 1, cosTheta);
    }
#endif

    // [POINT] 光線上のサンプル配置（区間端）。LUT のマッピング（テクセル ↔ 方向）とは別の話。
    //   2 次配置 t_k = (k/N)²（Sky-View LUT・レイマーチ）: カメラの近くを細かくする。地表付近の Mie は
    //     スケールハイト 1.2 km で急減するので、等間隔（Sky-View LUT の 30 区間で 1 区間 ~5 km）だと最初の区間の
    //     中点で密度を過小評価し、太陽の周りの光輪が 10〜19% 暗くなった（Phase 3 の V3 で発見）。
    //   Multi-scattering LUT も既定は 2 次配置（40 区間、design.md §12 Q8）。今の大気の 20 区間では等間隔のほうが
    //     誤差が小さかった（2.7% vs 4.9%）が、エアロゾルのプリセット（AOD 0.1）では等間隔だと地表付近の濃い層を
    //     粗く刻み、Ψ_ms が最大 24〜34% ずれた（§14 Phase 4.8）。等間隔 t_k = k/N は比較用に残す。
    // 参照実装（sebh/UnrealEngineSkyAtmosphere）も 2 次配置は Sky-View LUT と画面のレイマーチだけで、そこでは
    // 区間数も光線の長さで変える。本実装は区間数を固定し、区間内のサンプル位置は中点（参照実装は 0.3）。
    const float range = tEnd - tStart;
    const float invSteps = 1.0 / float(s.stepCount);
    float3 throughput = 1.0;
    float visibilitySum = 0.0;
    for (uint i = 0; i < s.stepCount; ++i) {
        const float u0 = float(i) * invSteps;
        const float u1 = float(i + 1) * invSteps;
        const float t0 = tStart + range * (s.uniformSteps ? u0 : u0 * u0);
        const float t1 = tStart + range * (s.uniformSteps ? u1 : u1 * u1);
        const float dt = t1 - t0;
        const float t = 0.5 * (t0 + t1);
        const float3 p = origin + t * dir;
        const MediumSample m = SampleMedium(a, length(p) - a.bottomRadiusKm);
#if MIE_GENERAL
        const float3 scattering = m.rayleighScattering + m.mieScattering + m.mie1Scattering;
#else
        const float3 scattering = m.rayleighScattering + m.mieScattering;
#endif
        const float3 segmentTransmittance = exp(-m.extinction * dt);

        // 区間内の光源項 S [km⁻¹·sr⁻¹]: 単散乱（太陽の直達光）+ 多重散乱（式 (11)）
        float3 sunTransmittance = SunTransmittance(a, luts, s, p, sunDir);
        // [POINT] 不透明物の影（参照実装の SHADOWMAP_ENABLED と同じ）: 単散乱の太陽光だけを遮る。多重散乱の項
        //         （Ψ_ms は周りの空からの光）には掛けないので、影の中の空気はやや明るめになる（design.md §5.5）。
        //         点の位置はカメラ相対 [m] で求める（惑星座標 [km] から戻すと float の精度が ~0.5 m に落ちる）。
        float visibility = 1.0;
        if (s.opaqueShadow.farSrv != kInvalidDescriptorIndex) {
            const float tM = t * 1000.0;
            visibility = SampleOpaqueShadow(s.opaqueShadow, dir * tM, s.opaqueShadowFilterRadiusPerM * tM,
                                            float(i) * kGoldenAngle);
            sunTransmittance *= visibility;
        }
        visibilitySum += visibility;
        const float3 mieNoPhase = m.mieScattering * sunTransmittance;  // [km⁻¹]
#if MIE_GENERAL
        const float3 mie1NoPhase = m.mie1Scattering * sunTransmittance;
#endif
        float3 source = m.rayleighScattering * phaseRayleigh * sunTransmittance;
        if (!s.separateMiePhase) {
            source += mieNoPhase * phaseMie;
#if MIE_GENERAL
            source += mie1NoPhase * phaseMie1;
#endif
        }
        float3 multipleSource = 0.0;
        if (s.multipleScattering) {
#if MS_SH
            multipleSource = MultipleScatteringSourceSh(a, luts, p, dir, sunDir, m);
#else
            multipleSource = scattering * SampleMultipleScattering(a, luts, p, sunDir);
#endif
            source += multipleSource;
        }
        // [POINT] 区間内で媒質が一定とみなした解析積分（Hillaire の "energy-conserving" 積分）:
        //           ∫₀^dt S·e^(−σt·x) dx = S·(1 − e^(−σt·dt)) / σt = S·dt·(1 − e^(−x))/x、x = σt·dt
        //         少ないステップ数（Sky-View LUT は 30）でも、区間内の減衰を正しく扱える。
        //         参照実装の (S − S·T)/σt は x が小さいと桁落ちするので、OneMinusExpOverX を使う。
        const float3 segmentWeight = dt * OneMinusExpOverX(m.extinction * dt);
        result.luminance += throughput * source * segmentWeight;
        result.mieSingleNoPhase += throughput * mieNoPhase * segmentWeight;
#if MIE_GENERAL
        result.mie1SingleNoPhase += throughput * mie1NoPhase * segmentWeight;
#endif
        result.multiScatAs1 += throughput * scattering * segmentWeight;
        result.multipleScattering += throughput * multipleSource * segmentWeight;
#if MS_SH
        result.rayleighAs1 += throughput * m.rayleighScattering * segmentWeight;
        result.mieAs1 += throughput * m.mieScattering * segmentWeight;
#if MIE_GENERAL
        result.mie1As1 += throughput * m.mie1Scattering * segmentWeight;
#endif
#endif
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
    result.opaqueShadowMean = visibilitySum * invSteps;
    return result;
}

#endif // ATMOSPHERE_INTEGRATE_HLSLI
