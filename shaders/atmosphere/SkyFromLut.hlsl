// LUT による合成（Hillaire 2020 の実用パス）:
//   空のピクセル  … Sky-View LUT から空 + 太陽ディスク
//   地形のピクセル … 空気遠近法のボリュームから内散乱を、ボリュームか Transmittance LUT の比（§5.7）から透過率を
//                    求めて地形の色に掛ける
// Mie の単散乱だけは位相関数を LUT の外でピクセルごとに掛ける（本実装の変更点。SkyViewLut.hlsl 参照）。
// 地形の影は、ボリュームに入れる（方式 0 / 1）か、ここでピクセルごとに差し引く（方式 5）。design.md §5.6。
#include "atmosphere/SkyPassConstants.hlsli"

// 空気遠近法のボリュームを、画面 UV と距離 [km] で引いた値（カメラから距離 distanceKm までの累積）
struct AerialPerspectiveSample
{
    float3 rayleigh;    // Rayleigh の単散乱 [sr⁻¹]
    float3 mieNoPhase;  // Mie の単散乱（位相関数なし）[無次元]
    float3 multiple;    // 多重散乱 [sr⁻¹]
    float transmittance;  // 透過率の RGB 平均
};

// ボリュームのテクスチャ座標と、最初のスライスより手前のフェードの重み
float3 AerialPerspectiveUvw(float2 uv, float distanceKm, out float fade)
{
    // [POINT] スライス座標 s = 距離 / 4 km。ボリュームは s = ((z + 0.5)/32)² × 32 の 2 次配置で作ったので、
    //         テクスチャ座標は w = √(s / 32)（参照実装と同じ）。カメラのすぐ近く（s < 0.5）は最初の
    //         スライスより手前なので、内散乱と不透明度を 2s 倍してフェードする（段差の防止）。
    float slice = distanceKm / gApKmPerSlice;
    fade = slice < 0.5 ? saturate(slice * 2.0) : 1.0;
    slice = max(slice, 0.5);
    return float3(uv, sqrt(slice / gApSliceCount));
}

AerialPerspectiveSample SampleAerialPerspective(float2 uv, float distanceKm)
{
    float fade;
    const float3 uvw = AerialPerspectiveUvw(uv, distanceKm, fade);
    Texture3D<float4> ap = ResourceDescriptorHeap[gAerialPerspectiveSrv];
    Texture3D<float4> apMie = ResourceDescriptorHeap[gAerialPerspectiveMieSrv];
    Texture3D<float4> apMs = ResourceDescriptorHeap[gAerialPerspectiveMsSrv];
    const float4 rayleigh = ap.SampleLevel(gLinearClamp, uvw, 0.0);
    AerialPerspectiveSample s;
    s.rayleigh = fade * rayleigh.rgb;
    s.mieNoPhase = fade * apMie.SampleLevel(gLinearClamp, uvw, 0.0).rgb;
    s.multiple = fade * apMs.SampleLevel(gLinearClamp, uvw, 0.0).rgb;
    // 透過率は RGB の平均（論文・参照実装と同じ近似）。フェードは不透明度 (1 − T) に掛ける。
    s.transmittance = 1.0 - fade * (1.0 - rayleigh.a);
    return s;
}

bool IsSunVisible(OpaqueShadowParameters shadow, float3 viewDir, float distanceKm)
{
    return SampleOpaqueShadow(shadow, viewDir * (distanceKm * 1000.0), 0.0, 0.0) >= 0.5;
}

// [POINT] 方式 5（design.md §5.6、research.md §10）: 影に入る区間の単散乱だけを、ピクセルごとに差し引く。
//         単散乱の累積 F(t)（ボリュームの値、カメラから距離 t まで、影なし）を使うと、影ありの単散乱は
//           L = F(d) − Σ [F(b_k) − F(a_k)]   （[a_k, b_k] は視線上で影に入る区間）
//         区間はピクセルの視線とシャドウマップで決めるので、影の形は画面とシャドウマップの解像度で出る。
//         ボリュームが担うのは、なめらかな F だけ。視線上の N 点で影を調べ、影の有無が変わった所だけ
//         二分探索で詰めて、その位置で F を引く（ボリュームを引くのは区間の端の数だけ）。
void ShadowedSingleScattering(float2 uv, float3 viewDir, float distanceKm, AerialPerspectiveSample end,
                              out float3 rayleigh, out float3 mieNoPhase)
{
    const OpaqueShadowParameters shadow = LoadOpaqueShadow();
    const uint steps = max(gApPerPixelSteps, 1u);
    float3 sumRayleigh = 0.0;
    float3 sumMie = 0.0;
    float3 startRayleigh = 0.0;  // 今の区間の始点の F（カメラ = 0）
    float3 startMie = 0.0;
    bool segmentLit = true;
    float previousT = 0.0;
    for (uint k = 0; k < steps; ++k) {
        // 視線上の点は 2 次配置（カメラの近くを細かく。近くの物の細い影を拾うため）
        const float u = (float(k) + 0.5) / float(steps);
        const float t = distanceKm * u * u;
        const bool lit = IsSunVisible(shadow, viewDir, t);
        if (k == 0) {
            segmentLit = lit;
        } else if (lit != segmentLit) {
            // 影の境界を、前の点と今の点のあいだで二分探索する
            float lo = previousT;
            float hi = t;
            for (uint b = 0; b < gApPerPixelBisectSteps; ++b) {
                const float mid = 0.5 * (lo + hi);
                if (IsSunVisible(shadow, viewDir, mid) == segmentLit) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            const AerialPerspectiveSample edge = SampleAerialPerspective(uv, 0.5 * (lo + hi));
            if (segmentLit) {
                sumRayleigh += edge.rayleigh - startRayleigh;
                sumMie += edge.mieNoPhase - startMie;
            }
            startRayleigh = edge.rayleigh;
            startMie = edge.mieNoPhase;
            segmentLit = lit;
        }
        previousT = t;
    }
    if (segmentLit) {
        sumRayleigh += end.rayleigh - startRayleigh;
        sumMie += end.mieNoPhase - startMie;
    }
    // 補間した F が単調でない所で、わずかに負になるのを防ぐ
    rayleigh = max(sumRayleigh, 0.0);
    mieNoPhase = max(sumMie, 0.0);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float3 viewDir = OutputViewDirection(id.xy);
    // [POINT] Mie の単散乱は、LUT の「位相関数を除いた積分」[無次元] に、このピクセルの正確な散乱角での
    //         位相関数 p_M(θ) [sr⁻¹] を掛けて足す。LUT のテクセルより細い光輪（g → 1）も形が崩れない
    //         （Sky-View LUT に位相関数を焼き込むと、g = 0.99 でピークが縦に伸び 6〜97% に暗くなった。design.md §14）。
    const float3 phaseMie = MiePhase(gAtmosphere, dot(viewDir, gSunDirection));

    if (gOutputMode == OUTPUT_CAMERA_FB && gDepthSrv != kInvalidDescriptorIndex) {
        Texture2D<float> depthTexture = ResourceDescriptorHeap[gDepthSrv];
        const float depth = depthTexture.Load(int3(id.xy, 0));
        if (depth > 0.0) {
            // 地形のピクセル: 空気遠近法のボリュームを、画面 UV と距離で引く
            const float distanceKm = DepthToDistanceKm(id.xy, depth);
            const float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);
            const AerialPerspectiveSample end = SampleAerialPerspective(uv, distanceKm);
            float3 rayleigh = end.rayleigh;
            float3 mieNoPhase = end.mieNoPhase;
            if (gApShadowMethod == AP_SHADOW_PER_PIXEL && gShadowFarSrv != kInvalidDescriptorIndex) {
                ShadowedSingleScattering(uv, viewDir, distanceKm, end, rayleigh, mieNoPhase);
            }
            const float3 inscatter = rayleigh + end.multiple + phaseMie * mieNoPhase;
            // [POINT] 透過率（design.md §5.7）: ボリュームの A は RGB の平均で、遠景で R が暗く B が明るくなる
            //         （5〜20 km で ±6%、V3c）。Transmittance LUT の比なら RGB のまま求まる。スライス 0 の手前の
            //         フェードはボリュームのためのものなので、比の透過率には掛けない（d → 0 で正しく 1 に近づく）。
            float3 transmittance = end.transmittance.xxx;
            if (gApTransmittanceMode == AP_TRANSMITTANCE_LUT_RATIO) {
                const float r = length(gCameraPosKm);
                const float mu = dot(gCameraPosKm, viewDir) / r;
                Texture2D<float4> lut = ResourceDescriptorHeap[gTransmittanceLutSrv];
                transmittance = TransmittanceToPoint(gAtmosphere, lut, gLinearClamp, r, mu, distanceKm,
                                                     gTransmittanceLutSize);
            } else if (gApTransmittanceMode == AP_TRANSMITTANCE_VOLUME_RGB) {
                // 方式 2: ボリュームに積分した RGB の透過率。フェードは方式 0 と同じく不透明度 (1 − T) に掛ける
                float fade;
                const float3 uvw = AerialPerspectiveUvw(uv, distanceKm, fade);
                Texture3D<float4> apT = ResourceDescriptorHeap[gAerialPerspectiveTransmittanceSrv];
                transmittance = 1.0 - fade * (1.0 - apT.SampleLevel(gLinearClamp, uvw, 0.0).rgb);
            }
            ApplyAerialPerspective(id.xy, inscatter, transmittance);
            return;
        }
    }

    float3 rayleighAndMs;
    float3 mieNoPhase;
    SampleSkyViewLuts(gCameraPosKm, viewDir, rayleighAndMs, mieNoPhase);
    float3 luminance = rayleighAndMs + phaseMie * mieNoPhase;
    if ((gFlags & SKY_FLAG_SUN_DISK) != 0) {
        luminance += SunDiskPerLux(gCameraPosKm, viewDir);
    }
    WriteSkyOutput(id.xy, luminance, viewDir);
}
