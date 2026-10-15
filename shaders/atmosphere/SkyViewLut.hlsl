// Sky-View LUT の生成（Hillaire 2020 §5.3）。毎フレーム、カメラの高度と太陽の向きで作り直す。
//
// 空は全体として低周波（Rayleigh・多重散乱は滑らか）なので、全画面でレイマーチせず、
// 低解像度（192×108）の経緯度テクスチャに 1 回だけ積分して、画面ではそれを引く。
// 地表の反射は含まない（地面は Phase 4 の地形が描く）。
//
// 出力は 2 枚:
//   gOutputUav        : Rayleigh の単散乱 + 多重散乱の輝度 [sr⁻¹]（単位照度あたり）
//   gSkyViewMieLutUav : Mie の単散乱から位相関数を除いた ∫ T_view·σs_M·T_sun dt [無次元]
// [POINT] Mie の位相関数は g が大きいと太陽の周りに鋭いピークを持ち（g = 0.99 で半値全幅 0.88°）、
//         太陽の位置での LUT の縦の 1 テクセル（天頂角 35° で 2.6°）では表せない（design.md §14 の調査）。
//         太陽は平行光なので 1 本の視線の上で散乱角 θ は一定で、Mie の単散乱は
//         p_M(θ) × ∫ T_view·σs_M·T_sun dt に正確に分けられる。∫ の側は方向に対して滑らかなので LUT に入れ、
//         p_M(θ) は描画時にピクセルごとの正確な角度で掛ける（SkyFromLut.hlsl。Bruneton 2008/2017 と同じ考え方）。
#include "atmosphere/SkyPassConstants.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float2 lutSize = float2(gWidth, gHeight);
    const float r = max(length(gCameraPosKm), gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);

    float viewZenithCos;
    float lightViewCos;
    SkyViewUvToViewParams(gAtmosphere, (float2(id.xy) + 0.5) / lutSize, lutSize, r, viewZenithCos, lightViewCos);

    // [POINT] カメラ局所座標: カメラは +Y 軸上、太陽は XY 平面内（+X 側）。
    //         太陽の鉛直面に対して空は左右対称なので、方位は「太陽との方位差」だけで足りる。
    const float sunZenithCos = dot(normalize(gCameraPosKm), gSunDirection);
    const float3 sunDir = float3(sqrt(saturate(1.0 - sunZenithCos * sunZenithCos)), sunZenithCos, 0.0);
    const float viewZenithSin = sqrt(saturate(1.0 - viewZenithCos * viewZenithCos));
    const float3 viewDir = float3(viewZenithSin * lightViewCos, viewZenithCos,
                                  viewZenithSin * sqrt(saturate(1.0 - lightViewCos * lightViewCos)));

    IntegrationSettings settings;
    settings.stepCount = gStepCount;
    settings.sunTransmittanceMode = gSunTransmittanceMode;
    settings.nestedSteps = gNestedSteps;
    settings.multipleScattering = (gFlags & SKY_FLAG_MULTIPLE_SCATTERING) != 0;
    settings.isotropicPhase = false;
    settings.groundRadiance = false;
    settings.uniformSteps = false;
    settings.separateMiePhase = true;
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                float3(0.0, r, 0.0), viewDir, sunDir);
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    output[id.xy] = float4(result.luminance, 1.0);
    RWTexture2D<float4> mieOutput = ResourceDescriptorHeap[gSkyViewMieLutUav];
    mieOutput[id.xy] = float4(result.mieSingleNoPhase, 1.0);
}
