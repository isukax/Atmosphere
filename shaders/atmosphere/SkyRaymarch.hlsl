// 空・地形の空気遠近法をピクセルごとにレイマーチする（リファレンス描画）。
//
// LUT を使う実用パス（SkyViewLut / AerialPerspective → SkyFromLut）と同じ近似（多重散乱は Multi-scattering LUT）を、
// Sky-View LUT・空気遠近法のボリュームを経由せず多数のステップで直接積分する。両者の差が LUT 化による誤差（V3 / V3c）。
// 多重散乱を切ると単散乱だけになる（Phase 2 の V2 はこのモード）。
#include "atmosphere/SkyPassConstants.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float3 viewDir = OutputViewDirection(id.xy);

    IntegrationSettings settings;
    settings.stepCount = gStepCount;
    settings.sunTransmittanceMode = gSunTransmittanceMode;
    settings.nestedSteps = gNestedSteps;
    settings.multipleScattering = (gFlags & SKY_FLAG_MULTIPLE_SCATTERING) != 0;
    settings.isotropicPhase = false;
    settings.groundRadiance = false;
    settings.uniformSteps = false;
    settings.separateMiePhase = false;  // リファレンスは Mie の位相関数も光線ごとにそのまま掛ける
    settings.maxDistanceKm = kNoDistanceLimitKm;
    DisableOpaqueShadow(settings);  // 空のピクセルの光芒は扱わない（LUT 描画と揃える、design.md §5.5）

    if (gOutputMode == OUTPUT_CAMERA_FB && gDepthSrv != kInvalidDescriptorIndex) {
        Texture2D<float> depthTexture = ResourceDescriptorHeap[gDepthSrv];
        const float depth = depthTexture.Load(int3(id.xy, 0));
        if (depth > 0.0) {
            // 地形のピクセル: カメラ → 地形の点までを積分し、透過率と内散乱を地形の色に掛ける
            settings.maxDistanceKm = DepthToDistanceKm(id.xy, depth);
            SetOpaqueShadowFromConstants(settings);  // 空気遠近法のボリュームと同じ影（V3f の比較用）
            const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                        gCameraPosKm, viewDir, gSunDirection);
            // [POINT] 透過率は本来 RGB ごとに違う。V3c ではボリューム側（RGB 平均）との差を分けるため、
            //         平均にするモードも用意する。
            const float3 transmittance = gRaymarchMeanTransmittance != 0
                                             ? dot(result.transmittance, 1.0 / 3.0).xxx
                                             : result.transmittance;
            ApplyAerialPerspective(id.xy, result.luminance, transmittance);
            return;
        }
    }

    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                gCameraPosKm, viewDir, gSunDirection);
    float3 luminance = result.luminance;
    if ((gFlags & SKY_FLAG_SUN_DISK) != 0) {
        luminance += SunDiskPerLux(gCameraPosKm, viewDir);
    }
    WriteSkyOutput(id.xy, luminance, viewDir);
}
