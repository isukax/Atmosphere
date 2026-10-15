// 空をピクセルごとにレイマーチする（リファレンス描画）。
//
// LUT を使う実用パス（SkyViewLut → SkyFromLut）と同じ近似（多重散乱は Multi-scattering LUT）を、
// Sky-View LUT を経由せず多数のステップで直接積分する。両者の差が「Sky-View LUT による誤差」（V3）。
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
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                gCameraPosKm, viewDir, gSunDirection);
    float3 luminance = result.luminance;
    if ((gFlags & SKY_FLAG_SUN_DISK) != 0) {
        luminance += SunDiskPerLux(gCameraPosKm, viewDir);
    }
    WriteSkyOutput(id.xy, luminance, viewDir);
}
