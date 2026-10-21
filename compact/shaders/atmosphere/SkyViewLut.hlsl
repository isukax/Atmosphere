

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

    const float sunZenithCos = dot(normalize(gCameraPosKm), gSunDirection);
    const float3 sunDir = float3(sqrt(saturate(1.0 - sunZenithCos * sunZenithCos)), sunZenithCos, 0.0);
    const float viewZenithSin = sqrt(saturate(1.0 - viewZenithCos * viewZenithCos));
    const float3 viewDir = float3(viewZenithSin * lightViewCos, viewZenithCos,
                                  viewZenithSin * sqrt(saturate(1.0 - lightViewCos * lightViewCos)));

    IntegrationSettings settings;
    settings.stepCount = gStepCount;
    settings.multipleScattering = true;
    settings.groundRadiance = false;
    settings.uniformSteps = false;
    settings.separateMiePhase = true;
    settings.maxDistanceKm = kNoDistanceLimitKm;
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                float3(0.0, r, 0.0), viewDir, sunDir);
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    output[id.xy] = float4(result.luminance, 1.0);
    RWTexture2D<float4> mieOutput = ResourceDescriptorHeap[gSkyViewMieLutUav];
    mieOutput[id.xy] = float4(result.mieSingleNoPhase, 1.0);
}
