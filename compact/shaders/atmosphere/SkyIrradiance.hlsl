

#include "atmosphere/SkyPassConstants.hlsli"
#include "common/SphericalHarmonics.hlsli"

static const uint kThreads = 256;
static const uint kGridWidth = 128;
static const uint kGridHeight = 64;
static const uint kMieThetaCount = 128;
static const uint kMiePhiCount = 16;

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

        output[thread.x] = float4(gsSum[thread.x][0] * kShCosineLobe[thread.x], 0.0);
    }
}
