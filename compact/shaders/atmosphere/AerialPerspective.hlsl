

#include "atmosphere/SkyPassConstants.hlsli"

float3 VoxelViewDirection(float2 voxel)
{
    const float2 uv = (voxel + 0.5) / gApSliceCount;
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, 1.0, 1.0));
    return normalize(h.xyz / h.w);
}

[numthreads(4, 4, 4)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const float sliceCount = gApSliceCount;
    const float3 voxelDir = VoxelViewDirection(float2(id.xy));
    float3 viewDir = voxelDir;

    // [POINT] Quadratic distance placement, with a fixed nominal 128 km extent at every resolution.
    float slice = (float(id.z) + 0.5) / sliceCount;
    slice = slice * slice * sliceCount;
    float distanceKm = slice * gApKmPerSlice;

    const float3 pointKm = gCameraPosKm + distanceKm * viewDir;
    if (length(pointKm) <= gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm) {
        const float3 moved = normalize(pointKm) * (gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm + 0.001);
        viewDir = normalize(moved - gCameraPosKm);
        distanceKm = length(moved - gCameraPosKm);
    }

    IntegrationSettings settings;

    settings.stepCount = max(1u, ((id.z + 1) * gApMaxSteps) / uint(gApSliceCount));
    settings.multipleScattering = true;
    settings.groundRadiance = false;
    settings.uniformSteps = true;
    settings.separateMiePhase = true;
    settings.maxDistanceKm = distanceKm;

    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                gCameraPosKm, viewDir, gSunDirection);

    RWTexture3D<float4> output = ResourceDescriptorHeap[gAerialPerspectiveUav];
    RWTexture3D<float4> mieOutput = ResourceDescriptorHeap[gAerialPerspectiveMieUav];
    RWTexture3D<float4> msOutput = ResourceDescriptorHeap[gAerialPerspectiveMsUav];

    output[id] = float4(result.luminance - result.multipleScattering, dot(result.transmittance, 1.0 / 3.0));
    mieOutput[id] = float4(result.mieSingleNoPhase, 1.0);
    msOutput[id] = float4(result.multipleScattering, 1.0);
}
