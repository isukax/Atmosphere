

#include "atmosphere/SkyPassConstants.hlsli"

struct AerialPerspectiveSample
{
    float3 rayleigh;
    float3 mieNoPhase;
    float3 multiple;
    float transmittance;
};

float3 AerialPerspectiveUvw(float2 uv, float distanceKm, out float fade)
{

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

    s.transmittance = 1.0 - fade * (1.0 - rayleigh.a);
    return s;
}

bool IsSunVisible(OpaqueShadowParameters shadow, float3 viewDir, float distanceKm)
{
    return SampleOpaqueShadow(shadow, viewDir * (distanceKm * 1000.0)) >= 0.5;
}

void ShadowedSingleScattering(float2 uv, float3 viewDir, float distanceKm, AerialPerspectiveSample end,
                             out float3 rayleigh, out float3 mieNoPhase)
{
    // [POINT] Integrals are cumulative. Subtract the single-scattering contribution in each shadowed segment.
    // Multiple scattering is deliberately unaffected by opaque shadows.
    const OpaqueShadowParameters shadow = LoadOpaqueShadow();
    const uint steps = max(gApPerPixelSteps, 1u);
    float3 sumRayleigh = 0.0;
    float3 sumMie = 0.0;
    float3 startRayleigh = 0.0;
    float3 startMie = 0.0;
    bool segmentLit = true;
    float previousT = 0.0;
    for (uint k = 0; k < steps; ++k) {

        const float u = (float(k) + 0.5) / float(steps);
        const float t = distanceKm * u * u;
        const bool lit = IsSunVisible(shadow, viewDir, t);
        if (k == 0) {
            segmentLit = lit;
        } else if (lit != segmentLit) {

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

    rayleigh = max(sumRayleigh, 0.0);
    mieNoPhase = max(sumMie, 0.0);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float3 viewDir = CameraViewDirection(id.xy);

    const float cosTheta = dot(viewDir, gSunDirection);
    const float3 phaseMie = MiePhase(gAtmosphere, cosTheta);

    if (gDepthSrv != kInvalidDescriptorIndex) {
        Texture2D<float> depthTexture = ResourceDescriptorHeap[gDepthSrv];
        const float depth = depthTexture.Load(int3(id.xy, 0));
        if (depth > 0.0) {

            const float distanceKm = DepthToDistanceKm(id.xy, depth);
            const float2 uv = (float2(id.xy) + 0.5) / float2(gWidth, gHeight);
            const AerialPerspectiveSample end = SampleAerialPerspective(uv, distanceKm);
            float3 rayleigh = end.rayleigh;
            float3 mieNoPhase = end.mieNoPhase;
            if (gShadowFarSrv != kInvalidDescriptorIndex) {
                ShadowedSingleScattering(uv, viewDir, distanceKm, end, rayleigh, mieNoPhase);
            }
            const float3 inscatter = rayleigh + end.multiple + phaseMie * mieNoPhase;

            ApplyAerialPerspective(id.xy, inscatter, end.transmittance);
            return;
        }
    }

    float3 rayleighAndMs;
    float3 mieNoPhase;
    SampleSkyViewLuts(gCameraPosKm, viewDir, rayleighAndMs, mieNoPhase);
    float3 luminance = rayleighAndMs + phaseMie * mieNoPhase;
     {
        luminance += SunDiskPerLux(gCameraPosKm, viewDir);
    }
    WriteSkyOutput(id.xy, luminance);
}
