#ifndef SKY_PASS_CONSTANTS_HLSLI
#define SKY_PASS_CONSTANTS_HLSLI
#include "atmosphere/AtmosphereIntegrate.hlsli"
cbuffer SkyPassConstants : register(b0) {
    AtmosphereParameters gAtmosphere : packoffset(c0);
    row_major float4x4 gInvViewProjRot : packoffset(c6);
    float3 gCameraPosKm : packoffset(c10);
    uint gOutputUav : packoffset(c10.w);
    float3 gSunDirection : packoffset(c11);
    uint gTransmittanceLutSrv : packoffset(c11.w);
    float3 gSunIlluminanceLux : packoffset(c12);
    float gExposureScale : packoffset(c12.w);
    uint gWidth : packoffset(c13);
    uint gHeight : packoffset(c13.y);
    uint gStepCount : packoffset(c13.z);
    uint gMultiScatteringLutSrv : packoffset(c13.w);
    float2 gTransmittanceLutSize : packoffset(c14);
    float2 gMultiScatteringLutSize : packoffset(c14.z);
    uint gSkyViewLutSrv : packoffset(c15);
    uint gSkyViewMieLutUav : packoffset(c15.y);
    uint gSkyViewMieLutSrv : packoffset(c15.z);
    uint gSkyIrradianceUav : packoffset(c15.w);
    float2 gSkyViewLutSize : packoffset(c16);
    uint gAerialPerspectiveUav : packoffset(c16.z);
    uint gAerialPerspectiveMieUav : packoffset(c16.w);
    uint gAerialPerspectiveSrv : packoffset(c17);
    uint gAerialPerspectiveMieSrv : packoffset(c17.y);
    uint gAerialPerspectiveMsUav : packoffset(c17.z);
    uint gAerialPerspectiveMsSrv : packoffset(c17.w);
    uint gDepthSrv : packoffset(c18);
    float gApKmPerSlice : packoffset(c18.y);
    float gApSliceCount : packoffset(c18.z);
    uint gApMaxSteps : packoffset(c18.w);
    row_major float4x4 gShadowFarToUvz : packoffset(c19);
    row_major float4x4 gShadowNearToUvz : packoffset(c23);
    uint gShadowFarSrv : packoffset(c27);
    uint gShadowNearSrv : packoffset(c27.y);
    uint gApPerPixelSteps : packoffset(c27.z);
    uint gApPerPixelBisectSteps : packoffset(c27.w);
    float4 gMsPhaseMoment0 : packoffset(c28);
    float4 gMsPhaseMoment1 : packoffset(c29);
    float4 gMsPhaseMoment2 : packoffset(c30);
};
OpaqueShadowParameters LoadOpaqueShadow()
{
    OpaqueShadowParameters p;
    p.farSrv = gShadowFarSrv;
    p.nearSrv = gShadowNearSrv;
    p.farToUvz = gShadowFarToUvz;
    p.nearToUvz = gShadowNearToUvz;
    return p;
}



AtmosphereLuts LoadAtmosphereLuts()
{
    AtmosphereLuts luts;
    luts.transmittanceSrv = gTransmittanceLutSrv;
    luts.transmittanceSize = gTransmittanceLutSize;
    luts.multiScatteringSrv = gMultiScatteringLutSrv;
    luts.multiScatteringSize = gMultiScatteringLutSize;
    [unroll] for (uint l = 0; l < 3; ++l) {
        const float3 moments[3] = {gMsPhaseMoment0.rgb, gMsPhaseMoment1.rgb, gMsPhaseMoment2.rgb};
        luts.mieChi[l] = moments[l];
    }
    return luts;
}

float3 CameraViewDirection(uint2 pixel)
{
    const float2 ndc = float2((float(pixel.x) + 0.5) / float(gWidth) * 2.0 - 1.0,
                              1.0 - (float(pixel.y) + 0.5) / float(gHeight) * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, 1.0, 1.0));
    return normalize(h.xyz / h.w);
}



static const float kMaxSceneColorFb = 65000.0;

void SampleSkyViewLuts(float3 cameraPosKm, float3 viewDir, out float3 rayleighAndMs, out float3 mieNoPhase)
{
    const float r = length(cameraPosKm);
    const float3 up = cameraPosKm / r;
    const float viewZenithCos = dot(viewDir, up);

    float lightViewCos = 1.0;
    const float3 side = cross(up, viewDir);
    if (dot(side, side) > 1e-12) {
        const float3 sideN = normalize(side);
        const float3 forward = normalize(cross(sideN, up));
        const float2 lightOnPlane = float2(dot(gSunDirection, forward), dot(gSunDirection, sideN));
        const float len = length(lightOnPlane);
        lightViewCos = len > 1e-6 ? lightOnPlane.x / len : 1.0;
    }

    float t0;
    float t1;
    const bool intersectGround = RaySphere(cameraPosKm, viewDir, gAtmosphere.bottomRadiusKm, t0, t1) && t0 > 0.0;
    const float lutR = max(r, gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);
    const float2 uv = SkyViewViewParamsToUv(gAtmosphere, intersectGround, viewZenithCos, lightViewCos, lutR,
                                            gSkyViewLutSize);
    Texture2D<float4> skyView = ResourceDescriptorHeap[gSkyViewLutSrv];
    Texture2D<float4> skyViewMie = ResourceDescriptorHeap[gSkyViewMieLutSrv];
    rayleighAndMs = skyView.SampleLevel(gLinearClamp, uv, 0.0).rgb;
    mieNoPhase = skyViewMie.SampleLevel(gLinearClamp, uv, 0.0).rgb;
}

float DepthToDistanceKm(uint2 pixel, float depth)
{
    const float2 ndc = float2((float(pixel.x) + 0.5) / float(gWidth) * 2.0 - 1.0,
                              1.0 - (float(pixel.y) + 0.5) / float(gHeight) * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, depth, 1.0));
    return length(h.xyz / h.w) * 0.001;
}

void ApplyAerialPerspective(uint2 pixel, float3 inscatterPerLux, float transmittance)
{
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    float3 fb = output[pixel].rgb * transmittance + inscatterPerLux * gSunIlluminanceLux * gExposureScale;
    output[pixel] = float4(min(fb, kMaxSceneColorFb), 1.0);
}

void WriteSkyOutput(uint2 pixel, float3 luminancePerLux)
{
    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    output[pixel] = float4(min(luminancePerLux * gSunIlluminanceLux * gExposureScale, kMaxSceneColorFb), 1.0);
}

float3 SunDiskPerLux(float3 cameraPosKm, float3 viewDir)
{
    const float cosAlpha = cos(gAtmosphere.sunAngularRadiusRad);
    if (dot(viewDir, gSunDirection) < cosAlpha) {
        return 0.0;
    }
    const float r = length(cameraPosKm);
    const float mu = dot(cameraPosKm, viewDir) / r;
    if (RayIntersectsGround(gAtmosphere, r, mu)) {
        return 0.0;
    }
    Texture2D<float4> lut = ResourceDescriptorHeap[gTransmittanceLutSrv];
    const float3 transmittance = SampleTransmittanceLut(gAtmosphere, lut, gLinearClamp, r, mu, gTransmittanceLutSize);
    const float solidAngleSr = 2.0 * kPi * (1.0 - cosAlpha);
    return transmittance / solidAngleSr;
}

#endif
