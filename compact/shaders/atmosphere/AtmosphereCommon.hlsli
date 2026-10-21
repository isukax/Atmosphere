

#ifndef ATMOSPHERE_COMMON_HLSLI
#define ATMOSPHERE_COMMON_HLSLI

#include "common/Math.hlsli"

struct AtmosphereParameters
{
    float bottomRadiusKm;
    float topRadiusKm;
    float rayleighScaleHeightKm;
    float mieScaleHeightKm;
    float3 rayleighScatteringPerKm;
    float ozoneCenterKm;
    float3 mieScatteringPerKm;
    float ozoneHalfWidthKm;
    float3 mieExtinctionPerKm;
    float sunAngularRadiusRad;
    float3 ozoneAbsorptionPerKm;
    uint miePhaseTableSrv;
    float3 groundAlbedo;
    float pad;
};

struct MediumSample
{
    float3 rayleighScattering;
    float3 mieScattering;
    float3 extinction;
};



MediumSample SampleMedium(AtmosphereParameters a, float altitudeKm)
{

    altitudeKm = max(altitudeKm, 0.0);

    const float mieDensity = exp(-altitudeKm / a.mieScaleHeightKm);
    const float rayleighDensity = exp(-altitudeKm / a.rayleighScaleHeightKm);
    const float ozoneDensity = max(0.0, 1.0 - abs(altitudeKm - a.ozoneCenterKm) / a.ozoneHalfWidthKm);

    MediumSample m;
    m.rayleighScattering = a.rayleighScatteringPerKm * rayleighDensity;
    m.mieScattering = a.mieScatteringPerKm * mieDensity;
    m.extinction = m.rayleighScattering + a.mieExtinctionPerKm * mieDensity + a.ozoneAbsorptionPerKm * ozoneDensity;
    return m;
}

bool RaySphere(float3 origin, float3 dir, float radius, out float t0, out float t1)
{
    const float r = length(origin);
    const float b = dot(origin, dir);

    const float c = (r - radius) * (r + radius);
    const float discriminant = b * b - c;
    t0 = 0.0;
    t1 = 0.0;
    if (discriminant < 0.0) {
        return false;
    }

    const float s = sqrt(discriminant);
    const float q = -(b + (b >= 0.0 ? s : -s));
    const float ta = q;
    const float tb = q != 0.0 ? c / q : -b;
    t0 = min(ta, tb);
    t1 = max(ta, tb);
    return true;
}

float DistanceToTopAtmosphere(AtmosphereParameters a, float r, float mu)
{
    const float discriminant = r * r * (mu * mu - 1.0) + a.topRadiusKm * a.topRadiusKm;
    return max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
}

bool RayIntersectsGround(AtmosphereParameters a, float r, float mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + a.bottomRadiusKm * a.bottomRadiusKm >= 0.0;
}

float3 ComputeTransmittanceToTop(AtmosphereParameters a, float r, float mu, uint stepCount)
{
    const float distanceKm = DistanceToTopAtmosphere(a, r, mu);
    const float dt = distanceKm / float(stepCount);
    float3 opticalDepth = 0.0;
    for (uint i = 0; i <= stepCount; ++i) {
        const float t = float(i) * dt;

        const float ri = sqrt(t * t + 2.0 * r * mu * t + r * r);
        const float weight = (i == 0 || i == stepCount) ? 0.5 : 1.0;
        opticalDepth += SampleMedium(a, ri - a.bottomRadiusKm).extinction * (weight * dt);
    }
    return exp(-opticalDepth);
}

float FromUnitToSubUv(float u, float resolution)
{
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

float FromSubUvToUnit(float u, float resolution)
{
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

float2 TransmittanceRMuToUv(AtmosphereParameters a, float r, float mu, float2 lutSize)
{
    const float H = sqrt(a.topRadiusKm * a.topRadiusKm - a.bottomRadiusKm * a.bottomRadiusKm);
    const float rho = sqrt(max(r * r - a.bottomRadiusKm * a.bottomRadiusKm, 0.0));
    const float d = DistanceToTopAtmosphere(a, r, mu);
    const float dMin = a.topRadiusKm - r;
    const float dMax = rho + H;
    const float xMu = (d - dMin) / (dMax - dMin);
    const float xR = rho / H;
    return float2(FromUnitToSubUv(xMu, lutSize.x), FromUnitToSubUv(xR, lutSize.y));
}

void TransmittanceUvToRMu(AtmosphereParameters a, float2 uv, float2 lutSize, out float r, out float mu)
{
    const float xMu = FromSubUvToUnit(uv.x, lutSize.x);
    const float xR = FromSubUvToUnit(uv.y, lutSize.y);
    const float H = sqrt(a.topRadiusKm * a.topRadiusKm - a.bottomRadiusKm * a.bottomRadiusKm);
    const float rho = H * xR;
    r = sqrt(rho * rho + a.bottomRadiusKm * a.bottomRadiusKm);
    const float dMin = a.topRadiusKm - r;
    const float dMax = rho + H;
    const float d = dMin + xMu * (dMax - dMin);
    mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

float3 SampleTransmittanceLut(AtmosphereParameters a, Texture2D<float4> lut, SamplerState s, float r, float mu,
                              float2 lutSize)
{
    return lut.SampleLevel(s, TransmittanceRMuToUv(a, r, mu, lutSize), 0.0).rgb;
}

static const float kSurfaceOffsetKm = 0.001;

float2 MultiScatteringRMuSunToUv(AtmosphereParameters a, float r, float muSun, float2 lutSize)
{
    const float2 unit = float2(muSun * 0.5 + 0.5,
                               saturate((r - a.bottomRadiusKm) / (a.topRadiusKm - a.bottomRadiusKm)));
    return float2(FromUnitToSubUv(unit.x, lutSize.x), FromUnitToSubUv(unit.y, lutSize.y));
}

void MultiScatteringUvToRMuSun(AtmosphereParameters a, float2 uv, float2 lutSize, out float r, out float muSun)
{
    const float2 unit = float2(FromSubUvToUnit(uv.x, lutSize.x), FromSubUvToUnit(uv.y, lutSize.y));
    muSun = clamp(unit.x * 2.0 - 1.0, -1.0, 1.0);
    r = clamp(a.bottomRadiusKm + saturate(unit.y) * (a.topRadiusKm - a.bottomRadiusKm),
              a.bottomRadiusKm + kSurfaceOffsetKm, a.topRadiusKm - kSurfaceOffsetKm);
}



void SkyViewUvToViewParams(AtmosphereParameters a, float2 uv, float2 lutSize, float r,
                           out float viewZenithCos, out float lightViewCos)
{
    uv = float2(FromSubUvToUnit(uv.x, lutSize.x), FromSubUvToUnit(uv.y, lutSize.y));
    const float horizonDistance = sqrt(max(r * r - a.bottomRadiusKm * a.bottomRadiusKm, 0.0));
    const float beta = acos(horizonDistance / r);
    const float zenithHorizonAngle = kPi - beta;
    float viewZenithAngle;
    if (uv.y < 0.5) {
        float coord = 1.0 - 2.0 * uv.y;
        coord = 1.0 - coord * coord;
        viewZenithAngle = zenithHorizonAngle * coord;
    } else {
        float coord = uv.y * 2.0 - 1.0;
        viewZenithAngle = zenithHorizonAngle + beta * coord * coord;
    }
    viewZenithCos = cos(viewZenithAngle);
    lightViewCos = -(uv.x * uv.x * 2.0 - 1.0);
}

float2 SkyViewViewParamsToUv(AtmosphereParameters a, bool intersectGround, float viewZenithCos, float lightViewCos,
                             float r, float2 lutSize)
{
    const float horizonDistance = sqrt(max(r * r - a.bottomRadiusKm * a.bottomRadiusKm, 0.0));
    const float beta = acos(horizonDistance / r);
    const float zenithHorizonAngle = kPi - beta;
    const float viewZenithAngle = acos(clamp(viewZenithCos, -1.0, 1.0));
    float2 uv;
    if (!intersectGround) {
        const float coord = 1.0 - sqrt(saturate(1.0 - viewZenithAngle / zenithHorizonAngle));
        uv.y = coord * 0.5;
    } else {
        const float coord = sqrt(saturate((viewZenithAngle - zenithHorizonAngle) / beta));
        uv.y = coord * 0.5 + 0.5;
    }
    uv.x = sqrt(saturate(-lightViewCos * 0.5 + 0.5));
    return float2(FromUnitToSubUv(uv.x, lutSize.x), FromUnitToSubUv(uv.y, lutSize.y));
}

float3 EquirectDirection(uint2 pixel, uint2 size)
{
    const float phi = 2.0 * kPi * (float(pixel.x) + 0.5) / float(size.x);
    const float elevation = 0.5 * kPi - kPi * (float(pixel.y) + 0.5) / float(size.y);
    return float3(cos(elevation) * cos(phi), sin(elevation), cos(elevation) * sin(phi));
}

#endif
