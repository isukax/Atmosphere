

#ifndef ATMOSPHERE_INTEGRATE_HLSLI
#define ATMOSPHERE_INTEGRATE_HLSLI

#include "common/Bindless.hlsli"
#include "common/Shadow.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"
float RayleighPhase(float c) { return 3.0 / (16.0 * kPi) * (1.0 + c * c); }
#include "atmosphere/MultiScatteringSh.hlsli"


static const uint kMiePhaseTableCount = 2049;

float3 MiePhase(AtmosphereParameters a, float cosTheta)
{
    StructuredBuffer<float4> table = ResourceDescriptorHeap[a.miePhaseTableSrv];
    const float u = sqrt(acos(clamp(cosTheta, -1.0, 1.0)) / kPi);
    const float f = u * float(kMiePhaseTableCount - 1);
    const uint i = min(uint(f), kMiePhaseTableCount - 2);
    return lerp(table[i].rgb, table[i + 1].rgb, f - float(i));
}

struct AtmosphereLuts
{
    uint   transmittanceSrv;
    float2 transmittanceSize;
    uint   multiScatteringSrv;
    float2 multiScatteringSize;

    float3 mieChi[3];
};

struct OpaqueShadowParameters
{
    uint farSrv;
    uint nearSrv;
    float4x4 farToUvz;
    float4x4 nearToUvz;
};


// [POINT] Choose the near cascade only inside its complete volume. Descriptor selection varies by lane.
float SampleOpaqueShadow(OpaqueShadowParameters p, float3 cameraRelativeM)
{
    if (p.farSrv == kInvalidDescriptorIndex) return 1.0;
    uint srv = p.farSrv;
    float4x4 toUvz = p.farToUvz;
    if (p.nearSrv != kInvalidDescriptorIndex) {
        float3 uvz = mul(p.nearToUvz, float4(cameraRelativeM, 1.0)).xyz;
        if (all(uvz > 0.01) && all(uvz < 0.99)) { srv = p.nearSrv; toUvz = p.nearToUvz; }
    }
    return SampleShadowMap(srv, mul(toUvz, float4(cameraRelativeM, 1.0)).xyz);
}

struct IntegrationSettings
{
    uint stepCount;
    bool multipleScattering;
    bool groundRadiance;
    bool uniformSteps;
    bool separateMiePhase;
    float maxDistanceKm;


};



static const float kNoDistanceLimitKm = 1e9;

struct ScatteringResult
{
    float3 luminance;
    float3 mieSingleNoPhase;

    float3 multipleScattering;
    float3 transmittance;

    float3 rayleighAs1;
    float3 mieAs1;
    bool   hitGround;
};

float3 OneMinusExpOverX(float3 x)
{
    // [POINT] Avoid cancellation of 1-exp(-x) in optically thin integration segments.
    const float3 series = 1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0)));
    return select(x < 1e-2, series, (1.0 - exp(-x)) / max(x, 1e-30));
}

float3 SunTransmittance(AtmosphereParameters a, AtmosphereLuts luts, float3 p, float3 sunDir)
{
    const float r = length(p);
    const float muSun = dot(p, sunDir) / r;
    if (RayIntersectsGround(a, r, muSun)) return 0.0;
    Texture2D<float4> lut = ResourceDescriptorHeap[luts.transmittanceSrv];
    return SampleTransmittanceLut(a, lut, gLinearClamp, r, muSun, luts.transmittanceSize);
}



float3 MultipleScatteringSourceSh(AtmosphereParameters a, AtmosphereLuts luts, float3 p, float3 dir, float3 sunDir,
                                  MediumSample m)
{
    // [POINT] Six mirror-symmetric SH coefficients; convolve each band with the fixed Mie moments.
    const float r = length(p);
    const float3 up = p / r;
    const float muSun = dot(up, sunDir);
    const float2 uv = MultiScatteringRMuSunToUv(a, r, muSun, luts.multiScatteringSize);
    Texture2D<float4> lut = ResourceDescriptorHeap[luts.multiScatteringSrv];
    float3 c[kMsShCount];
    [unroll] for (uint b = 0; b < kMsShCount; ++b) {
        c[b] = lut.SampleLevel(gLinearClamp, MsShAtlasUv(uv, b), 0.0).rgb;
    }

    const float3 sunHorizontal = sunDir - muSun * up;
    const float len2 = dot(sunHorizontal, sunHorizontal);
    const float azimuthValid = len2 > 1e-12 ? 1.0 : 0.0;
    const float3 xAxis = sunHorizontal * (azimuthValid * rsqrt(max(len2, 1e-30)));
    const float z = dot(dir, up);
    const float x = dot(dir, xAxis);
    const MsShBasis basis = EvalMsShBasis(x, z, max(1.0 - x * x - z * z, 0.0));
    const float3 band0 = c[0] * basis.y[0];
    const float3 band1 = c[1] * basis.y[1] + c[2] * basis.y[2];
    const float3 band2 = c[3] * basis.y[3] + c[4] * basis.y[4] + azimuthValid * c[5] * basis.y[5];
    float3 source = m.rayleighScattering * (band0 + 0.1 * band2) +
                    m.mieScattering * (luts.mieChi[0] * band0 + luts.mieChi[1] * band1 + luts.mieChi[2] * band2);
    return source;
}

ScatteringResult IntegrateScatteredLuminance(AtmosphereParameters a, AtmosphereLuts luts, IntegrationSettings s,
                                             float3 origin, float3 dir, float3 sunDir)
{
    ScatteringResult result;
    result.luminance = 0.0;
    result.mieSingleNoPhase = 0.0;
    result.multipleScattering = 0.0;
    result.transmittance = 1.0;
    result.rayleighAs1 = 0.0;
    result.mieAs1 = 0.0;
    result.hitGround = false;

    float tTop0;
    float tTop1;
    if (!RaySphere(origin, dir, a.topRadiusKm, tTop0, tTop1) || tTop1 <= 0.0) {
        return result;
    }
    const float tStart = max(tTop0, 0.0);
    float tEnd = tTop1;
    bool hitGround = false;
    float tGround0;
    float tGround1;
    if (RaySphere(origin, dir, a.bottomRadiusKm, tGround0, tGround1) && tGround0 > 0.0 && tGround0 < tEnd) {
        tEnd = tGround0;
        hitGround = true;
    }

    if (s.maxDistanceKm < tEnd) {
        tEnd = max(s.maxDistanceKm, tStart);
        hitGround = false;
    }
    result.hitGround = hitGround;

    const float cosTheta = dot(dir, sunDir);
    const float phaseRayleigh = RayleighPhase(cosTheta);
    const float3 phaseMie = MiePhase(a, cosTheta);

    const float range = tEnd - tStart;
    const float invSteps = 1.0 / float(s.stepCount);
    float3 throughput = 1.0;
    for (uint i = 0; i < s.stepCount; ++i) {
        const float u0 = float(i) * invSteps;
        const float u1 = float(i + 1) * invSteps;
        const float t0 = tStart + range * (s.uniformSteps ? u0 : u0 * u0);
        const float t1 = tStart + range * (s.uniformSteps ? u1 : u1 * u1);
        const float dt = t1 - t0;
        const float t = 0.5 * (t0 + t1);
        const float3 p = origin + t * dir;
        const MediumSample m = SampleMedium(a, length(p) - a.bottomRadiusKm);
        const float3 segmentTransmittance = exp(-m.extinction * dt);

        float3 sunTransmittance = SunTransmittance(a, luts, p, sunDir);

        const float3 mieNoPhase = m.mieScattering * sunTransmittance;
        float3 source = m.rayleighScattering * phaseRayleigh * sunTransmittance;
        if (!s.separateMiePhase) {
            source += mieNoPhase * phaseMie;
        }
        float3 multipleSource = 0.0;
        if (s.multipleScattering) {
            multipleSource = MultipleScatteringSourceSh(a, luts, p, dir, sunDir, m);
            source += multipleSource;
        }

        const float3 segmentWeight = dt * OneMinusExpOverX(m.extinction * dt);
        result.luminance += throughput * source * segmentWeight;
        result.mieSingleNoPhase += throughput * mieNoPhase * segmentWeight;
        result.multipleScattering += throughput * multipleSource * segmentWeight;
        result.rayleighAs1 += throughput * m.rayleighScattering * segmentWeight;
        result.mieAs1 += throughput * m.mieScattering * segmentWeight;
        throughput *= segmentTransmittance;
    }

    if (hitGround && s.groundRadiance) {

        const float3 groundPos = origin + tEnd * dir;
        const float3 normal = normalize(groundPos);
        const float3 sunT = SunTransmittance(a, luts, groundPos, sunDir);
        result.luminance += throughput * sunT * saturate(dot(normal, sunDir)) * a.groundAlbedo / kPi;
    }
    result.transmittance = throughput;
    return result;
}

#endif
