

#include "atmosphere/AtmosphereIntegrate.hlsli"

cbuffer MultiScatteringConstants : register(b0) {
    AtmosphereParameters gAtmosphere : packoffset(c0);
    uint gOutputUav : packoffset(c6);
    uint gWidth : packoffset(c6.y);
    uint gHeight : packoffset(c6.z);
    uint gStepCount : packoffset(c6.w);
    float2 gTransmittanceLutSize : packoffset(c7);
    uint gTransmittanceLutSrv : packoffset(c7.z);
    uint gTransferUav : packoffset(c7.w);
    uint gTransferSrv : packoffset(c8);
    uint gPad0 : packoffset(c8.y);
    uint gPad1 : packoffset(c8.z);
    uint gPad2 : packoffset(c8.w);
    float4 gPhaseMoment0 : packoffset(c9);
    float4 gPhaseMoment1 : packoffset(c10);
    float4 gPhaseMoment2 : packoffset(c11);
};


static const uint kShRings = 32;
static const uint kShAzimuths = 32;

static const uint kShAzimuthsIntegrated = kShAzimuths / 2;
static const float kShAzimuthWeight = 2.0 * 2.0 * kPi / float(kShAzimuths);

static const uint kShThreads = 128;
static const uint kShDirectionsPerThread = kShRings * kShAzimuthsIntegrated / kShThreads;

static const uint kShTransferF0 = 0;
static const uint kShTransferF1 = 9;
static const uint kShTransferF2 = 13;
static const uint kShTransferG = 14;
static const uint kShTransferCount = 17;

static const uint kShZenithGround = 64;
static const uint kShZenithSky = 64;
static const uint kShTransferThreads = kShZenithGround + kShZenithSky;

static const uint kShBlock0Band[3] = {0, 1, 2};
static const uint kShBlock1Band[2] = {1, 2};

static const float kShHemisphereCosine[3] = {0.886226925, 1.023326708, 0.495415912};
static const float kShDeterminantFloor = 1e-4;

AtmosphereLuts MakeShLuts()
{
    AtmosphereLuts luts;
    luts.transmittanceSrv = gTransmittanceLutSrv;
    luts.transmittanceSize = gTransmittanceLutSize;
    luts.multiScatteringSrv = kInvalidDescriptorIndex;
    luts.multiScatteringSize = 0.0;
    [unroll] for (uint l = 0; l < 3; ++l) {
        const float3 moments[3] = {gPhaseMoment0.rgb, gPhaseMoment1.rgb, gPhaseMoment2.rgb};
        luts.mieChi[l] = moments[l];
    }
    return luts;
}

IntegrationSettings MakeShSettings()
{
    IntegrationSettings settings;
    settings.stepCount = gStepCount;
    settings.multipleScattering = false;
    settings.groundRadiance = true;
    settings.uniformSteps = false;
    settings.separateMiePhase = false;
    settings.maxDistanceKm = kNoDistanceLimitKm;
    return settings;
}

groupshared float3 gsShTransfer[kShTransferCount][kShTransferThreads];

[numthreads(kShTransferThreads, 1, 1)]
void CSTransfer(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint row = group.x;
    const uint2 texel = group.xy;
    const uint t = thread.x;
    const float2 lutSize = float2(gWidth, gHeight);
    float r;
    float muSun;
    MultiScatteringUvToRMuSun(gAtmosphere, float2(0.5, float(row) + 0.5) / lutSize, lutSize, r, muSun);
    const float bottom = gAtmosphere.bottomRadiusKm;

    const float zh = -sqrt(saturate((r - bottom) * (r + bottom) / (r * r)));
    const bool groundSide = t < kShZenithGround;
    const float s = groundSide ? (float(t) + 0.5) / float(kShZenithGround)
                               : (float(t - kShZenithGround) + 0.5) / float(kShZenithSky);
    const float span = groundSide ? 1.0 + zh : 1.0 - zh;
    const float z = groundSide ? zh - span * s * s : zh + span * s * s;
    const float dz = span * 2.0 * s / float(groundSide ? kShZenithGround : kShZenithSky);
    const float s2 = saturate(1.0 - z * z);

    const AtmosphereLuts luts = MakeShLuts();
    IntegrationSettings settings = MakeShSettings();

    const float3 origin = float3(0.0, r, 0.0);
    const float3 dir = float3(sqrt(s2), z, 0.0);
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, luts, settings, origin, dir,
                                                                float3(0.0, 1.0, 0.0));

    float3 bandAs1[3];
    bandAs1[0] = result.rayleighAs1 + luts.mieChi[0] * result.mieAs1;
    bandAs1[1] = luts.mieChi[1] * result.mieAs1;
    bandAs1[2] = 0.1 * result.rayleighAs1 + luts.mieChi[2] * result.mieAs1;

    const float f0[3] = {kMsShY00, kMsShY1 * z, kMsShY20 * (3.0 * z * z - 1.0)};
    const float f1[2] = {kMsShY1, kMsShY21 * z};
    const float w0 = 2.0 * kPi * dz;
    const float w1 = kPi * s2 * dz;
    [unroll] for (uint i0 = 0; i0 < 3; ++i0) {
        [unroll] for (uint j0 = 0; j0 < 3; ++j0) {
            gsShTransfer[kShTransferF0 + i0 * 3 + j0][t] = w0 * f0[i0] * f0[j0] * bandAs1[kShBlock0Band[j0]];
        }
        gsShTransfer[kShTransferG + i0][t] = result.hitGround ? w0 * f0[i0] * result.transmittance : 0.0;
    }
    [unroll] for (uint i1 = 0; i1 < 2; ++i1) {
        [unroll] for (uint j1 = 0; j1 < 2; ++j1) {
            gsShTransfer[kShTransferF1 + i1 * 2 + j1][t] = w1 * f1[i1] * f1[j1] * bandAs1[kShBlock1Band[j1]];
        }
    }
    gsShTransfer[kShTransferF2][t] = kPi * kMsShY22 * kMsShY22 * s2 * s2 * dz * bandAs1[2];
    GroupMemoryBarrierWithGroupSync();
    if (t < kShTransferCount) {
        float3 sum = 0.0;
        for (uint i = 0; i < kShTransferThreads; ++i) {
            sum += gsShTransfer[t][i];
        }
        RWTexture2D<float4> transfer = ResourceDescriptorHeap[gTransferUav];
        transfer[uint2(t, row)] = float4(sum, 0.0);
    }
}

groupshared float3 gsShWave[kMsShCount][kShThreads / 4];

[numthreads(kShThreads, 1, 1)]
void CSMain(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    const uint2 texel = group.xy;
    const uint t = thread.x;
    const float2 lutSize = float2(gWidth, gHeight);

    float r;
    float muSun;
    MultiScatteringUvToRMuSun(gAtmosphere, (float2(texel) + 0.5) / lutSize, lutSize, r, muSun);

    const float3 origin = float3(0.0, r, 0.0);
    const float sinSun = sqrt(saturate(1.0 - muSun * muSun));
    const float3 sunDir = float3(sinSun, muSun, 0.0);

    const float3 gridX = float3(-muSun, sinSun, 0.0);
    const float3 gridY = float3(0.0, 0.0, 1.0);

    const AtmosphereLuts luts = MakeShLuts();
    const IntegrationSettings settings = MakeShSettings();

    float3 acc[kMsShCount];
    [unroll] for (uint k = 0; k < kMsShCount; ++k) {
        acc[k] = 0.0;
    }
    for (uint j = 0; j < kShDirectionsPerThread; ++j) {
        const uint d = t + j * kShThreads;
        const uint ring = d / kShAzimuthsIntegrated;
        const uint azimuth = d % kShAzimuthsIntegrated;
        const float u0 = float(ring) / float(kShRings);
        const float u1 = float(ring + 1) / float(kShRings);
        const float uc = (float(ring) + 0.5) / float(kShRings);
        const float theta = kPi * uc * uc;
        const float phi = 2.0 * kPi * (float(azimuth) + 0.5) / float(kShAzimuths);
        const float weight = (cos(kPi * u0 * u0) - cos(kPi * u1 * u1)) * kShAzimuthWeight;
        const float3 local = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        const float3 dir = local.x * gridX + local.y * gridY + local.z * sunDir;

        const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, luts, settings, origin, dir, sunDir);

        const MsShBasis basis = EvalMsShBasis(dir.x, dir.y, dir.z * dir.z);
        [unroll] for (uint b = 0; b < kMsShCount; ++b) {
            acc[b] += weight * basis.y[b] * result.luminance;
        }
    }

    float3 c1[kMsShCount];
    const uint lanes = WaveGetLaneCount();
    [unroll] for (uint k2 = 0; k2 < kMsShCount; ++k2) {
        const float3 waveSum = WaveActiveSum(acc[k2]);
        if (WaveIsFirstLane()) {
            gsShWave[k2][t / lanes] = waveSum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t != 0) {
        return;
    }
    [unroll] for (uint b = 0; b < kMsShCount; ++b) {
        c1[b] = 0.0;
        for (uint w = 0; w < kShThreads / lanes; ++w) {
            c1[b] += gsShWave[b][w];
        }
    }
    Texture2D<float4> transferTable = ResourceDescriptorHeap[gTransferSrv];
    float3 transfer[kShTransferCount];
    [unroll] for (uint k3 = 0; k3 < kShTransferCount; ++k3) {
        transfer[k3] = transferTable.Load(int3(k3, texel.y, 0)).rgb;
    }

    float3 a[3][3];
    [unroll] for (uint i = 0; i < 3; ++i) {
        [unroll] for (uint j = 0; j < 3; ++j) {
            const float3 ground = gAtmosphere.groundAlbedo / kPi * transfer[kShTransferG + i] * kShHemisphereCosine[j];
            a[i][j] = (i == j ? 1.0 : 0.0) - transfer[kShTransferF0 + i * 3 + j] - ground;
        }
    }
    const float3 rhs0[3] = {c1[0], c1[1], c1[3]};
    const float3 cof00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    const float3 cof01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
    const float3 cof02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    const float3 det0 = a[0][0] * cof00 + a[0][1] * cof01 + a[0][2] * cof02;

    const float3 x0 = (cof00 * rhs0[0] + (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * rhs0[1] +
                       (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * rhs0[2]) / det0;
    const float3 x1 = (cof01 * rhs0[0] + (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * rhs0[1] +
                       (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * rhs0[2]) / det0;
    const float3 x3 = (cof02 * rhs0[0] + (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * rhs0[1] +
                       (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * rhs0[2]) / det0;

    const float3 b00 = 1.0 - transfer[kShTransferF1 + 0];
    const float3 b01 = -transfer[kShTransferF1 + 1];
    const float3 b10 = -transfer[kShTransferF1 + 2];
    const float3 b11 = 1.0 - transfer[kShTransferF1 + 3];
    const float3 det1 = b00 * b11 - b01 * b10;
    const float3 x2 = (b11 * c1[2] - b01 * c1[4]) / det1;
    const float3 x4 = (b00 * c1[4] - b10 * c1[2]) / det1;
    const float3 x5 = c1[5] / max(1.0 - transfer[kShTransferF2], 1e-3);

    const bool3 singular0 = abs(det0) < kShDeterminantFloor;
    const bool3 singular1 = abs(det1) < kShDeterminantFloor;
    float3 c[kMsShCount];
    c[0] = select(singular0, c1[0] / max(a[0][0], 1e-3), x0);
    c[1] = select(singular0, c1[1] / max(a[1][1], 1e-3), x1);
    c[3] = select(singular0, c1[3] / max(a[2][2], 1e-3), x3);
    c[2] = select(singular1, c1[2] / max(b00, 1e-3), x2);
    c[4] = select(singular1, c1[4] / max(b11, 1e-3), x4);
    c[5] = x5;

    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    [unroll] for (uint b2 = 0; b2 < kMsShCount; ++b2) {
        output[uint2(texel.x + b2 * gWidth, texel.y)] = float4(c[b2], 1.0);
    }

}
