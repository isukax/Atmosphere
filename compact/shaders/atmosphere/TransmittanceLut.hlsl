

#include "common/Bindless.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"

cbuffer TransmittanceConstants : register(b0) {
    AtmosphereParameters gAtmosphere : packoffset(c0);
    uint gOutputUav : packoffset(c6);
    uint gWidth : packoffset(c6.y);
    uint gHeight : packoffset(c6.z);
    uint gStepCount : packoffset(c6.w);
};


[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float2 lutSize = float2(gWidth, gHeight);
    const float2 uv = (float2(id.xy) + 0.5) / lutSize;
    float r;
    float mu;
    TransmittanceUvToRMu(gAtmosphere, uv, lutSize, r, mu);

    RWTexture2D<float4> output = ResourceDescriptorHeap[gOutputUav];
    output[id.xy] = float4(ComputeTransmittanceToTop(gAtmosphere, r, mu, gStepCount), 1.0);


}
