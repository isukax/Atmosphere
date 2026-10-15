// Transmittance LUT の生成（Hillaire 2020 §5.2 / Bruneton 2017）。
//
// 各テクセル = ある高度 r から、ある方向 mu に大気上端まで進む光線。値はその透過率 T（無次元、RGB）。
// 太陽方向の透過率（地表・空気中の点に届く太陽光の減衰）を、ピクセルごとの入れ子レイマーチの
// 代わりにこの LUT 1 回の参照で求める。大気パラメータが変わったときだけ作り直せばよい。
#include "common/Bindless.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"

cbuffer TransmittanceLutConstants : register(b0)
{
    AtmosphereParameters gAtmosphere : packoffset(c0);
    uint gOutputUav    : packoffset(c6.x);
    uint gWidth        : packoffset(c6.y);
    uint gHeight       : packoffset(c6.z);
    uint gStepCount    : packoffset(c6.w);
    uint gDebugRMuUav  : packoffset(c7.x);  // 検証用: 各テクセルの (r [km], mu) を書く。不要なら無効値
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

    if (gDebugRMuUav != kInvalidDescriptorIndex) {
        RWTexture2D<float4> debugOutput = ResourceDescriptorHeap[gDebugRMuUav];
        debugOutput[id.xy] = float4(r, mu, 0.0, 0.0);
    }
}
