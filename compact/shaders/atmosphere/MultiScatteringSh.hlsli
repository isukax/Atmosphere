

#ifndef MULTI_SCATTERING_SH_HLSLI
#define MULTI_SCATTERING_SH_HLSLI

static const uint kMsShCount = 6;
static const float kMsShY00 = 0.282094791774;
static const float kMsShY1 = 0.488602511903;
static const float kMsShY20 = 0.315391565253;
static const float kMsShY21 = 1.092548430592;
static const float kMsShY22 = 0.546274215296;

struct MsShBasis
{
    float y[kMsShCount];
};

MsShBasis EvalMsShBasis(float x, float z, float y2)
{
    MsShBasis b;
    b.y[0] = kMsShY00;
    b.y[1] = kMsShY1 * z;
    b.y[2] = kMsShY1 * x;
    b.y[3] = kMsShY20 * (3.0 * z * z - 1.0);
    b.y[4] = kMsShY21 * x * z;
    b.y[5] = kMsShY22 * (x * x - y2);
    return b;
}

float2 MsShAtlasUv(float2 tileUv, uint b)
{
    return float2((tileUv.x + float(b)) / float(kMsShCount), tileUv.y);
}

#endif
