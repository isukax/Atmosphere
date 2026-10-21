// 出力用の伝達関数（OETF / 逆 EOTF）。
#ifndef TRANSFER_FUNCTIONS_HLSLI
#define TRANSFER_FUNCTIONS_HLSLI

// [POINT] HDR10 の符号化。入力は表示輝度 [nit]、出力は [0,1] の PQ 信号値。
//         PQ は 0〜10,000 nit の絶対輝度を符号化する（SMPTE ST 2084）。負値は表現できない。
float3 PQEncodeNits(float3 nits)
{
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float3 y = saturate(nits / 10000.0);
    float3 ym = pow(y, m1);
    return pow((c1 + c2 * ym) / (1.0 + c3 * ym), m2);
}

// sRGB の OETF（区分関数）。入力 [0,1] の linear。
float3 SrgbEncode(float3 v)
{
    v = saturate(v);
    return select(v <= 0.0031308, v * 12.92, 1.055 * pow(v, 1.0 / 2.4) - 0.055);
}

// sRGB の EOTF（区分関数）。UI テクスチャの linear 化に使う。
float3 SrgbDecode(float3 v)
{
    return select(v <= 0.04045, v / 12.92, pow((v + 0.055) / 1.055, 2.4));
}

#endif // TRANSFER_FUNCTIONS_HLSLI
