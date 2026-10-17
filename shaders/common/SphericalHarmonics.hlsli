// 実数球面調和（L2、9 係数）と、余弦の畳み込みによる照度（Ramamoorthi & Hanrahan 2001）。
// 座標軸はワールドのまま（+Y 上）。射影と評価で同じ基底を使えば、軸の取り方は結果に影響しない。
#ifndef SPHERICAL_HARMONICS_HLSLI
#define SPHERICAL_HARMONICS_HLSLI

#include "common/Math.hlsli"

static const uint kShCount = 9;

void ShBasis(float3 n, out float b[9])
{
    b[0] = 0.282095;
    b[1] = 0.488603 * n.y;
    b[2] = 0.488603 * n.z;
    b[3] = 0.488603 * n.x;
    b[4] = 1.092548 * n.x * n.y;
    b[5] = 1.092548 * n.y * n.z;
    b[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    b[7] = 1.092548 * n.x * n.z;
    b[8] = 0.546274 * (n.x * n.x - n.y * n.y);
}

// [POINT] 余弦の畳み込みの係数 Â_l（l = 0, 1, 2）。輝度の SH 係数に掛けると照度の SH 係数になる:
//         E(n) = Σ Â_l · L_lm · Y_lm(n)。l ≥ 3 の成分は Â_l が小さく、L2 で照度の誤差は 3% 程度に収まる。
static const float kShCosineLobe[9] = {
    kPi,
    2.0 * kPi / 3.0, 2.0 * kPi / 3.0, 2.0 * kPi / 3.0,
    kPi / 4.0, kPi / 4.0, kPi / 4.0, kPi / 4.0, kPi / 4.0,
};

// 照度の SH 係数（余弦の畳み込み済み、rgb）から、法線 n の面の照度を評価する
float3 EvaluateShIrradiance(StructuredBuffer<float4> coefficients, float3 n)
{
    float b[9];
    ShBasis(n, b);
    float3 e = 0.0;
    [unroll] for (uint i = 0; i < kShCount; ++i) {
        e += coefficients[i].rgb * b[i];
    }
    return max(e, 0.0);
}

#endif // SPHERICAL_HARMONICS_HLSLI
