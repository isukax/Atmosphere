// 数学定数と小さな関数
#ifndef MATH_HLSLI
#define MATH_HLSLI

static const float kPi = 3.14159265358979;

// 単位ベクトル n に直交する 2 本の単位ベクトル（Duff et al. 2017, "Building an Orthonormal Basis, Revisited"）
void OrthonormalBasis(float3 n, out float3 tangent, out float3 bitangent)
{
    const float s = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + n.z);
    const float b = n.x * n.y * a;
    tangent = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    bitangent = float3(b, s + n.y * n.y * a, -n.y);
}

#endif // MATH_HLSLI
