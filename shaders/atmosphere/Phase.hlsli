// 位相関数 [sr⁻¹]（全立体角で積分すると 1）。cosTheta は入射方向と散乱方向のなす角の余弦
// （ここでは「視線方向」と「太陽へ向かう方向」の内積）。
#ifndef PHASE_HLSLI
#define PHASE_HLSLI

static const float kPi = 3.14159265358979;

// Rayleigh: 3/(16π)·(1 + cos²θ)
float RayleighPhase(float cosTheta)
{
    return 3.0 / (16.0 * kPi) * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks（Mie の近似。Hillaire 2020 の既定）:
//   3/(8π) · (1−g²)(1+cos²θ) / ((2+g²)(1+g²−2g·cosθ)^1.5)
// g は非対称パラメータ（0.8 で強い前方散乱 = 太陽の周りの光輪）。
float CornetteShanksPhase(float g, float cosTheta)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return 3.0 / (8.0 * kPi) * (1.0 - g2) * (1.0 + cosTheta * cosTheta) / ((2.0 + g2) * denom * sqrt(denom));
}

#endif // PHASE_HLSLI
