// 多重散乱の球面調和（SH）近似の基底とアトラス（design.md §5.11、MS_SH の版だけが使う）。
//
// 各点の局所の枠: z = 天頂（p/|p|）、x = 太陽の方向を水平に射影した向き、y = z × x。大気が天頂の周りに回転対称で、
// 太陽と天頂を含む面について鏡映対称なので、入ってくる散乱光の場は cos(mφ) の項だけで書ける（l ≤ 2 で 6 個）。
// [POINT] 基底は Condon–Shortley の符号を付けない多項式で書く（reference/ms_sh.py と同じ）。
//         並び b = 0〜5: (l, m) = (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
#ifndef MULTI_SCATTERING_SH_HLSLI
#define MULTI_SCATTERING_SH_HLSLI

static const uint kMsShCount = 6;
static const float kMsShY00 = 0.282094791774;  // 1/(2√π)
static const float kMsShY1 = 0.488602511903;   // √(3/(4π))
static const float kMsShY20 = 0.315391565253;  // √(5/(16π))
static const float kMsShY21 = 1.092548430592;  // √(15/(4π))
static const float kMsShY22 = 0.546274215296;  // √(15/(16π))

struct MsShBasis
{
    float y[kMsShCount];
};

// 局所の枠での方向の成分 x, z と y²（y² = 1 − x² − z²）から、6 つの基底の値
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

// [POINT] 係数はアトラス（32×32 のタイルを横に 6 枚、192×32）に置く。tileUv は Multi-scattering LUT の (r, μs) の
//         サブ UV（[0.5/32, 31.5/32]）なので、双線形補間でも隣のタイルは混ざらない
float2 MsShAtlasUv(float2 tileUv, uint b)
{
    return float2((tileUv.x + float(b)) / float(kMsShCount), tileUv.y);
}

#endif // MULTI_SCATTERING_SH_HLSLI
