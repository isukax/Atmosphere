// 乱数・ノイズ。
#ifndef NOISE_HLSLI
#define NOISE_HLSLI

// PCG ハッシュ（32bit 整数 → 32bit 整数）
uint PcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// [0,1) の一様乱数
float Hash01(uint v)
{
    return float(PcgHash(v) >> 8) * (1.0 / 16777216.0);
}

// [POINT] 三角分布ノイズ（範囲 [-1, 1)、平均 0）。2 つの一様乱数の差で作る。
//         量子化直前に ±1 LSB 幅で加えると、量子化誤差が信号と無相関になり、
//         空のなだらかなグラデーションでのバンディング（縞）が消える。
float TriangularNoise(uint2 pixel, uint frame, uint channel)
{
    uint seed = pixel.x * 73856093u ^ pixel.y * 19349663u ^ frame * 83492791u ^ channel * 2654435761u;
    return Hash01(seed) - Hash01(seed ^ 0x9E3779B9u);
}

#endif // NOISE_HLSLI
