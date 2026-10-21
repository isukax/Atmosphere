// GT7 Tone Mapping（シェーダー側の正本）
//
// 出典: Polyphony Digital「GT7 Tone Mapping」サンプル
//       (Copyright (c) 2025 Polyphony Digital Inc., MIT License)
//       K. Yasutomi, K. Suzuki, H. Uchimura, "Driving Toward Reality: Physically Based Tone
//       Mapping and Perceptual Fidelity in Gran Turismo 7", SIGGRAPH 2025 PBS Course.
// 移植元: GraphicsTest/HDR/testbed/shaders/tonemap.hlsli（toneMapGT7）
//   変更点: カーブ定数と target の UCS 輝度を CPU で事前計算して定数で受け取る形にした
//           （src/display/GT7ToneMapping.cpp の makeGT7Constants）。
//
// 入力 : linear Rec.2020、GT フレームバッファ値（fb、1.0 = 100 cd/m²）
// 出力 : SDR … [0,1]（1.0 = SDR 白 = 250 nit 想定）
//        HDR … fb（[0, peakNits/100]。×100 で nit）
#ifndef GT7_TONE_MAP_HLSLI
#define GT7_TONE_MAP_HLSLI

struct GT7Params
{
    float targetFb;
    float sdrCorrection;
    float midPoint;
    float linearSection;
    float toeStrength;
    float k;
    float kA;
    float targetUcsI;
    float blendRatio;
    float fadeStart;
    float fadeEnd;
};

// fb 値 → PQ 信号値（GT7 の UCS 変換用。1.0 fb = 100 nit を 10,000 nit 基準に直してから PQ）
float GT7InverseEotfFb(float fb)
{
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float y = max(fb * 100.0 / 10000.0, 0.0);
    float ym = pow(y, m1);
    return exp2(m2 * (log2(c1 + c2 * ym) - log2(1.0 + c3 * ym)));
}

// PQ 信号値 → fb 値
float GT7EotfFb(float n)
{
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    n = saturate(n);
    float np = pow(n, 1.0 / m2);
    float y = pow(max(np - c1, 0.0) / (c2 - c3 * np), 1.0 / m1);
    return y * 10000.0 / 100.0;
}

// [POINT] GT7 は知覚均等色空間（UCS）として ICtCp を使う（サンプルの既定。JzAzBz も選べる）。
//         入力は linear Rec.2020 の fb 値。
float3 GT7RgbToICtCp(float3 rgb)
{
    float3 lms;
    lms.x = dot(rgb, float3(1688.0, 2146.0, 262.0)) / 4096.0;
    lms.y = dot(rgb, float3(683.0, 2951.0, 462.0)) / 4096.0;
    lms.z = dot(rgb, float3(99.0, 309.0, 3688.0)) / 4096.0;
    float3 p = float3(GT7InverseEotfFb(lms.x), GT7InverseEotfFb(lms.y), GT7InverseEotfFb(lms.z));
    return float3(
        (2048.0 * p.x + 2048.0 * p.y) / 4096.0,
        (6610.0 * p.x - 13613.0 * p.y + 7003.0 * p.z) / 4096.0,
        (17933.0 * p.x - 17390.0 * p.y - 543.0 * p.z) / 4096.0);
}

float3 GT7ICtCpToRgb(float3 v)
{
    float3 p = float3(
        v.x + 0.00860904 * v.y + 0.11103 * v.z,
        v.x - 0.00860904 * v.y - 0.11103 * v.z,
        v.x + 0.560031 * v.y - 0.320627 * v.z);
    float3 lms = float3(GT7EotfFb(p.x), GT7EotfFb(p.y), GT7EotfFb(p.z));
    return max(float3(
        dot(lms, float3(3.43661, -2.50645, 0.0698454)),
        dot(lms, float3(-0.79133, 1.9836, -0.192271)),
        dot(lms, float3(-0.0259499, -0.0989137, 1.12486))), 0.0);
}

// GT Tone Mapping V2 カーブ（チャンネル独立）
//   x <  linearSection·target : トゥ（midPoint 以下）と線形区間を smoothstep でブレンド
//   x >= linearSection·target : 指数ショルダー（target に漸近）
float3 GT7Curve(float3 x, GT7Params p)
{
    float3 weightLinear = smoothstep(0.0, p.midPoint, x);
    // [POINT] トゥは midPoint より上では使われない。入力を midPoint でクランプして、
    //         大きな x で pow() が溢れて 0·INF = NaN になるのを防ぐ（移植元の対策）。
    float3 toeMapped = p.midPoint * pow(min(x, p.midPoint) / p.midPoint, p.toeStrength);
    float3 lower = (1.0 - weightLinear) * toeMapped + weightLinear * x;
    // [POINT] サンプルの kA + kB·exp(x·kC) と代数的に同じ式。ショルダーは
    //         x >= linearSection·target でしか使わないので、指数を 0 以下にクランプして
    //         使われない側の exp の溢れを防ぐ。
    float3 shoulderExponent = min((p.linearSection - x / p.targetFb) / p.k, 0.0);
    float3 shoulder = p.kA - p.targetFb * p.k * exp(shoulderExponent);
    return max(lerp(lower, shoulder, step(p.linearSection * p.targetFb, x)), 0.0);
}

float3 GT7ToneMap(float3 rgb, GT7Params p)
{
    rgb = max(rgb, 0.0);
    // Step 1: チャンネル別カーブで「ねじれた（hue がずれた）」RGB を作る
    float3 ucs = GT7RgbToICtCp(rgb);
    float3 skewedRgb = GT7Curve(rgb, p);
    // Step 2-3: UCS で彩度を落とした「ねじれを戻した」RGB を作る。
    //   輝度はカーブ後（skewed）の I、色度は元の Ct/Cp を、元の輝度が target に近づくほど 0 へ。
    float3 skewedUcs = GT7RgbToICtCp(skewedRgb);
    float chromaScale = 1.0 - smoothstep(p.fadeStart, p.fadeEnd, ucs.x / p.targetUcsI);
    float3 scaledRgb = GT7ICtCpToRgb(float3(skewedUcs.x, ucs.yz * chromaScale));
    // Step 4: 2 つを RGB（linear Rec.2020）でブレンドし、target でハードクリップ
    float3 blended = lerp(skewedRgb, scaledRgb, p.blendRatio);
    return p.sdrCorrection * min(blended, p.targetFb);
}

#endif // GT7_TONE_MAP_HLSLI
