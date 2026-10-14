#pragma once

// GT7 Tone Mapping の CPU 側実装。
//
// 出典: Polyphony Digital「GT7 Tone Mapping」サンプル
//       (Copyright (c) 2025 Polyphony Digital Inc., MIT License)
//       K. Yasutomi, K. Suzuki, H. Uchimura, "Driving Toward Reality: Physically Based Tone
//       Mapping and Perceptual Fidelity in Gran Turismo 7", SIGGRAPH 2025 PBS Course.
// 移植元: GraphicsTest/LUTGenGT7/gt7_tone_mapping.py（元の C++ と照合済みの Python 移植）
//
// 役割:
//   1. ピーク輝度が変わったときにカーブ定数を計算し、シェーダーへ渡す（GT7Constants）。
//   2. UI のピクセルピッカー用に、同じ処理を CPU でも評価する（applyGT7）。
// 正本はシェーダー側（shaders/display/GT7ToneMap.hlsli）で、こちらはその鏡像。

#include <glm/vec3.hpp>

namespace display {

/// [POINT] GT のフレームバッファ規約: fb 値 1.0 = 表示輝度 100 cd/m²。
inline constexpr float kGTReferenceLuminanceNits = 100.0f;
/// [POINT] SDR の白（出力 1.0）を 250 nit のペーパーホワイトとみなす（GT7 の前提。sRGB の 2.5 倍）。
inline constexpr float kGTSdrPaperWhiteNits = 250.0f;
/// [POINT] HDR のピーク輝度の範囲。下限 250 はカーブのパラメータが 250 nit 前提で
///         決められているため（サンプルのコメントによる）。上限は PQ の 10,000 nit。
inline constexpr float kGTMinPeakNits = 250.0f;
inline constexpr float kGTMaxPeakNits = 10000.0f;

enum class GT7Mode { SDR, HDR };

/// シェーダーへ渡すカーブ定数。ピーク輝度（とモード）が変わったときだけ再計算する。
struct GT7Constants {
    float targetFb;        // 目標ピーク [fb]（SDR: 2.5、HDR: peakNits/100）
    float sdrCorrection;   // SDR: 1/2.5（出力を [0,1] に）、HDR: 1
    float midPoint;        // 0.538（トゥと線形区間の境）
    float linearSection;   // 0.444（ピークに対する線形区間の終端の割合）
    float toeStrength;     // 1.280
    float k;               // (linearSection - 1) / (alpha - 1)
    float kA;              // ショルダーの漸近値 = target·(linearSection + k)
    float targetUcsI;      // target の ICtCp 輝度 I（クロマフェードの基準）
    float blendRatio;      // 0.6（チャンネル別カーブと UCS 版のブレンド比）
    float fadeStart;       // 0.98
    float fadeEnd;         // 1.16
};

GT7Constants makeGT7Constants(GT7Mode mode, float peakNits);

/// 入力: linear Rec.2020 の fb 値。出力: SDR は [0,1]（1 = SDR 白）、HDR は fb（×100 で nit）。
glm::vec3 applyGT7(const GT7Constants& c, const glm::vec3& rec2020Fb);

/// PQ（ST 2084）の逆 EOTF。入力 nit、出力 [0,1] の信号値。
float pqEncodeNits(float nits);

} // namespace display
