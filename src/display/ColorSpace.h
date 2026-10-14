#pragma once

#include <glm/vec3.hpp>

namespace display {

/// 作業色空間（シーン・大気の RGB の意味）。design.md §2.6。
enum class WorkingColorSpace {
    Rec709,   // linear, Rec.709 原色, D65
    Rec2020,  // linear, Rec.2020 原色, D65
};

/// 行優先の 3x3 行列。`m[row][col]`。HLSL の `row_major float3x3` と同じ並び。
struct Mat3 {
    float m[3][3];
};

/// [POINT] 原色変換行列（白色点はどちらも D65 なので色順応は不要）。
///         値は ITU-R BT.2087 の Rec.709 → Rec.2020 行列と、その逆行列。
///         Python 検証（reference/validate_phase1.py）で colour-science の値と照合する。
extern const Mat3 kRec709ToRec2020;
extern const Mat3 kRec2020ToRec709;
extern const Mat3 kIdentity3;

/// 作業色空間 → Rec.2020（GT7 の入力色空間）。Rec.2020 作業時は単位行列。
const Mat3& workingToRec2020(WorkingColorSpace space);

/// [POINT] 輝度 Y の重み（その色空間の原色の Y 成分）。RGB が「Y = 輝度 [cd/m²]」となる
///         単位系なので、dot(rgb, weights) がそのまま cd/m²（または fb）になる。
glm::vec3 luminanceWeights(WorkingColorSpace space);

const char* toString(WorkingColorSpace space);

glm::vec3 mul(const Mat3& a, const glm::vec3& v);

} // namespace display
