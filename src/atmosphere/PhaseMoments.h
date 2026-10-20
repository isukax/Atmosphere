#pragma once

// 位相関数の Legendre のモーメント χ_l = 2π ∫ p(μ) P_l(μ) dμ（l = 0〜2、RGB）。
// 多重散乱の球面調和の近似（design.md §5.11）で、最後の散乱を帯 l ごとに χ_l 倍する（Funk–Hecke）。

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <array>
#include <vector>

namespace atmosphere {

struct AtmosphereParams;

/// χ_0〜χ_2（RGB）
using PhaseMoments = std::array<glm::vec3, 3>;

/// 位相関数の表（θ = π·u²、2,049 点、RGB）のモーメント。補間は MiePhase（AtmosphereIntegrate.hlsli）と同じく
/// u で線形。χ_0 は表の正規化のずれ（~0.1%）をそのまま持つ（reference/ms_sh.py と同じ）
PhaseMoments legendreMomentsTable(const std::vector<glm::vec4>& table);

/// Cornette-Shanks（g）のモーメント（閉じた式。χ_0 = 1）
PhaseMoments legendreMomentsCornetteShanks(float g);

/// 今のパラメータの Mie の成分 0・1 のモーメント。表のときはプリセットで求めた値、Cornette-Shanks のときは
/// 今の g から求める（UI で g を変えても追従する）。成分 1 がなければ 0
struct MiePhaseMoments {
    PhaseMoments component[2]{};
};
MiePhaseMoments mieLegendreMoments(const AtmosphereParams& params);

} // namespace atmosphere
