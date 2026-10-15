#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 2 の数値検証（design.md §8.2 V1 / V2）。GPU の結果を EXR + サイドカーで保存し、
/// 比較は Python（reference/validate_phase2.py）で行う。サイドカーには使った大気パラメータを
/// すべて書くので、Python はそれを読んで同じ条件で積分する。
///   "transmittance"     … Transmittance LUT（float32、台形則 500 / 40 区間）と各テクセルの (r, mu)
///   "single_scattering" … 単散乱の GPU レイマーチ（入れ子 = LUT なし / LUT 使用）。方向グリッド 32×16
int runPhase2Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir);

} // namespace batch
