#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 4.11 の検証（design.md §5.11、多重散乱の SH 近似）。比較は Python（reference/validate_phase4_11.py）。
///   "multiscattering_sh" … <outDir>/ms_sh: SH の Multi-scattering LUT（検証用の入れ子 256 区間 / 既定の設定）と、比較用の
///                          等方の LUT（既定）。プリセット・アルベド・AOD・g の組ごと（V3j-SH）
///                          <outDir>/sky_sh: 全天の SH の LUT 描画とレイマーチ、等方の LUT 描画（V3k-SH）
///                          <outDir>/v4b_sh: 地表アルベド 0 の LUT 描画（等方 / SH）。パストレースと比べる（V4b-2）
int runPhase411Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                    const std::filesystem::path& outDir);

} // namespace batch
