#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 4.8 の検証（design.md §5.9、エアロゾルのプリセット・1 成分）。
/// GPU の結果を EXR + サイドカーで保存し、比較は Python（reference/validate_phase4_8.py）で行う。
///   "aerosol_presets" … プリセットごとに、Transmittance LUT（V3i）、単散乱の入れ子レイマーチ（V3i・V9）、
///                       Multi-scattering LUT（V3j）、全天の LUT 描画とレイマーチ（V3k）、太陽の周り（V3b）、
///                       地表アルベド 0 の LUT 描画（V4b。Phase 4.7 のパストレースと比べる）
int runPhase48Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                   const std::filesystem::path& outDir);

} // namespace batch
