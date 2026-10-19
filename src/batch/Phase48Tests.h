#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 4.8 / 4.9 の検証（design.md §5.9、エアロゾルのプリセット。4.9 で 2 成分の C+D・C+A+B+D を足した）。
/// GPU の結果を EXR + サイドカーで保存し、比較は Python（reference/validate_phase4_8.py・validate_phase4_9.py）で行う。
///   "aerosol_presets" … プリセットごとに、Transmittance LUT（V3i）、単散乱の入れ子レイマーチ（V3i・V9）、
///                       Multi-scattering LUT（V3j）、全天の LUT 描画とレイマーチ（V3k）、太陽の周り（V3b）、
///                       地表アルベド 0 の LUT 描画（V4b。パストレースと比べる）
///   "aerosol_scene"   … 2 成分のプリセット（Hillaire2020 の C+A+B+D）と、比較用に同じ量の 1 成分の C+A+B で、地形の
///                       空気遠近法（V3c と同じ）、空気遠近法の影の方式（V3g と同じ）、空からの照度（V3d と同じ）を
///                       <outDir>/<プリセット>/ に書く
int runPhase48Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                   const std::filesystem::path& outDir);

} // namespace batch
