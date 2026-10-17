#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 4 の検証（design.md §8.2 V3c / V3d / V8）。結果を EXR / JSON で保存し、比較は Python
/// （reference/validate_phase4.py）で行う。
///   "aerial_perspective" … V3c: 地形の空気遠近法。ボリューム経由の合成 vs 深度までのレイマーチ
///                          （透過率 RGB / RGB 平均）。単位照度あたりの輝度で保存
///   "sky_irradiance"     … V3d: 空からの照度の SH 係数と、同じカメラ位置の空（LUT 描画、2048×1024 方向）
///   "perf"               … V8: パスごとの GPU 時間（1920×1080 / 3840×2160、各 100 フレームの中央値）。Release で実行する
int runPhase4Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir);

} // namespace batch
