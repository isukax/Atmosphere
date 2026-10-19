#pragma once

#include "atmosphere/AtmosphereParams.h"
#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 4 の検証（design.md §8.2 V3c / V3d / V8）。結果を EXR / JSON で保存し、比較は Python
/// （reference/validate_phase4.py）で行う。
///   "aerial_perspective" … V3c: 地形の空気遠近法。ボリューム経由の合成 vs 深度までのレイマーチ
///                          （透過率 RGB / RGB 平均）。単位照度あたりの輝度で保存
///   "ap_transmittance"   … V3h（Phase 4.6）: 空気遠近法の透過率だけを書き出す（ボリュームの RGB 平均 /
///                          Transmittance LUT の比 / ボリュームの RGB / レイマーチの RGB・RGB 平均）。
///                          カメラ 3 種、LUT は RGBA32F / RGBA16F
///   "sky_irradiance"     … V3d: 空からの照度の SH 係数と、同じカメラ位置の空（LUT 描画、2048×1024 方向）
///   "perf"               … V8: パスごとの GPU 時間（1920×1080 / 3840×2160、各 100 フレームの中央値）。Release で実行する。
///                          Phase 4.9 から、2 成分のプリセット（C+A+B+D）の毎フレームのパスも "*_c_a_b_d" で測る
int runPhase4Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir);

/// V3c / V3d を、与えた大気（エアロゾルのプリセットを適用したもの。Phase 4.9）で描く。g を変えたケースは描かない。
/// 出力のファイル名は "aerial_perspective" / "sky_irradiance" と同じ
int runAerialPerspectiveWithAtmosphere(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                                       const std::filesystem::path& outDir, const atmosphere::AtmosphereParams& params);
int runSkyIrradianceWithAtmosphere(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                                   const std::filesystem::path& outDir, const atmosphere::AtmosphereParams& params);

} // namespace batch
