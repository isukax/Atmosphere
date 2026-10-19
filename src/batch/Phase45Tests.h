#pragma once

#include "atmosphere/AtmosphereParams.h"
#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 4.5 の検証（design.md §8.2 V3e / V3f）。結果を EXR / JSON で保存し、比較は Python
/// （reference/validate_phase4_5.py）で行う。
///   "shadow_surface"            … V3e: 地表の太陽の見える割合（シャドウマップ、PCF）。R = 見える割合、G = N·L、
///                                 B = 近景のカスケードの重み。メタデータにカスケードの範囲と球の位置を書く
///   "shadow_aerial_perspective" … V3f: 影ありの空気遠近法（ボリューム / 深度までのレイマーチ）と、影なしのレイマーチ
///   "shadow_ap_methods"         … V3g: 空気遠近法の影の方式（§5.6 の 0 / 1 / 5）と、基準のレイマーチ（影あり / なし）
int runPhase45Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                   const std::filesystem::path& outDir);

/// V3g を、与えた大気（エアロゾルのプリセットを適用したもの。Phase 4.9）で描く。出力は "shadow_ap_methods" と同じ
int runShadowApMethodsWithAtmosphere(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                                     const std::filesystem::path& outDir, const atmosphere::AtmosphereParams& params);

} // namespace batch
