#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 3 の数値検証（design.md §8.2 V3 / V4 / V9 と、Multi-scattering LUT の照合）。
/// GPU の結果を EXR + サイドカーで保存し、比較は Python（reference/validate_phase3.py）で行う。
///   "multiscattering" … Multi-scattering LUT（float32）。検証用（太陽方向は入れ子 256 区間、視線 256 区間）と
///                        既定設定（AtmosphereQuality: 40 区間・2 次配置、Transmittance LUT）の 2 通り、
///                        各テクセルの (r, μs)
///   "sky_lut"         … V3: Sky-View LUT 経由の描画 vs ピクセルごとのレイマーチ（同じ近似）
///   "pathtrace"       … V4: LUT 描画・レイマーチ vs パストレーサー。散乱 1 回のパストレースと単散乱の照合も
///   "pathtrace_aerosol" … V4b（design.md §5.8）: outDir/media の表の媒質でパストレース（散乱 1 回 / 32 回、
///                        6 シナリオの全天、低い太陽と夕方の太陽の周り）。比較は reference/validate_v4b.py
///   "pathtrace_aerosol_views" … V4b の見た目の比較: C+A+B+D で太陽の方と反対の画面（画角 50°、地表アルベド 0.3）。
///                        比較は reference/compare_v4b_views.py
///   "spectral"        … V9: 係数セットごとの単散乱（入れ子レイマーチ）。Python の分光計算と比べる
///   "sun_closeup"     … V3b: 太陽の周り（画角 6°）を LUT 描画とレイマーチで描く。Mie の g と太陽天頂角を振る
///                        （design.md §14「Mie の g が大きいとき…」）
int runPhase3Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir);

} // namespace batch
