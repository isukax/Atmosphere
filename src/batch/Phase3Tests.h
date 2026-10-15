#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 3 の数値検証（design.md §8.2 V3 / V4 / V9 と、Multi-scattering LUT の照合）。
/// GPU の結果を EXR + サイドカーで保存し、比較は Python（reference/validate_phase3.py）で行う。
///   "multiscattering" … Multi-scattering LUT（float32）。検証用（太陽方向は入れ子 256 区間、視線 256 区間）と
///                        既定設定（20 区間、Transmittance LUT）の 2 通り、各テクセルの (r, μs)
///   "sky_lut"         … V3: Sky-View LUT 経由の描画 vs ピクセルごとのレイマーチ（同じ近似）
///   "pathtrace"       … V4: LUT 描画・レイマーチ vs パストレーサー。散乱 1 回のパストレースと単散乱の照合も
///   "spectral"        … V9: 係数セットごとの単散乱（入れ子レイマーチ）。Python の分光計算と比べる
///   "sun_closeup"     … V3b: 太陽の周り（画角 6°）を LUT 描画とレイマーチで描く。Mie の g と太陽天頂角を振る
///                        （design.md §14「Mie の g が大きいとき…」）
int runPhase3Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir);

} // namespace batch
