#pragma once

#include "gfx/Pipeline.h"

#include <filesystem>
#include <string_view>

namespace batch {

/// Phase 1 の数値検証（design.md §8.2 V5 / V6）。表示パスの実際のシェーダーをオフスクリーン RT に
/// 描いて読み戻し、入力と出力を EXR + サイドカーで `outDir` に保存する。比較は Python 側
/// （reference/validate_phase1.py）で行う。
///   "gt7" … GT7 Tone Mapping（SDR / HDR450 / HDR1000 / HDR2000）。float32 RT に生出力を書く。
///   "pq"  … PQ エンコード。R10G10B10A2 RT に書いて 10bit コードを読み戻す。
/// 成功で 0 を返す。
int runPhase1Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir);

} // namespace batch
