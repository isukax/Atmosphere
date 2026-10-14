#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace io {

/// float32 RGBA の画像を EXR（float32 のまま、半精度にしない）で保存し、
/// 同名の `.meta.json` サイドカーを書く。
///
/// [POINT] EXR 単体では「値の単位」も「色空間」も分からない。サイドカーに
///         units（例: "fb (1.0 = 100 cd/m2)"、"PQ code (10bit)"）と color_space を必ず書き、
///         Python 側はメタデータだけを見て解釈する（GraphicsTest/HDR と同じ方針）。
void saveExr(const std::filesystem::path& path, uint32_t width, uint32_t height, const std::vector<float>& rgba,
             const nlohmann::json& meta);

} // namespace io
