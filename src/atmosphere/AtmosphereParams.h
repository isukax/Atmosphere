#pragma once

// 大気の物理パラメータ（docs/design.md §5.1）。Shoga へ移植する単位の一部。
//
// 単位の規約（§2.5 の命名規約）:
//   *Km      … km（大気計算は惑星中心原点の km で行う。m → km の変換は toGpu() の 1 箇所だけ）
//   *PerKm   … km⁻¹（散乱・吸収係数）
//   *Lux     … lx（照度）
//   *Rad/Deg … 角度

#include "display/ColorSpace.h"

#include <glm/vec3.hpp>
#include <nlohmann/json_fwd.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace atmosphere {

/// data/atmosphere_coefficients.json の 1 セット（reference/derive_coefficients.py が生成）。
struct CoefficientSet {
    std::string name;
    display::WorkingColorSpace workingSpace = display::WorkingColorSpace::Rec709;
    glm::vec3 rayleighScatteringPerKm{0.0f};
    float mieScatteringPerKm = 0.0f;
    float mieAbsorptionPerKm = 0.0f;
    glm::vec3 ozoneAbsorptionPerKm{0.0f};
    glm::vec3 sunColor{1.0f};  // 作業色空間の RGB、輝度 Y = 1
};

/// [POINT] 係数はアプリに手書きしない。Python で分光データから導出した JSON を読む（§2.6）。
std::vector<CoefficientSet> loadCoefficientSets(const std::filesystem::path& path);
/// 同じ JSON にある「分光データから求めた大気圏外の太陽照度 [lx]」（参考値。UI に表示する）。
float loadSolarIlluminanceFromSpectrumLux(const std::filesystem::path& path);

struct AtmosphereParams {
    // 幾何
    float bottomRadiusKm = 6360.0f;
    float topRadiusKm = 6460.0f;
    // 媒質（海面高度 / オゾン最大密度での値）
    glm::vec3 rayleighScatteringPerKm{5.802e-3f, 13.558e-3f, 33.1e-3f};
    float rayleighScaleHeightKm = 8.0f;
    glm::vec3 mieScatteringPerKm{3.996e-3f};
    glm::vec3 mieAbsorptionPerKm{0.444e-3f};
    float mieScaleHeightKm = 1.2f;
    float miePhaseG = 0.8f;
    glm::vec3 ozoneAbsorptionPerKm{0.650e-3f, 1.881e-3f, 0.085e-3f};
    float ozoneCenterKm = 25.0f;
    float ozoneHalfWidthKm = 15.0f;
    glm::vec3 groundAlbedo{0.3f};
    // 太陽
    float sunIlluminanceLux = 128000.0f;  // 大気上端、太陽方向に垂直な面での照度
    glm::vec3 sunColor{1.0f};             // 作業色空間の RGB、Y = 1
    float sunAngularRadiusRad = 0.004675f;
    float sunZenithDeg = 45.0f;
    float sunAzimuthDeg = 0.0f;
    // ワールド原点（y = 0）の惑星表面からの高度
    float originAltitudeM = 0.0f;
    // 係数の由来
    std::string coefficientSet = "Hillaire2020";
    display::WorkingColorSpace workingSpace = display::WorkingColorSpace::Rec709;

    /// 係数セットの値（Rayleigh・Mie・オゾン・太陽の色・作業色空間）で上書きする。
    void applyCoefficientSet(const CoefficientSet& set);
    /// 太陽へ向かう単位ベクトル（ワールド = 惑星座標の向き。+Y が天頂）。
    glm::vec3 sunDirection() const;
};

/// [POINT] シェーダーの AtmosphereParameters（shaders/atmosphere/AtmosphereCommon.hlsli）と同じ並び。
///         float3 + float で 16 バイトの行を埋める。
struct AtmosphereGpu {
    float bottomRadiusKm;
    float topRadiusKm;
    float rayleighScaleHeightKm;
    float mieScaleHeightKm;
    float rayleighScatteringPerKm[3];
    float miePhaseG;
    float mieScatteringPerKm[3];
    float ozoneCenterKm;
    float mieExtinctionPerKm[3];
    float ozoneHalfWidthKm;
    float ozoneAbsorptionPerKm[3];
    float sunAngularRadiusRad;
    float groundAlbedo[3];
    float pad0;

    bool operator==(const AtmosphereGpu&) const = default;
};
static_assert(sizeof(AtmosphereGpu) == 6 * 16);

AtmosphereGpu toGpu(const AtmosphereParams& params);

/// 惑星座標 [km] でのカメラ位置: ワールド座標 [m] × 0.001 + (0, R_bottom + 原点高度, 0)。
glm::vec3 worldToPlanetKm(const AtmosphereParams& params, const glm::vec3& worldPosM);

/// 検証ダンプのサイドカーに書くため、全パラメータを単位付きの名前で JSON にする。
nlohmann::json toJson(const AtmosphereParams& params);

} // namespace atmosphere
