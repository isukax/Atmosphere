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
#include <glm/vec4.hpp>
#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace atmosphere {

/// GPU に上げた位相関数の表がないこと（gfx::kInvalidDescriptorIndex と同じ値。シェーダーは Cornette-Shanks を使う）
inline constexpr uint32_t kNoMiePhaseTable = 0xFFFFFFFFu;
/// 位相関数の表の点の数（θ = π·u²。shaders/atmosphere/AtmosphereIntegrate.hlsli の kMiePhaseTableCount）
inline constexpr uint32_t kMiePhaseTableCount = 2049;

/// エアロゾルのプリセットの Mie の 1 成分（design.md §5.9）。
struct MieComponentPreset {
    std::string name;
    glm::vec3 scatteringPerKm{0.0f};  // 鉛直分布の値が 1 のところ
    glm::vec3 absorptionPerKm{0.0f};
    float boundaryLayerTopKm = 0.0f;  // 鉛直分布 ρ(h) = exp(−max(h − h_top, 0)/H) + a·exp(−h/H_FT)
    float scaleHeightKm = 1.2f;
    float freeTropDensity = 0.0f;        // 自由大気の背景の a（0 ならなし）
    float freeTropScaleHeightKm = 1.0f;  // H_FT（> 0）
    float phaseG = 0.8f;  // Cornette-Shanks の g。表のときは平均の余弦（G。パストレーサーのサンプリング用）
    std::vector<glm::vec4> phaseTable;          // 位相関数の表 [sr⁻¹]（RGB + 0）。空なら Cornette-Shanks
    uint32_t phaseTableSrv = kNoMiePhaseTable;  // GPU に上げた表（MiePhaseTables が設定する）
};

/// 2 つ目の Mie の成分（2 成分のプリセット。design.md §5.9）。意味は AtmosphereParams の成分 0 の同名の値と同じ。
struct MieComponentParams {
    glm::vec3 scatteringPerKm{0.0f};
    glm::vec3 absorptionPerKm{0.0f};
    float scaleHeightKm = 1.2f;
    float boundaryLayerTopKm = 0.0f;
    float freeTropDensity = 0.0f;
    float freeTropScaleHeightKm = 1.0f;
    float phaseG = 0.8f;
    uint32_t phaseTableSrv = kNoMiePhaseTable;
};

/// エアロゾルのプリセット（research.md §12 の変種。base / base+A / base+B / base+A+B / C / C+A / C+B / C+A+B / C+D /
/// C+A+B+D）。
struct AerosolPreset {
    std::string name;
    float aod550 = 0.0f;     // 550 nm の鉛直の光学的厚さ（AOD を変えるときの基準、design.md §5.10）
    glm::vec3 aodRgb{0.0f};  // 参考: 鉛直の光学的厚さ（RGB）
    std::vector<MieComponentPreset> components;
};

/// data/atmosphere_coefficients.json の 1 セット（reference/derive_coefficients.py が生成）。
struct CoefficientSet {
    std::string name;
    display::WorkingColorSpace workingSpace = display::WorkingColorSpace::Rec709;
    glm::vec3 rayleighScatteringPerKm{0.0f};
    float mieScatteringPerKm = 0.0f;
    float mieAbsorptionPerKm = 0.0f;
    glm::vec3 ozoneAbsorptionPerKm{0.0f};
    glm::vec3 sunColor{1.0f};  // 作業色空間の RGB、輝度 Y = 1
    std::vector<AerosolPreset> aerosolPresets;  // 作業色空間ごとに持つ（B の係数・A の位相関数の RGB が変わるため）

    const AerosolPreset* findAerosolPreset(const std::string& presetName) const;
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
    float mieBoundaryLayerTopKm = 0.0f;            // 0 なら指数分布（design.md §5.9）
    float mieFreeTropDensity = 0.0f;               // 自由大気の背景の項 a·exp(−h/H_FT) の a（0 ならなし）
    float mieFreeTropScaleHeightKm = 1.0f;         // H_FT（> 0）
    float miePhaseG = 0.8f;                        // Cornette-Shanks の g（表のときは平均の余弦）
    uint32_t miePhaseTableSrv = kNoMiePhaseTable;  // 位相関数の表（GPU）。無効値なら Cornette-Shanks
    // [POINT] 上の mie* は Mie の成分 0。2 成分のプリセット（鉛直分布が違う fine / coarse、design.md §5.9）では
    //         成分 1 を mie1 に持つ
    uint32_t mieComponentCount = 1;
    MieComponentParams mie1;
    glm::vec3 ozoneAbsorptionPerKm{0.650e-3f, 1.881e-3f, 0.085e-3f};
    float ozoneCenterKm = 25.0f;
    float ozoneHalfWidthKm = 15.0f;
    glm::vec3 groundAlbedo{0.3f};
    // 太陽
    // [POINT] 大気上端・太陽方向に垂直な面での照度 [lx]。既定値はここに書かず、
    //         data/atmosphere_coefficients.json の solar_illuminance_lux_from_spectrum
    //         （ASTM G-173 から計算、約 133,121 lx）を loadDefaultParams() で設定する。
    //         太陽の色（sunColor）と同じスペクトルから求めた値にして、出どころを 1 つにするため（design.md D14）。
    float sunIlluminanceLux = 0.0f;
    glm::vec3 sunColor{1.0f};             // 作業色空間の RGB、Y = 1
    float sunAngularRadiusRad = 0.004675f;
    float sunZenithDeg = 45.0f;
    float sunAzimuthDeg = 0.0f;
    // ワールド原点（y = 0）の惑星表面からの高度
    float originAltitudeM = 0.0f;
    // 係数の由来
    std::string coefficientSet = "Hillaire2020";
    std::string aerosolPreset = "base";  // [POINT] 係数セットを替えても、同じ名前のプリセットを選び直す
    float aerosolPresetAod550 = 0.0f;    // 今のプリセットの AOD550（applyAerosolPreset が設定する）
    // [POINT] UI で決めた AOD550（design.md §5.10）。設定されていれば、プリセットや係数セットを替えても保ち、
    //         Mie の σs・σa（全成分・RGB）に AOD / プリセットの AOD を掛ける。未設定ならプリセットの量のまま
    std::optional<float> aerosolAod550Override;
    // [POINT] UI で Cornette-Shanks の g を変えた。g や AOD をプリセットの名前のまま記録すると、サイドカーと JSON の
    //         プリセットの値が食い違うので、名前に " (modified)" を付けて記録する（aerosolPresetModified()）
    bool miePhaseGModified = false;
    display::WorkingColorSpace workingSpace = display::WorkingColorSpace::Rec709;

    /// 係数セットの値（Rayleigh・Mie・オゾン・太陽の色・作業色空間）で上書きする。
    /// セットに aerosolPreset と同じ名前のプリセットがあれば、Mie はそのプリセットの値にする（変更は消える）。
    void applyCoefficientSet(const CoefficientSet& set);
    /// エアロゾルのプリセットの Mie の値で上書きする（1 成分か 2 成分。3 成分以上は例外）。
    /// 位相関数の表のプリセットは、表を GPU に上げてあること（MiePhaseTables）。そうでなければ例外。
    /// g の変更は消える。AOD の上書き（aerosolAod550Override）は保ち、新しいプリセットの係数に掛ける。
    void applyAerosolPreset(const AerosolPreset& preset);
    /// AOD550 を設定する（nullopt ならプリセットの量に戻す）。preset は今のプリセット（名前が違えば例外）。
    /// σs・σa は毎回プリセットの値から計算し直す（掛け算を重ねない）。g・鉛直分布・位相関数は変えない。
    void setAerosolAod550(const AerosolPreset& preset, std::optional<float> aod550);
    /// 今の AOD550（上書きがあればその値、なければプリセットの値）
    float aerosolAod550() const { return aerosolAod550Override.value_or(aerosolPresetAod550); }
    /// プリセットの値から変えたか（g か AOD）
    bool aerosolPresetModified() const { return miePhaseGModified || aerosolAod550Override.has_value(); }
    /// 記録・表示用のプリセットの名前（変更していれば "C (modified)" のように）
    std::string aerosolPresetLabel() const;
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
    float mieBoundaryLayerTopKm;
    uint32_t miePhaseTableSrv;
    uint32_t mieComponentCount;
    float mie1BoundaryLayerTopKm;  // 以下、2 つ目の Mie の成分（1 成分なら係数 0）
    float mie1ScaleHeightKm;
    float mie1ScatteringPerKm[3];
    float mie1PhaseG;
    float mie1ExtinctionPerKm[3];
    uint32_t mie1PhaseTableSrv;
    float mieFreeTropDensity;  // 自由大気の背景の項（design.md §5.9）
    float mieFreeTropScaleHeightKm;
    float mie1FreeTropDensity;
    float mie1FreeTropScaleHeightKm;

    bool operator==(const AtmosphereGpu&) const = default;
};
static_assert(sizeof(AtmosphereGpu) == 10 * 16);

/// 既定のパラメータ: 係数セット "Hillaire2020" を適用し、太陽照度を JSON のスペクトル由来の値にする。
AtmosphereParams loadDefaultParams(const std::filesystem::path& coefficientPath);

AtmosphereGpu toGpu(const AtmosphereParams& params);

/// Mie の一般形（2 成分か自由大気の背景）を扱うシェーダーの版（MIE_GENERAL = 1）が要るか（design.md §5.9）。
/// false なら 1 成分・背景なしの版（Phase 4.8 と同じコード）で同じ結果になる。
bool usesGeneralMieMedium(const AtmosphereParams& params);

/// 惑星座標 [km] でのカメラ位置: ワールド座標 [m] × 0.001 + (0, R_bottom + 原点高度, 0)。
glm::vec3 worldToPlanetKm(const AtmosphereParams& params, const glm::vec3& worldPosM);

/// 検証ダンプのサイドカーに書くため、全パラメータを単位付きの名前で JSON にする。
nlohmann::json toJson(const AtmosphereParams& params);

} // namespace atmosphere
