#include "atmosphere/AtmosphereParams.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace atmosphere {
namespace {

static_assert(offsetof(AtmosphereGpu, rayleighScatteringPerKm) == 1 * 16);
static_assert(offsetof(AtmosphereGpu, mieScatteringPerKm) == 2 * 16);
static_assert(offsetof(AtmosphereGpu, mieExtinctionPerKm) == 3 * 16);
static_assert(offsetof(AtmosphereGpu, ozoneAbsorptionPerKm) == 4 * 16);
static_assert(offsetof(AtmosphereGpu, groundAlbedo) == 5 * 16);
static_assert(offsetof(AtmosphereGpu, miePhaseTableSrv) == 6 * 16);
static_assert(offsetof(AtmosphereGpu, mie1ScatteringPerKm) == 7 * 16);
static_assert(offsetof(AtmosphereGpu, mie1ExtinctionPerKm) == 8 * 16);
static_assert(offsetof(AtmosphereGpu, mieFreeTropDensity) == 9 * 16);

glm::vec3 vec3From(const nlohmann::json& j)
{
    return {j.at(0).get<float>(), j.at(1).get<float>(), j.at(2).get<float>()};
}

nlohmann::json toJson(const glm::vec3& v)
{
    return {v.x, v.y, v.z};
}

void store(float (&dst)[3], const glm::vec3& v)
{
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
}

MieComponentPreset parseMieComponent(const nlohmann::json& j)
{
    MieComponentPreset c;
    c.name = j.at("name").get<std::string>();
    c.scatteringPerKm = vec3From(j.at("scattering_per_km"));
    c.absorptionPerKm = vec3From(j.at("absorption_per_km"));
    c.boundaryLayerTopKm = j.at("boundary_layer_top_km").get<float>();
    c.scaleHeightKm = j.at("scale_height_km").get<float>();
    c.freeTropDensity = j.at("free_troposphere_density").get<float>();
    c.freeTropScaleHeightKm = j.at("free_troposphere_scale_height_km").get<float>();
    if (!(c.freeTropScaleHeightKm > 0.0f) || c.freeTropDensity < 0.0f) {
        // [POINT] H_FT ≤ 0 だとシェーダーで h = 0 のとき 0/0 になる（背景のない成分も H_FT は正の値で書く）
        throw std::runtime_error("Mie component " + c.name + ": free troposphere density must be >= 0 and scale "
                                 "height > 0");
    }
    const nlohmann::json& phase = j.at("phase");
    const std::string model = phase.at("model").get<std::string>();
    if (model == "cornette_shanks") {
        c.phaseG = phase.at("g").get<float>();
    } else if (model == "table") {
        c.phaseG = phase.at("asymmetry_g").at(1).get<float>();
        const nlohmann::json& values = phase.at("values_rgb");
        if (values.size() != kMiePhaseTableCount) {
            throw std::runtime_error("Mie phase table must have " + std::to_string(kMiePhaseTableCount) + " points");
        }
        c.phaseTable.reserve(values.size());
        for (const nlohmann::json& v : values) {
            c.phaseTable.emplace_back(v.at(0).get<float>(), v.at(1).get<float>(), v.at(2).get<float>(), 0.0f);
        }
    } else {
        throw std::runtime_error("Unknown Mie phase model: " + model);
    }
    return c;
}

} // namespace

const AerosolPreset* CoefficientSet::findAerosolPreset(const std::string& presetName) const
{
    for (const AerosolPreset& p : aerosolPresets) {
        if (p.name == presetName) {
            return &p;
        }
    }
    return nullptr;
}

std::vector<CoefficientSet> loadCoefficientSets(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Coefficient file not found: " + path.string());
    }
    const nlohmann::json root = nlohmann::json::parse(file);
    std::vector<CoefficientSet> sets;
    for (const nlohmann::json& s : root.at("sets")) {
        CoefficientSet set;
        set.name = s.at("name").get<std::string>();
        set.workingSpace = s.at("working_space").get<std::string>() == "Rec2020" ? display::WorkingColorSpace::Rec2020
                                                                                  : display::WorkingColorSpace::Rec709;
        set.rayleighScatteringPerKm = vec3From(s.at("rayleigh_scattering_per_km"));
        set.mieScatteringPerKm = s.at("mie_scattering_per_km").get<float>();
        set.mieAbsorptionPerKm = s.at("mie_absorption_per_km").get<float>();
        set.ozoneAbsorptionPerKm = vec3From(s.at("ozone_absorption_per_km"));
        set.sunColor = vec3From(s.at("sun_color"));
        if (s.contains("aerosol_presets")) {
            for (const nlohmann::json& p : s.at("aerosol_presets")) {
                AerosolPreset preset;
                preset.name = p.at("name").get<std::string>();
                preset.aod550 = p.at("aod550").get<float>();
                if (!(preset.aod550 > 0.0f)) {
                    // [POINT] AOD を変えるときに AOD / aod550 を掛けるので、0 だと割れない
                    throw std::runtime_error("Aerosol preset " + preset.name + ": aod550 must be > 0");
                }
                preset.aodRgb = vec3From(p.at("aod_rgb"));
                for (const nlohmann::json& c : p.at("components")) {
                    preset.components.push_back(parseMieComponent(c));
                }
                set.aerosolPresets.push_back(std::move(preset));
            }
        }
        sets.push_back(std::move(set));
    }
    return sets;
}

float loadSolarIlluminanceFromSpectrumLux(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Coefficient file not found: " + path.string());
    }
    return nlohmann::json::parse(file).at("solar_illuminance_lux_from_spectrum").get<float>();
}

AtmosphereParams loadDefaultParams(const std::filesystem::path& coefficientPath)
{
    AtmosphereParams params;
    for (const CoefficientSet& set : loadCoefficientSets(coefficientPath)) {
        if (set.name == "Hillaire2020") {
            params.applyCoefficientSet(set);
        }
    }
    params.sunIlluminanceLux = loadSolarIlluminanceFromSpectrumLux(coefficientPath);
    return params;
}

void AtmosphereParams::applyCoefficientSet(const CoefficientSet& set)
{
    rayleighScatteringPerKm = set.rayleighScatteringPerKm;
    mieScatteringPerKm = glm::vec3(set.mieScatteringPerKm);
    mieAbsorptionPerKm = glm::vec3(set.mieAbsorptionPerKm);
    ozoneAbsorptionPerKm = set.ozoneAbsorptionPerKm;
    sunColor = set.sunColor;
    coefficientSet = set.name;
    workingSpace = set.workingSpace;
    if (const AerosolPreset* preset = set.findAerosolPreset(aerosolPreset)) {
        applyAerosolPreset(*preset);
    }
}

void AtmosphereParams::applyAerosolPreset(const AerosolPreset& preset)
{
    if (preset.components.empty() || preset.components.size() > 2) {
        throw std::runtime_error("Aerosol preset " + preset.name + ": 1 or 2 Mie components are supported");
    }
    for (const MieComponentPreset& c : preset.components) {
        if (!c.phaseTable.empty() && c.phaseTableSrv == kNoMiePhaseTable) {
            // [POINT] 表を GPU に上げずに使うと、シェーダーは黙って Cornette-Shanks に戻る。誤りに気付けるよう止める
            throw std::runtime_error("Aerosol preset " + preset.name + ": phase table is not uploaded (MiePhaseTables)");
        }
    }
    const MieComponentPreset& c = preset.components.front();
    mieScatteringPerKm = c.scatteringPerKm;
    mieAbsorptionPerKm = c.absorptionPerKm;
    mieBoundaryLayerTopKm = c.boundaryLayerTopKm;
    mieScaleHeightKm = c.scaleHeightKm;
    mieFreeTropDensity = c.freeTropDensity;
    mieFreeTropScaleHeightKm = c.freeTropScaleHeightKm;
    miePhaseG = c.phaseG;
    miePhaseTableSrv = c.phaseTable.empty() ? kNoMiePhaseTable : c.phaseTableSrv;
    mieComponentCount = static_cast<uint32_t>(preset.components.size());
    mie1 = {};
    if (mieComponentCount == 2) {
        const MieComponentPreset& c1 = preset.components[1];
        mie1.scatteringPerKm = c1.scatteringPerKm;
        mie1.absorptionPerKm = c1.absorptionPerKm;
        mie1.scaleHeightKm = c1.scaleHeightKm;
        mie1.boundaryLayerTopKm = c1.boundaryLayerTopKm;
        mie1.freeTropDensity = c1.freeTropDensity;
        mie1.freeTropScaleHeightKm = c1.freeTropScaleHeightKm;
        mie1.phaseG = c1.phaseG;
        mie1.phaseTableSrv = c1.phaseTable.empty() ? kNoMiePhaseTable : c1.phaseTableSrv;
    }
    aerosolPreset = preset.name;
    aerosolPresetAod550 = preset.aod550;
    miePhaseGModified = false;
    if (aerosolAod550Override) {
        setAerosolAod550(preset, aerosolAod550Override);
    }
}

void AtmosphereParams::setAerosolAod550(const AerosolPreset& preset, std::optional<float> aod550)
{
    if (preset.name != aerosolPreset) {
        throw std::runtime_error("setAerosolAod550: preset " + preset.name + " is not the current preset " +
                                 aerosolPreset);
    }
    if (aod550 && !(*aod550 > 0.0f)) {
        throw std::runtime_error("setAerosolAod550: AOD must be > 0");
    }
    aerosolAod550Override = aod550;
    // [POINT] 同じ粒子・同じ鉛直分布なら σs・σa は AOD に比例する（design.md §5.10）。単散乱アルベド・波長依存・
    //         fine と coarse の割合・自由大気の割合は変わらないので、全成分の RGB に同じ倍率を掛ける。
    //         上書きがなければ倍率を掛けず、プリセットの値そのものにする（回帰で値が 1 ビットも変わらないように）
    const float k = aod550 ? *aod550 / preset.aod550 : 1.0f;
    const auto scaled = [&](const glm::vec3& v) { return aod550 ? v * k : v; };
    mieScatteringPerKm = scaled(preset.components.front().scatteringPerKm);
    mieAbsorptionPerKm = scaled(preset.components.front().absorptionPerKm);
    if (mieComponentCount == 2) {
        mie1.scatteringPerKm = scaled(preset.components[1].scatteringPerKm);
        mie1.absorptionPerKm = scaled(preset.components[1].absorptionPerKm);
    }
}

std::string AtmosphereParams::aerosolPresetLabel() const
{
    return aerosolPresetModified() ? aerosolPreset + " (modified)" : aerosolPreset;
}

glm::vec3 AtmosphereParams::sunDirection() const
{
    // 天頂角 θ（+Y からの角度）と方位角 φ（+X から +Z へ）による単位ベクトル。
    const float theta = sunZenithDeg * std::numbers::pi_v<float> / 180.0f;
    const float phi = sunAzimuthDeg * std::numbers::pi_v<float> / 180.0f;
    return {std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
}

AtmosphereGpu toGpu(const AtmosphereParams& p)
{
    AtmosphereGpu g{};
    g.bottomRadiusKm = p.bottomRadiusKm;
    g.topRadiusKm = p.topRadiusKm;
    g.rayleighScaleHeightKm = p.rayleighScaleHeightKm;
    g.mieScaleHeightKm = p.mieScaleHeightKm;
    store(g.rayleighScatteringPerKm, p.rayleighScatteringPerKm);
    g.miePhaseG = p.miePhaseG;
    store(g.mieScatteringPerKm, p.mieScatteringPerKm);
    g.ozoneCenterKm = p.ozoneCenterKm;
    // [POINT] 消散 = 散乱 + 吸収。Mie は吸収もある（単散乱アルベド 0.9）。Rayleigh は吸収しない。
    store(g.mieExtinctionPerKm, p.mieScatteringPerKm + p.mieAbsorptionPerKm);
    g.ozoneHalfWidthKm = p.ozoneHalfWidthKm;
    store(g.ozoneAbsorptionPerKm, p.ozoneAbsorptionPerKm);
    g.sunAngularRadiusRad = p.sunAngularRadiusRad;
    store(g.groundAlbedo, p.groundAlbedo);
    g.mieBoundaryLayerTopKm = p.mieBoundaryLayerTopKm;
    g.miePhaseTableSrv = p.miePhaseTableSrv;
    g.mieFreeTropDensity = p.mieFreeTropDensity;
    g.mieFreeTropScaleHeightKm = p.mieFreeTropScaleHeightKm;
    g.mieComponentCount = p.mieComponentCount;
    // [POINT] 1 成分でも成分 1 のスケールハイトは正の値にする（シェーダーは成分の数で分岐するが、念のため 0/0 を作らない）
    const MieComponentParams mie1 = p.mieComponentCount > 1 ? p.mie1 : MieComponentParams{};
    g.mie1BoundaryLayerTopKm = mie1.boundaryLayerTopKm;
    g.mie1ScaleHeightKm = mie1.scaleHeightKm;
    store(g.mie1ScatteringPerKm, mie1.scatteringPerKm);
    g.mie1PhaseG = mie1.phaseG;
    store(g.mie1ExtinctionPerKm, mie1.scatteringPerKm + mie1.absorptionPerKm);
    g.mie1PhaseTableSrv = mie1.phaseTableSrv;
    g.mie1FreeTropDensity = mie1.freeTropDensity;
    g.mie1FreeTropScaleHeightKm = mie1.freeTropScaleHeightKm;
    return g;
}

bool usesGeneralMieMedium(const AtmosphereParams& p)
{
    // [POINT] MIE_GENERAL = 0 の版は成分 1 と背景の項を読まない（AtmosphereCommon.hlsli）。どちらかがあれば 1 の版
    return p.mieComponentCount > 1 || p.mieFreeTropDensity > 0.0f;
}

glm::vec3 worldToPlanetKm(const AtmosphereParams& p, const glm::vec3& worldPosM)
{
    // [POINT] ワールド（m、+Y 上、原点 = 地表）→ 惑星座標（km、原点 = 惑星中心）。
    //         m → km の変換はここに集約する（散在させると単位の取り違えが起きる）。
    constexpr float kKmPerM = 0.001f;
    return worldPosM * kKmPerM + glm::vec3(0.0f, p.bottomRadiusKm + p.originAltitudeM * kKmPerM, 0.0f);
}

nlohmann::json toJson(const AtmosphereParams& p)
{
    nlohmann::json j = {
        {"coefficient_set", p.coefficientSet},
        {"working_space", display::toString(p.workingSpace)},
        {"bottom_radius_km", p.bottomRadiusKm},
        {"top_radius_km", p.topRadiusKm},
        {"rayleigh_scattering_per_km", toJson(p.rayleighScatteringPerKm)},
        {"rayleigh_scale_height_km", p.rayleighScaleHeightKm},
        {"mie_scattering_per_km", toJson(p.mieScatteringPerKm)},
        {"mie_absorption_per_km", toJson(p.mieAbsorptionPerKm)},
        {"mie_scale_height_km", p.mieScaleHeightKm},
        {"mie_boundary_layer_top_km", p.mieBoundaryLayerTopKm},
        {"mie_phase_g", p.miePhaseG},
        {"mie_phase_model", p.miePhaseTableSrv == kNoMiePhaseTable ? "cornette_shanks" : "table"},
        {"mie_free_troposphere_density", p.mieFreeTropDensity},
        {"mie_free_troposphere_scale_height_km", p.mieFreeTropScaleHeightKm},
        {"mie_component_count", p.mieComponentCount},
        {"aerosol_preset", p.aerosolPresetLabel()},
        // [POINT] 位相関数の表はプリセットの名前で JSON から引く（reference/atmosphere_ref.py）。
        //         aerosol_preset は " (modified)" が付くことがあるので、元の名前を別に書く
        {"aerosol_preset_name", p.aerosolPreset},
        {"aerosol_preset_modified", p.aerosolPresetModified()},
        {"mie_phase_g_modified", p.miePhaseGModified},
        {"aerosol_aod550", p.aerosolAod550()},
        {"aerosol_preset_aod550", p.aerosolPresetAod550},
        {"aerosol_aod550_overridden", p.aerosolAod550Override.has_value()},
        {"ozone_absorption_per_km", toJson(p.ozoneAbsorptionPerKm)},
        {"ozone_center_km", p.ozoneCenterKm},
        {"ozone_half_width_km", p.ozoneHalfWidthKm},
        {"ground_albedo", toJson(p.groundAlbedo)},
        {"sun_illuminance_lux", p.sunIlluminanceLux},
        {"sun_color", toJson(p.sunColor)},
        {"sun_angular_radius_rad", p.sunAngularRadiusRad},
        {"sun_zenith_deg", p.sunZenithDeg},
        {"sun_azimuth_deg", p.sunAzimuthDeg},
        {"origin_altitude_m", p.originAltitudeM},
    };
    if (p.mieComponentCount > 1) {
        // 2 つ目の Mie の成分（reference/atmosphere_ref.py の from_meta が読む）
        j["mie1"] = {
            {"scattering_per_km", toJson(p.mie1.scatteringPerKm)},
            {"absorption_per_km", toJson(p.mie1.absorptionPerKm)},
            {"scale_height_km", p.mie1.scaleHeightKm},
            {"boundary_layer_top_km", p.mie1.boundaryLayerTopKm},
            {"free_troposphere_density", p.mie1.freeTropDensity},
            {"free_troposphere_scale_height_km", p.mie1.freeTropScaleHeightKm},
            {"phase_g", p.mie1.phaseG},
            {"phase_model", p.mie1.phaseTableSrv == kNoMiePhaseTable ? "cornette_shanks" : "table"},
        };
    }
    return j;
}

} // namespace atmosphere
