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
    if (preset.components.size() != 1) {
        throw std::runtime_error("Aerosol preset " + preset.name + ": only 1 Mie component is supported (Phase 4.8)");
    }
    const MieComponentPreset& c = preset.components.front();
    if (!c.phaseTable.empty() && c.phaseTableSrv == kNoMiePhaseTable) {
        // [POINT] 表を GPU に上げずに使うと、シェーダーは黙って Cornette-Shanks に戻る。誤りに気付けるよう止める
        throw std::runtime_error("Aerosol preset " + preset.name + ": phase table is not uploaded (MiePhaseTables)");
    }
    mieScatteringPerKm = c.scatteringPerKm;
    mieAbsorptionPerKm = c.absorptionPerKm;
    mieBoundaryLayerTopKm = c.boundaryLayerTopKm;
    mieScaleHeightKm = c.scaleHeightKm;
    miePhaseG = c.phaseG;
    miePhaseTableSrv = c.phaseTable.empty() ? kNoMiePhaseTable : c.phaseTableSrv;
    aerosolPreset = preset.name;
    aerosolPresetModified = false;
}

std::string AtmosphereParams::aerosolPresetLabel() const
{
    return aerosolPresetModified ? aerosolPreset + " (modified)" : aerosolPreset;
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
    g.mieComponentCount = 1;
    g.mie1PhaseTableSrv = kNoMiePhaseTable;
    return g;
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
    return {
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
        {"aerosol_preset", p.aerosolPresetLabel()},
        {"aerosol_preset_modified", p.aerosolPresetModified},
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
}

} // namespace atmosphere
