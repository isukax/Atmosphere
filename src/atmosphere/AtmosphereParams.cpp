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

} // namespace

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

void AtmosphereParams::applyCoefficientSet(const CoefficientSet& set)
{
    rayleighScatteringPerKm = set.rayleighScatteringPerKm;
    mieScatteringPerKm = glm::vec3(set.mieScatteringPerKm);
    mieAbsorptionPerKm = glm::vec3(set.mieAbsorptionPerKm);
    ozoneAbsorptionPerKm = set.ozoneAbsorptionPerKm;
    sunColor = set.sunColor;
    coefficientSet = set.name;
    workingSpace = set.workingSpace;
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
        {"mie_phase_g", p.miePhaseG},
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
