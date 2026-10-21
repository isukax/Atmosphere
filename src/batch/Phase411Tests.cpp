#include "batch/Phase411Tests.h"

#include "atmosphere/AtmosphereRenderer.h"
#include "atmosphere/MiePhaseTables.h"
#include "io/ExrWriter.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <functional>

namespace batch {
namespace {

const std::filesystem::path kCoefficientPath = std::filesystem::path(ATMOS_DATA_DIR) / "atmosphere_coefficients.json";

std::vector<float> toFloats(const std::vector<uint8_t>& bytes)
{
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

/// ファイル名: "+" を "_" に（"C+A+B" → "C_A_B"）
std::string presetFileName(std::string name)
{
    std::replace(name.begin(), name.end(), '+', '_');
    return name;
}

nlohmann::json withFields(nlohmann::json base, const nlohmann::json& extra)
{
    base.update(extra);
    return base;
}

const char* modelName(atmosphere::MultiScatteringModel model)
{
    return model == atmosphere::MultiScatteringModel::ShL2Ground ? "sh_l2_ground" : "isotropic";
}

/// SH の LUT のアトラスの並び（reference/validate_phase4_11.py が読む）
const nlohmann::json kShLayout = {
    {"tiles", 6},
    {"tile_size", 32},
    {"coefficients_lm", {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}}},
    {"basis", "real SH, cos(m phi) terms only, no Condon-Shortley phase; local frame z = zenith, x = sun azimuth"},
    {"units", "SH coefficients of the incoming scattered radiance per illuminance [sr^-1]"},
    // F（閉じ方の伝達行列）の求積: 行（高度）ごとに天頂角の 1 次元の求積（地面側, 空側の光線の数）。c₁ は太陽中心の格子
    {"transfer", {{"method", "zenith"}, {"rays", {64, 64}}}},
};

/// 検証する組（design.md §5.11）。albedo < 0 なら既定（0.3）、aod550 > 0 なら AOD の上書き、g > 0 なら
/// Cornette-Shanks の g を変える（UI の g の変更と同じ）
struct ShCombo {
    const char* set;
    const char* preset;
    float albedo = -1.0f;
    float aod550 = 0.0f;
    float g = 0.0f;
    bool record = false;  // 記録だけ（判定しない）
};
constexpr ShCombo kShCombos[] = {
    {"Hillaire2020", "base"},
    {"Hillaire2020", "base", -1.0f, 0.0f, 0.6f},
    {"Hillaire2020", "C"},
    {"Hillaire2020", "C+A+B"},
    {"Hillaire2020", "C+A+B+D"},
    {"Hillaire2020", "C+A+B+D", 0.0f},
    {"Hillaire2020", "C+A+B+D", -1.0f, 0.3f, 0.0f, true},
    // 負荷試験（記録）: 光学的に厚く、地表アルベド 1。閉じ方の行列 I − F が特異に近づく
    {"Hillaire2020", "C+A+B+D", 1.0f, 1.0f, 0.0f, true},
};

struct Scenario {
    const char* name;
    float altitudeKm;
    float sunZenithDeg;
};
// research.md §12・V4b（design.md §5.8）と同じシナリオ
constexpr Scenario kSkyScenarios[] = {
    {"day", 0.2f, 45.0f},       {"low_sun", 0.2f, 80.0f},      {"sunset", 0.2f, 89.0f},
    {"day_10km", 10.0f, 45.0f}, {"altitude_3km", 3.0f, 80.0f},
};
constexpr Scenario kV4bScenarios[] = {
    {"day", 0.2f, 45.0f},          {"low_sun", 0.2f, 80.0f},   {"sunset", 0.2f, 89.0f},
    {"earth_shadow", 0.2f, 96.0f}, {"day_10km", 10.0f, 45.0f}, {"altitude_3km", 3.0f, 80.0f},
};

glm::vec3 cameraPosKm(const atmosphere::AtmosphereParams& params, float altitudeKm)
{
    return {0.0f, params.bottomRadiusKm + altitudeKm, 0.0f};
}

/// 正距円筒の方向グリッドに、単位照度あたりの輝度を描いて保存する（Phase48Tests.cpp と同じ）
class PerLuxGrid {
public:
    PerLuxGrid(gfx::Device& device, uint32_t w, uint32_t h)
        : m_device(device),
          m_image(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"ShPerLux"}),
          m_directions(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"ShDirections"})
    {
    }
    uint32_t width() const { return m_image.desc().width; }
    uint32_t height() const { return m_image.desc().height; }

    atmosphere::SkyView view(const glm::vec3& cameraKm) const
    {
        atmosphere::SkyView v;
        v.cameraPosKm = cameraKm;
        v.output = atmosphere::RaymarchOutput::TestPerLux;
        v.directionUav = m_directions.uavIndex();
        v.sunDisk = false;
        return v;
    }

    void render(const std::function<void(ID3D12GraphicsCommandList*, uint32_t)>& record)
    {
        m_device.upload().beginSlot(0);
        m_device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            m_image.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            m_directions.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            record(cmd, m_image.uavIndex());
        });
    }

    void save(const std::filesystem::path& path, nlohmann::json meta)
    {
        meta["units"] = "luminance per illuminance [sr^-1] (= cd/m2 per lux)";
        io::saveExr(path, width(), height(), toFloats(m_image.readback(m_device, 16)), meta);
    }

    void saveDirections(const std::filesystem::path& path)
    {
        io::saveExr(path, width(), height(), toFloats(m_directions.readback(m_device, 16)),
                    {{"units", "unit view direction (planet frame, +Y = zenith)"}});
    }

private:
    gfx::Device& m_device;
    gfx::Texture m_image;
    gfx::Texture m_directions;
};

/// 係数セット・プリセットを当てたパラメータ（見つからなければ nullopt）
std::optional<atmosphere::AtmosphereParams> presetParams(const std::vector<atmosphere::CoefficientSet>& sets,
                                                         const atmosphere::AtmosphereParams& defaults, const char* set,
                                                         const char* preset)
{
    const auto it = std::find_if(sets.begin(), sets.end(), [&](const auto& s) { return s.name == set; });
    if (it == sets.end() || it->findAerosolPreset(preset) == nullptr) {
        gfx::log(std::format("multiscattering_sh: {} / {} not found", set, preset));
        return std::nullopt;
    }
    atmosphere::AtmosphereParams params = defaults;
    params.aerosolPreset = preset;
    params.applyCoefficientSet(*it);
    return params;
}

int runMultiScatteringSh(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    std::vector<atmosphere::CoefficientSet> sets = atmosphere::loadCoefficientSets(kCoefficientPath);
    const atmosphere::MiePhaseTables tables(device, sets);
    const atmosphere::AtmosphereParams defaults = atmosphere::loadDefaultParams(kCoefficientPath);
    // [POINT] 数値検証では LUT を float32 で作る（RGBA16F の格納丸めは別問題として切り離す）
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    const atmosphere::AtmosphereQuality isoQuality;  // 既定（T 500 / MS 40・2 次配置 / Sky-View 30）
    atmosphere::AtmosphereQuality shQuality;
    shQuality.multiScatteringModel = atmosphere::MultiScatteringModel::ShL2Ground;

    for (const char* dir : {"ms_sh", "sky_sh", "v4b_sh"}) {
        std::filesystem::create_directories(outDir / dir);
    }
    constexpr uint32_t mn = atmosphere::AtmosphereRenderer::kMultiScatteringSize;
    constexpr uint32_t shWidth = mn * atmosphere::AtmosphereRenderer::kMultiScatteringShCoefficients;
    gfx::Texture msDebug(device, {mn, mn, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"MsShDebug"});
    PerLuxGrid skyGrid(device, 128, 64);
    auto meta = [](const atmosphere::AtmosphereParams& p, const char* set, const char* preset) {
        return nlohmann::json{{"color_space", display::toString(p.workingSpace)},
                              {"coefficient_set", set},
                              {"aerosol_preset", preset},
                              {"atmosphere", atmosphere::toJson(p)}};
    };

    // V3j-SH: SH の Multi-scattering LUT（検証用 = 入れ子 256 区間・視線 256 区間・等間隔 / 既定 = AtmosphereQuality）
    bool first = true;
    for (const ShCombo& combo : kShCombos) {
        std::optional<atmosphere::AtmosphereParams> found = presetParams(sets, defaults, combo.set, combo.preset);
        if (!found) {
            return 1;
        }
        atmosphere::AtmosphereParams params = *found;
        std::string stem = std::string(combo.set) + "_" + presetFileName(combo.preset);
        if (combo.aod550 > 0.0f) {
            const auto set = std::find_if(sets.begin(), sets.end(), [&](const auto& s) { return s.name == combo.set; });
            params.setAerosolAod550(*set->findAerosolPreset(combo.preset), combo.aod550);
            stem += std::format("_aod{:g}", combo.aod550);
        }
        if (combo.g > 0.0f) {
            params.miePhaseG = combo.g;
            params.miePhaseGModified = true;
            stem += std::format("_g{:g}", combo.g);
        }
        if (combo.albedo >= 0.0f) {
            params.groundAlbedo = glm::vec3(combo.albedo);
            stem += std::format("_albedo{:g}", combo.albedo);
        }
        const nlohmann::json base = withFields(meta(params, combo.set, combo.preset), {{"record_only", combo.record}});
        for (const bool reference : {true, false}) {
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                msDebug.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                renderer.updateTransmittanceLut(cmd, device.upload(), params, 500);
                if (reference) {
                    renderer.updateMultiScatteringLut(cmd, device.upload(), params, 256,
                                                      atmosphere::SunTransmittanceMode::Nested, 256,
                                                      gfx::kInvalidDescriptorIndex, true, shQuality.multiScatteringModel);
                } else {
                    renderer.updateMultiScatteringLut(cmd, device.upload(), params, shQuality.multiScatteringSteps,
                                                      atmosphere::SunTransmittanceMode::Lut, 0, msDebug.uavIndex(),
                                                      shQuality.multiScatteringUniformSteps,
                                                      shQuality.multiScatteringModel);
                }
            });
            io::saveExr(outDir / "ms_sh" / std::format("{}_sh_{}.exr", stem, reference ? "reference" : "default"),
                        shWidth, mn, toFloats(renderer.multiScatteringShLut().readback(device, 16)),
                        withFields(base, {{"multi_scattering_model", modelName(shQuality.multiScatteringModel)},
                                          {"layout", kShLayout},
                                          {"step_count", reference ? 256u : shQuality.multiScatteringSteps},
                                          {"step_distribution", reference ? "uniform" : "quadratic"},
                                          {"sun_transmittance", reference ? "nested256" : "lut500"}}));
        }
        io::saveExr(outDir / "ms_sh" / std::format("{}_sh_debug.exr", stem), mn, mn, toFloats(msDebug.readback(device, 16)),
                    {{"units", "R = r [km], G = mu_sun, B = min over RGB of det(I - F) of the m = 0 block, "
                               "A = F_00 (G channel); default settings"}});
        if (first) {
            io::saveExr(outDir / "ms_sh" / "r_musun.exr", mn, mn, toFloats(msDebug.readback(device, 16)),
                        {{"units", "R = r [km], G = mu_sun"}});
        }
        // 比較用: 等方の LUT（既定の設定）
        device.upload().beginSlot(0);
        device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            renderer.updateMultiScatteringLut(cmd, device.upload(), params, isoQuality.multiScatteringSteps,
                                              atmosphere::SunTransmittanceMode::Lut, 0, gfx::kInvalidDescriptorIndex,
                                              isoQuality.multiScatteringUniformSteps, isoQuality.multiScatteringModel);
        });
        io::saveExr(outDir / "ms_sh" / std::format("{}_iso_default.exr", stem), mn, mn,
                    toFloats(renderer.multiScatteringLut().readback(device, 16)),
                    withFields(base, {{"multi_scattering_model", "isotropic"},
                                      {"units", "Psi_ms per illuminance [sr^-1]"},
                                      {"step_count", isoQuality.multiScatteringSteps}}));
        gfx::log(std::format("multiscattering_sh: ms {}", stem));
        first = false;
    }

    // V3k-SH: 全天の LUT 描画とレイマーチ（SH）、比較用に等方の LUT 描画（地表アルベドは既定の 0.3）
    bool firstSky = true;
    for (const char* preset : {"C+A+B+D", "C+A+B"}) {
        std::optional<atmosphere::AtmosphereParams> params = presetParams(sets, defaults, "Hillaire2020", preset);
        if (!params) {
            return 1;
        }
        const std::string stem = "Hillaire2020_" + presetFileName(preset);
        for (const Scenario& s : kSkyScenarios) {
            atmosphere::AtmosphereParams p = *params;
            p.sunZenithDeg = s.sunZenithDeg;
            const glm::vec3 camera = cameraPosKm(p, s.altitudeKm);
            const nlohmann::json m = withFields(meta(p, "Hillaire2020", preset),
                                                {{"scenario", s.name}, {"camera_altitude_km", s.altitudeKm}});
            for (const bool sh : {true, false}) {
                const atmosphere::AtmosphereQuality& quality = sh ? shQuality : isoQuality;
                skyGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                    renderer.updateLuts(cmd, device.upload(), p, quality, camera);
                    renderer.renderSkyFromLut(cmd, device.upload(), p, skyGrid.view(camera), uav, skyGrid.width(),
                                              skyGrid.height());
                });
                skyGrid.save(outDir / "sky_sh" / std::format("{}_{}_{}_lut.exr", stem, s.name, sh ? "sh" : "iso"),
                             withFields(m, {{"path", "sky_view_lut"},
                                            {"multi_scattering_model", modelName(quality.multiScatteringModel)}}));
            }
            // SH のレイマーチ（LUT は直前に作った SH のものを使うので、もう一度 SH の LUT にする）
            atmosphere::RaymarchSettings settings;
            settings.primarySteps = 2048;
            settings.sunMode = atmosphere::SunTransmittanceMode::Lut;
            settings.multipleScattering = true;
            skyGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                renderer.updateLuts(cmd, device.upload(), p, shQuality, camera);
                renderer.renderSkyRaymarch(cmd, device.upload(), p, skyGrid.view(camera), settings, uav,
                                           skyGrid.width(), skyGrid.height());
            });
            skyGrid.save(outDir / "sky_sh" / std::format("{}_{}_sh_raymarch.exr", stem, s.name),
                         withFields(m, {{"path", "raymarch"},
                                        {"primary_steps", settings.primarySteps},
                                        {"multi_scattering_model", modelName(shQuality.multiScatteringModel)}}));
            if (firstSky) {
                skyGrid.saveDirections(outDir / "sky_sh" / "directions.exr");
                firstSky = false;
            }
        }
        gfx::log(std::format("multiscattering_sh: sky {}", stem));
    }

    // V4b-2: 地表アルベド 0 の LUT 描画（等方 / SH）。パストレースのダンプ（dumps/v4b）と比べる
    for (const char* preset : {"C", "C+A+B", "C+A+B+D"}) {
        std::optional<atmosphere::AtmosphereParams> params = presetParams(sets, defaults, "Hillaire2020", preset);
        if (!params) {
            return 1;
        }
        for (const Scenario& s : kV4bScenarios) {
            atmosphere::AtmosphereParams p = *params;
            p.sunZenithDeg = s.sunZenithDeg;
            p.groundAlbedo = glm::vec3(0.0f);
            const glm::vec3 camera = cameraPosKm(p, s.altitudeKm);
            for (const bool sh : {true, false}) {
                const atmosphere::AtmosphereQuality& quality = sh ? shQuality : isoQuality;
                skyGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                    renderer.updateLuts(cmd, device.upload(), p, quality, camera);
                    renderer.renderSkyFromLut(cmd, device.upload(), p, skyGrid.view(camera), uav, skyGrid.width(),
                                              skyGrid.height());
                });
                skyGrid.save(outDir / "v4b_sh" /
                                 std::format("{}_{}_lut_{}.exr", presetFileName(preset), s.name, sh ? "sh" : "iso"),
                             withFields(meta(p, "Hillaire2020", preset),
                                        {{"scenario", s.name},
                                         {"camera_altitude_km", s.altitudeKm},
                                         {"path", "sky_view_lut"},
                                         {"multi_scattering_model", modelName(quality.multiScatteringModel)}}));
            }
        }
        gfx::log(std::format("multiscattering_sh: v4b {}", preset));
    }
    return 0;
}

} // namespace

int runPhase411Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                    const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    if (name == "multiscattering_sh") {
        return runMultiScatteringSh(device, pipelines, outDir);
    }
    gfx::log("Unknown test: " + std::string(name));
    return 2;
}

} // namespace batch
