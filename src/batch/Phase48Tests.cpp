#include "batch/Phase48Tests.h"

#include "atmosphere/AtmosphereRenderer.h"
#include "atmosphere/MiePhaseTables.h"
#include "io/ExrWriter.h"
#include "scene/Camera.h"

#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>

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

/// 検証する係数セットとプリセット（1 成分）。A・B を含まない C・C+A は色空間で Mie が変わらないので Hillaire2020 だけ
struct Combo {
    const char* set;
    const char* preset;
};
constexpr Combo kCombos[] = {
    {"Hillaire2020", "C"},      {"Hillaire2020", "C+A"},     {"Hillaire2020", "C+B"},
    {"Hillaire2020", "C+A+B"},  {"SpectralFit709", "C+B"},   {"SpectralFit709", "C+A+B"},
    {"SpectralFit2020", "C+B"}, {"SpectralFit2020", "C+A+B"},
};

/// ファイル名: "+" を "_" に（"C+A+B" → "C_A_B"）
std::string presetFileName(std::string name)
{
    for (char& c : name) {
        if (c == '+') {
            c = '_';
        }
    }
    return name;
}

struct Scenario {
    const char* name;
    float altitudeKm;
    float sunZenithDeg;
};
// research.md §12・V4b（design.md §5.8）と同じシナリオ
constexpr Scenario kSkyScenarios[] = {
    {"day", 0.2f, 45.0f},      {"low_sun", 0.2f, 80.0f},      {"sunset", 0.2f, 89.0f},
    {"day_10km", 10.0f, 45.0f}, {"altitude_3km", 3.0f, 80.0f},
};
constexpr Scenario kV4bScenarios[] = {
    {"day", 0.2f, 45.0f},           {"low_sun", 0.2f, 80.0f},   {"sunset", 0.2f, 89.0f},
    {"earth_shadow", 0.2f, 96.0f}, {"day_10km", 10.0f, 45.0f}, {"altitude_3km", 3.0f, 80.0f},
};

/// 正距円筒の方向グリッドに、単位照度あたりの輝度を描いて保存する
class PerLuxGrid {
public:
    PerLuxGrid(gfx::Device& device, uint32_t w, uint32_t h)
        : m_device(device),
          m_image(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"AerosolPerLux"}),
          m_directions(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"AerosolDirections"})
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

/// サイドカーの JSON に項目を足す
nlohmann::json withFields(nlohmann::json base, const nlohmann::json& extra)
{
    base.update(extra);
    return base;
}

glm::vec3 cameraPosKm(const atmosphere::AtmosphereParams& params, float altitudeKm)
{
    return {0.0f, params.bottomRadiusKm + altitudeKm, 0.0f};
}

int runAerosolPresets(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    std::vector<atmosphere::CoefficientSet> sets = atmosphere::loadCoefficientSets(kCoefficientPath);
    const atmosphere::MiePhaseTables tables(device, sets);
    const atmosphere::AtmosphereParams defaults = atmosphere::loadDefaultParams(kCoefficientPath);
    // [POINT] 数値検証では LUT を float32 で作る（RGBA16F の格納丸めは別問題として切り離す）
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    const atmosphere::AtmosphereQuality quality;  // 既定（T 500 / MS 40・2 次配置 / Sky-View 30、多重散乱あり）

    for (const char* dir : {"transmittance", "single", "ms", "sky", "closeup", "v4b"}) {
        std::filesystem::create_directories(outDir / dir);
    }
    constexpr uint32_t tw = atmosphere::AtmosphereRenderer::kTransmittanceWidth;
    constexpr uint32_t th = atmosphere::AtmosphereRenderer::kTransmittanceHeight;
    constexpr uint32_t mn = atmosphere::AtmosphereRenderer::kMultiScatteringSize;
    gfx::Texture transmittanceRMu(device, {tw, th, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"TRMu"});
    gfx::Texture msRMu(device, {mn, mn, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"MSRMu"});
    PerLuxGrid singleGrid(device, 32, 16);
    PerLuxGrid skyGrid(device, 128, 64);
    constexpr uint32_t kCloseupSize = 128;
    constexpr float kCloseupFovDeg = 12.0f;  // 太陽の周り ±6°
    gfx::Texture closeup(device, {kCloseupSize, kCloseupSize, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false,
                                  L"AerosolCloseup"});

    bool first = true;
    for (const Combo& combo : kCombos) {
        const auto set = std::find_if(sets.begin(), sets.end(), [&](const auto& s) { return s.name == combo.set; });
        if (set == sets.end() || set->findAerosolPreset(combo.preset) == nullptr) {
            gfx::log(std::format("aerosol_presets: {} / {} not found", combo.set, combo.preset));
            return 1;
        }
        atmosphere::AtmosphereParams params = defaults;
        params.aerosolPreset = combo.preset;
        params.applyCoefficientSet(*set);
        const std::string stem = std::string(combo.set) + "_" + presetFileName(combo.preset);
        auto meta = [&](const atmosphere::AtmosphereParams& p) {
            return nlohmann::json{{"color_space", display::toString(p.workingSpace)},
                                  {"coefficient_set", combo.set},
                                  {"aerosol_preset", combo.preset},
                                  {"atmosphere", atmosphere::toJson(p)}};
        };

        // V3i: Transmittance LUT（500 区間）
        device.upload().beginSlot(0);
        device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            transmittanceRMu.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            renderer.updateTransmittanceLut(cmd, device.upload(), params, 500, transmittanceRMu.uavIndex());
        });
        io::saveExr(outDir / "transmittance" / (stem + ".exr"), tw, th,
                    toFloats(renderer.transmittanceLut().readback(device, 16)),
                    withFields(meta(params), {{"units", "transmittance (unitless)"}, {"step_count", 500}}));
        if (first) {
            io::saveExr(outDir / "transmittance" / "r_mu.exr", tw, th, toFloats(transmittanceRMu.readback(device, 16)),
                        {{"units", "R = r [km] (planet center), G = mu (cos zenith)"}});
        }
        // [POINT] AOD 0.1 では地平線付近の光学的厚さが今の大気の数倍になり、台形則の誤差もそれに比例して増える。
        //         区間数を 8 倍にした結果も書き、誤差が区間数で減る（実装の誤りではない）ことを確かめる
        device.upload().beginSlot(0);
        device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            renderer.updateTransmittanceLut(cmd, device.upload(), params, 4000);
        });
        io::saveExr(outDir / "transmittance" / (stem + "_steps4000.exr"), tw, th,
                    toFloats(renderer.transmittanceLut().readback(device, 16)),
                    withFields(meta(params), {{"units", "transmittance (unitless)"}, {"step_count", 4000}}));

        // V3i・V9: 単散乱（太陽方向は入れ子レイマーチ、Phase 2 の V2 と同じ設定）
        for (const float altitudeKm : {0.2f, 10.0f}) {
            for (const float sunZenithDeg : {30.0f, 60.0f, 85.0f}) {
                atmosphere::AtmosphereParams p = params;
                p.sunZenithDeg = sunZenithDeg;
                atmosphere::RaymarchSettings settings;
                settings.primarySteps = 2048;
                settings.sunMode = atmosphere::SunTransmittanceMode::Nested;
                settings.nestedSteps = 256;
                settings.multipleScattering = false;
                const glm::vec3 camera = cameraPosKm(p, altitudeKm);
                singleGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                    renderer.renderSkyRaymarch(cmd, device.upload(), p, singleGrid.view(camera), settings, uav,
                                               singleGrid.width(), singleGrid.height());
                });
                singleGrid.save(outDir / "single" / std::format("{}_h{:g}_sz{:g}.exr", stem, altitudeKm, sunZenithDeg),
                                withFields(meta(p), {{"camera_altitude_km", altitudeKm},
                                                         {"primary_steps", settings.primarySteps},
                                                         {"nested_steps", settings.nestedSteps}}));
                // 太陽方向の区間数を 4 倍にした結果（収束の確認）
                settings.nestedSteps = 1024;
                singleGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                    renderer.renderSkyRaymarch(cmd, device.upload(), p, singleGrid.view(camera), settings, uav,
                                               singleGrid.width(), singleGrid.height());
                });
                singleGrid.save(
                    outDir / "single" / std::format("{}_h{:g}_sz{:g}_nested1024.exr", stem, altitudeKm, sunZenithDeg),
                    withFields(meta(p), {{"camera_altitude_km", altitudeKm},
                                         {"primary_steps", settings.primarySteps},
                                         {"nested_steps", settings.nestedSteps}}));
                if (first) {
                    singleGrid.saveDirections(outDir / "single" / "directions.exr");
                }
            }
        }

        // V3j: Multi-scattering LUT（検証用 = 入れ子 256 区間・視線 256 区間・等間隔 / 既定 = AtmosphereQuality の
        //      区間数・配置、Transmittance LUT）
        for (const bool reference : {true, false}) {
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                msRMu.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                renderer.updateTransmittanceLut(cmd, device.upload(), params, 500);
                if (reference) {
                    renderer.updateMultiScatteringLut(cmd, device.upload(), params, 256,
                                                      atmosphere::SunTransmittanceMode::Nested, 256, msRMu.uavIndex());
                } else {
                    renderer.updateMultiScatteringLut(cmd, device.upload(), params, quality.multiScatteringSteps,
                                                      atmosphere::SunTransmittanceMode::Lut, 0, msRMu.uavIndex(),
                                                      quality.multiScatteringUniformSteps);
                }
            });
            io::saveExr(outDir / "ms" / std::format("{}_{}.exr", stem, reference ? "reference" : "default"), mn, mn,
                        toFloats(renderer.multiScatteringLut().readback(device, 16)),
                        withFields(meta(params), {{"units", "Psi_ms per illuminance [sr^-1]"},
                                                      {"step_count", reference ? 256u : quality.multiScatteringSteps},
                                                      {"sun_transmittance", reference ? "nested" : "lut500"}}));
        }
        if (first) {
            io::saveExr(outDir / "ms" / "r_musun.exr", mn, mn, toFloats(msRMu.readback(device, 16)),
                        {{"units", "R = r [km], G = mu_sun"}});
        }
        // 記録: Multi-scattering LUT の区間数と配置（Transmittance LUT を引く実用の設定）。AOD 0.1 で以前の既定の 20 区間
        //       （等間隔）が足りるかを見る（design.md §14 Phase 4.8）。Hillaire2020 の C・C+A+B だけ
        if (std::string_view(combo.set) == "Hillaire2020" &&
            (std::string_view(combo.preset) == "C" || std::string_view(combo.preset) == "C+A+B")) {
            std::filesystem::create_directories(outDir / "ms_steps");
            for (const uint32_t steps : {20u, 40u, 64u, 128u}) {
                for (const bool uniform : {true, false}) {
                    device.upload().beginSlot(0);
                    device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                        renderer.updateMultiScatteringLut(cmd, device.upload(), params, steps,
                                                          atmosphere::SunTransmittanceMode::Lut, 0,
                                                          gfx::kInvalidDescriptorIndex, uniform);
                    });
                    io::saveExr(outDir / "ms_steps" /
                                    std::format("{}_lut{}_{}.exr", stem, steps, uniform ? "uniform" : "quadratic"),
                                mn, mn, toFloats(renderer.multiScatteringLut().readback(device, 16)),
                                withFields(meta(params), {{"units", "Psi_ms per illuminance [sr^-1]"},
                                                          {"step_count", steps},
                                                          {"step_distribution", uniform ? "uniform" : "quadratic"}}));
                }
            }
        }

        // V3k: 全天の LUT 描画 / レイマーチ（同じ Transmittance・Multi-scattering LUT、地表アルベドは既定の 0.3）
        for (const Scenario& s : kSkyScenarios) {
            atmosphere::AtmosphereParams p = params;
            p.sunZenithDeg = s.sunZenithDeg;
            const glm::vec3 camera = cameraPosKm(p, s.altitudeKm);
            nlohmann::json m = withFields(meta(p), {{"scenario", s.name}, {"camera_altitude_km", s.altitudeKm}});
            skyGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                renderer.updateLuts(cmd, device.upload(), p, quality, camera);
                renderer.renderSkyFromLut(cmd, device.upload(), p, skyGrid.view(camera), uav, skyGrid.width(),
                                          skyGrid.height());
            });
            m["path"] = "sky_view_lut";
            skyGrid.save(outDir / "sky" / std::format("{}_{}_lut.exr", stem, s.name), m);
            atmosphere::RaymarchSettings settings;
            settings.primarySteps = 2048;
            settings.sunMode = atmosphere::SunTransmittanceMode::Lut;
            settings.multipleScattering = true;
            skyGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                renderer.renderSkyRaymarch(cmd, device.upload(), p, skyGrid.view(camera), settings, uav,
                                           skyGrid.width(), skyGrid.height());
            });
            m["path"] = "raymarch";
            m["primary_steps"] = settings.primarySteps;
            skyGrid.save(outDir / "sky" / std::format("{}_{}_raymarch.exr", stem, s.name), m);
            if (first) {
                skyGrid.saveDirections(outDir / "sky" / "directions.exr");
            }

            // V3b: 太陽の周り ±6°（低い太陽・夕方）。カメラを太陽に向ける（方位 0° = +X、yaw 90° で +X）
            if (std::string_view(s.name) != "low_sun" && std::string_view(s.name) != "sunset") {
                continue;
            }
            scene::Camera cam;
            cam.yawRad = glm::radians(90.0f);
            cam.pitchRad = glm::radians(90.0f - s.sunZenithDeg);
            cam.verticalFovRad = glm::radians(kCloseupFovDeg);
            atmosphere::SkyView view;
            view.invViewProjRot = glm::inverse(cam.projection(1.0f) * cam.viewRotation());
            view.cameraPosKm = camera;
            view.exposureScale = 1.0f / p.sunIlluminanceLux;  // fb = L / E_sun = 単位照度あたりの輝度
            view.sunDisk = false;
            for (const bool lut : {true, false}) {
                device.upload().beginSlot(0);
                device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                    closeup.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    renderer.updateLuts(cmd, device.upload(), p, quality, camera);
                    if (lut) {
                        renderer.renderSkyFromLut(cmd, device.upload(), p, view, closeup.uavIndex(), kCloseupSize,
                                                  kCloseupSize);
                    } else {
                        renderer.renderSkyRaymarch(cmd, device.upload(), p, view, settings, closeup.uavIndex(),
                                                   kCloseupSize, kCloseupSize);
                    }
                });
                nlohmann::json cm = withFields(meta(p), {{"scenario", s.name},
                                                             {"camera_altitude_km", s.altitudeKm},
                                                             {"camera_yaw_deg", 90.0f},
                                                             {"camera_pitch_deg", 90.0f - s.sunZenithDeg},
                                                             {"vertical_fov_deg", kCloseupFovDeg},
                                                             {"path", lut ? "sky_view_lut" : "raymarch"},
                                                             {"units", "luminance per illuminance [sr^-1]"}});
                io::saveExr(outDir / "closeup" / std::format("{}_{}_{}.exr", stem, s.name, lut ? "lut" : "raymarch"),
                            kCloseupSize, kCloseupSize, toFloats(closeup.readback(device, 16)), cm);
            }
        }

        // V4b: 地表アルベド 0 の LUT 描画（Phase 4.7 のパストレースのダンプと比べる。Hillaire2020 の C と C+A+B）
        const std::string_view preset = combo.preset;
        if (std::string_view(combo.set) == "Hillaire2020" && (preset == "C" || preset == "C+A+B")) {
            for (const Scenario& s : kV4bScenarios) {
                atmosphere::AtmosphereParams p = params;
                p.sunZenithDeg = s.sunZenithDeg;
                p.groundAlbedo = glm::vec3(0.0f);
                const glm::vec3 camera = cameraPosKm(p, s.altitudeKm);
                skyGrid.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                    renderer.updateLuts(cmd, device.upload(), p, quality, camera);
                    renderer.renderSkyFromLut(cmd, device.upload(), p, skyGrid.view(camera), uav, skyGrid.width(),
                                              skyGrid.height());
                });
                skyGrid.save(outDir / "v4b" / std::format("{}_{}_lut.exr", presetFileName(combo.preset), s.name),
                             withFields(meta(p), {{"scenario", s.name}, {"camera_altitude_km", s.altitudeKm},
                                                      {"path", "sky_view_lut"}}));
            }
        }
        gfx::log(std::format("aerosol_presets: {}", stem));
        first = false;
    }
    return 0;
}

} // namespace

int runPhase48Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                   const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    if (name == "aerosol_presets") {
        return runAerosolPresets(device, pipelines, outDir);
    }
    gfx::log("Unknown test: " + std::string(name));
    return 2;
}

} // namespace batch
