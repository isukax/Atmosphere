#include "batch/Phase3Tests.h"

#include "atmosphere/AtmosphereRenderer.h"
#include "atmosphere/PathTracer.h"
#include "io/ExrWriter.h"
#include "scene/Camera.h"

#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>

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

/// 検証シナリオ: カメラ高度と太陽天頂角（論文 Fig. 9 の昼・夕方・地球の影に相当）
struct Scenario {
    const char* name;
    float altitudeKm;
    float sunZenithDeg;
};
constexpr Scenario kScenarios[] = {
    {"day", 0.2f, 45.0f},
    {"sunset", 0.2f, 89.0f},
    {"earth_shadow", 0.2f, 96.0f},
    {"day_10km", 10.0f, 45.0f},
};

/// 正距円筒の方向グリッド（全球）
constexpr uint32_t kGridWidth = 128;
constexpr uint32_t kGridHeight = 64;

/// [POINT] 多重散乱の比較（V3 / V4）は地表アルベド 0 で行う（論文の RMSE 測定と同じ条件。
///         パストレーサーとの比較で地表の反射を切り離すため）。
atmosphere::AtmosphereParams scenarioParams(const Scenario& s)
{
    atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
    params.sunZenithDeg = s.sunZenithDeg;
    params.groundAlbedo = glm::vec3(0.0f);
    return params;
}

glm::vec3 cameraPosKm(const atmosphere::AtmosphereParams& params, float altitudeKm)
{
    return {0.0f, params.bottomRadiusKm + altitudeKm, 0.0f};
}

/// 単位照度あたりの輝度の画像（float32）を描いて保存する
class PerLuxTarget {
public:
    PerLuxTarget(gfx::Device& device, uint32_t w, uint32_t h)
        : m_device(device),
          m_image(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"PerLux"}),
          m_directions(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"PerLuxDirections"})
    {
    }

    atmosphere::SkyView view(const glm::vec3& cameraKm) const
    {
        atmosphere::SkyView v;
        v.cameraPosKm = cameraKm;
        v.output = atmosphere::RaymarchOutput::TestPerLux;
        v.directionUav = m_directions.uavIndex();
        v.sunDisk = false;  // 比較は空の散乱光だけ（太陽ディスクは解析的に足す同じ式なので比べる意味がない）
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
        io::saveExr(path, m_image.desc().width, m_image.desc().height, toFloats(m_image.readback(m_device, 16)), meta);
    }

    void saveDirections(const std::filesystem::path& path)
    {
        io::saveExr(path, m_directions.desc().width, m_directions.desc().height,
                    toFloats(m_directions.readback(m_device, 16)),
                    {{"units", "unit view direction (planet frame, +Y = zenith)"}});
    }

private:
    gfx::Device& m_device;
    gfx::Texture m_image;
    gfx::Texture m_directions;
};

nlohmann::json baseMeta(const atmosphere::AtmosphereParams& params, const Scenario& s)
{
    return {{"color_space", display::toString(params.workingSpace)},
            {"scenario", s.name},
            {"camera_altitude_km", s.altitudeKm},
            {"atmosphere", atmosphere::toJson(params)}};
}

// ---------------------------------------------------------------------------

int runMultiScattering(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
    params.groundAlbedo = glm::vec3(0.3f);  // 地表反射の項も検証する（論文本文の値）
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    constexpr uint32_t n = atmosphere::AtmosphereRenderer::kMultiScatteringSize;
    gfx::Texture debug(device, {n, n, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"MSDebug"});

    struct Variant {
        const char* name;
        uint32_t steps;
        atmosphere::SunTransmittanceMode mode;
        uint32_t nestedSteps;
    };
    const Variant variants[] = {
        {"reference", 256, atmosphere::SunTransmittanceMode::Nested, 256},  // 検証対象（LUT なし・多ステップ）
        {"default", 20, atmosphere::SunTransmittanceMode::Lut, 0},          // 実用設定（参考値）
    };
    for (const Variant& v : variants) {
        device.upload().beginSlot(0);
        device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            debug.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            renderer.updateTransmittanceLut(cmd, device.upload(), params, 500);
            renderer.updateMultiScatteringLut(cmd, device.upload(), params, v.steps, v.mode, v.nestedSteps,
                                              debug.uavIndex());
        });
        io::saveExr(outDir / std::format("ms_{}.exr", v.name), n, n,
                    toFloats(renderer.multiScatteringLut().readback(device, 16)),
                    {{"units", "Psi_ms per illuminance [sr^-1]"},
                     {"color_space", display::toString(params.workingSpace)},
                     {"step_count", v.steps},
                     {"sun_transmittance", v.mode == atmosphere::SunTransmittanceMode::Nested ? "nested" : "lut500"},
                     {"nested_steps", v.nestedSteps},
                     {"step_distribution", "uniform"},
                     {"atmosphere", atmosphere::toJson(params)}});
    }
    io::saveExr(outDir / "ms_r_musun.exr", n, n, toFloats(debug.readback(device, 16)),
                {{"units", "R = r [km], G = mu_sun"}, {"atmosphere", atmosphere::toJson(params)}});

    // 区間の配置の比較（2 次 vs 等間隔）: 実用と同じ Transmittance LUT（500 区間）で、10 / 20 / 40 区間。
    // 既定は等間隔（参照実装と同じ）。2 次配置との差を記録として残す（design.md §14 Phase 3）。
    for (const uint32_t steps : {10u, 20u, 40u}) {
        for (const bool uniform : {false, true}) {
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                debug.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                renderer.updateMultiScatteringLut(cmd, device.upload(), params, steps,
                                                  atmosphere::SunTransmittanceMode::Lut, 0, debug.uavIndex(), uniform);
            });
            io::saveExr(outDir / std::format("ms_lut{}_{}.exr", steps, uniform ? "uniform" : "quadratic"), n, n,
                        toFloats(renderer.multiScatteringLut().readback(device, 16)),
                        {{"units", "Psi_ms per illuminance [sr^-1]"},
                         {"step_count", steps},
                         {"step_distribution", uniform ? "uniform" : "quadratic"},
                         {"sun_transmittance", "lut500"},
                         {"atmosphere", atmosphere::toJson(params)}});
        }
    }
    gfx::log("multiscattering: wrote LUTs");
    return 0;
}

int runSkyLut(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    PerLuxTarget target(device, kGridWidth, kGridHeight);
    const atmosphere::AtmosphereQuality quality;  // 既定（T 500 / MS 20 / Sky-View 30、多重散乱あり）
    for (const Scenario& s : kScenarios) {
        const atmosphere::AtmosphereParams params = scenarioParams(s);
        const glm::vec3 camera = cameraPosKm(params, s.altitudeKm);
        nlohmann::json meta = baseMeta(params, s);

        // 実用パス: LUT 3 枚 → Sky-View LUT を引いて描く
        target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
            renderer.updateLuts(cmd, device.upload(), params, quality, camera);
            renderer.renderSkyFromLut(cmd, device.upload(), params, target.view(camera), uav, kGridWidth, kGridHeight);
        });
        meta["path"] = "sky_view_lut";
        meta["sky_view_steps"] = quality.skyViewSteps;
        target.save(outDir / std::format("{}_lut.exr", s.name), meta);

        // リファレンス: 同じ Transmittance / Multi-scattering LUT で、ピクセルごとに 2048 区間
        atmosphere::RaymarchSettings settings;
        settings.primarySteps = 2048;
        settings.sunMode = atmosphere::SunTransmittanceMode::Lut;
        settings.multipleScattering = true;
        target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
            renderer.renderSkyRaymarch(cmd, device.upload(), params, target.view(camera), settings, uav, kGridWidth,
                                       kGridHeight);
        });
        meta["path"] = "raymarch";
        meta["primary_steps"] = settings.primarySteps;
        target.save(outDir / std::format("{}_raymarch.exr", s.name), meta);
        target.saveDirections(outDir / std::format("{}_directions.exr", s.name));
        gfx::log(std::string("sky_lut: ") + s.name);
    }
    return 0;
}

int runPathTrace(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    atmosphere::PathTracer tracer(device, pipelines);
    tracer.resize(kGridWidth, kGridHeight);
    PerLuxTarget target(device, kGridWidth, kGridHeight);
    const atmosphere::AtmosphereQuality quality;

    // [POINT] パストレースの設定: 散乱は最大 32 回（地球の大気では 5 回程度でほぼ収束する）、
    //         1 ディスパッチで 64 サンプル × 1024 回 = チャンネルあたり 65,536 サンプル。
    //         光学的厚さが小さく大半の経路が衝突せずに抜けるので分散が大きい（2048 サンプルでは
    //         ピクセルあたり 5〜13% のノイズがあり、近似の誤差と区別できなかった）。
    //         散乱 1 回（単散乱）の確認は 16,384 サンプル。
    constexpr uint32_t kSamplesPerDispatch = 64;
    auto trace = [&](const atmosphere::AtmosphereParams& params, const glm::vec3& camera, uint32_t maxDepth,
                     uint32_t dispatches, const std::filesystem::path& stem, nlohmann::json meta) {
        tracer.reset();
        for (uint32_t i = 0; i < dispatches; ++i) {
            target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                tracer.render(cmd, device.upload(), renderer, params, target.view(camera), maxDepth,
                              kSamplesPerDispatch, uav);
            });
        }
        meta["path"] = "path_trace";
        meta["max_depth"] = maxDepth;
        meta["samples_per_channel"] = tracer.samplesPerChannel();
        target.save(stem.string() + ".exr", meta);
        // 偶数・奇数フレームの累積（rgb = 和、a = サンプル数）。2 つの平均の差からノイズを見積もる。
        io::saveExr(stem.string() + "_accA.exr", kGridWidth, kGridHeight,
                    toFloats(tracer.accumulationA().readback(device, 16)), {{"units", "sum (rgb), count (a)"}});
        io::saveExr(stem.string() + "_accB.exr", kGridWidth, kGridHeight,
                    toFloats(tracer.accumulationB().readback(device, 16)), {{"units", "sum (rgb), count (a)"}});
    };

    for (const Scenario& s : kScenarios) {
        const atmosphere::AtmosphereParams params = scenarioParams(s);
        const glm::vec3 camera = cameraPosKm(params, s.altitudeKm);
        const nlohmann::json meta = baseMeta(params, s);

        // LUT 描画（実用パス）
        target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
            renderer.updateLuts(cmd, device.upload(), params, quality, camera);
            renderer.renderSkyFromLut(cmd, device.upload(), params, target.view(camera), uav, kGridWidth, kGridHeight);
        });
        target.save(outDir / std::format("{}_lut.exr", s.name), meta);

        // レイマーチ（多重散乱は同じ近似、Sky-View LUT なし）
        atmosphere::RaymarchSettings settings;
        settings.primarySteps = 2048;
        settings.multipleScattering = true;
        target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
            renderer.renderSkyRaymarch(cmd, device.upload(), params, target.view(camera), settings, uav, kGridWidth,
                                       kGridHeight);
        });
        target.save(outDir / std::format("{}_raymarch.exr", s.name), meta);

        // 単散乱（入れ子レイマーチ、Phase 2 で検証済み）とパストレース（散乱 1 回）
        settings.multipleScattering = false;
        settings.sunMode = atmosphere::SunTransmittanceMode::Nested;
        settings.nestedSteps = 256;
        target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
            renderer.renderSkyRaymarch(cmd, device.upload(), params, target.view(camera), settings, uav, kGridWidth,
                                       kGridHeight);
        });
        target.save(outDir / std::format("{}_single.exr", s.name), meta);
        trace(params, camera, 1, 256, outDir / std::format("{}_pt_single", s.name), meta);

        // パストレース（多重散乱の真値）
        trace(params, camera, 32, 1024, outDir / std::format("{}_pt", s.name), meta);
        gfx::log(std::string("pathtrace: ") + s.name);
    }
    return 0;
}

int runSpectral(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    constexpr uint32_t w = 16;
    constexpr uint32_t h = 8;
    PerLuxTarget target(device, w, h);
    const std::vector<atmosphere::CoefficientSet> sets = atmosphere::loadCoefficientSets(kCoefficientPath);
    for (const atmosphere::CoefficientSet& set : sets) {
        for (const float sunZenithDeg : {30.0f, 60.0f, 85.0f}) {
            atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
            params.applyCoefficientSet(set);
            params.sunZenithDeg = sunZenithDeg;
            const glm::vec3 camera = cameraPosKm(params, 0.2f);
            atmosphere::RaymarchSettings settings;
            settings.primarySteps = 2048;
            settings.sunMode = atmosphere::SunTransmittanceMode::Nested;
            settings.nestedSteps = 256;
            settings.multipleScattering = false;  // V9 は透過率 + 単散乱で比べる（分光の多重散乱は扱わない）
            target.render([&](ID3D12GraphicsCommandList* cmd, uint32_t uav) {
                renderer.renderSkyRaymarch(cmd, device.upload(), params, target.view(camera), settings, uav, w, h);
            });
            const std::string stem = std::format("{}_sz{:g}", set.name, sunZenithDeg);
            target.save(outDir / (stem + ".exr"),
                        {{"color_space", display::toString(params.workingSpace)},
                         {"coefficient_set", set.name},
                         {"camera_altitude_km", 0.2f},
                         {"atmosphere", atmosphere::toJson(params)}});
            target.saveDirections(outDir / (stem + "_directions.exr"));
        }
        gfx::log("spectral: " + set.name);
    }
    return 0;
}

/// V3b: 太陽の周りを狭い画角（6°）で、Sky-View LUT 描画とレイマーチで描き比べる。
/// Mie の g と太陽天頂角を振り、LUT の解像度・パラメタ化で光輪の形がどう崩れるかを見る。
/// 出力は単位照度あたりの輝度 [sr⁻¹]（露出を 1/E_sun にして fb = L/E にしている）。
int runSunCloseup(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    constexpr uint32_t n = 256;
    constexpr float kFovDeg = 6.0f;
    gfx::Texture image(device, {n, n, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"SunCloseup"});
    const atmosphere::AtmosphereQuality quality;

    struct Case {
        float g;
        float sunZenithDeg;
    };
    std::vector<Case> cases = {{0.8f, 35.0f}, {0.99f, 35.0f}};
    for (int i = 0; i <= 8; ++i) {
        cases.push_back({0.99f, 34.0f + 0.25f * static_cast<float>(i)});  // 太陽の位置とテクセルの関係を見る
    }
    for (const Case& c : cases) {
        atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
        params.miePhaseG = c.g;
        params.sunZenithDeg = c.sunZenithDeg;
        params.sunAzimuthDeg = 0.0f;

        // カメラを太陽に向ける（方位 0° = +X、yaw 90° で +X を向く）
        scene::Camera camera;
        camera.yawRad = glm::radians(90.0f);
        camera.pitchRad = glm::radians(90.0f - c.sunZenithDeg);
        camera.verticalFovRad = glm::radians(kFovDeg);
        atmosphere::SkyView view;
        view.invViewProjRot = glm::inverse(camera.projection(1.0f) * camera.viewRotation());
        view.cameraPosKm = glm::vec3(0.0f, params.bottomRadiusKm + 0.2f, 0.0f);
        view.exposureScale = 1.0f / params.sunIlluminanceLux;  // fb = L / E_sun = 単位照度あたりの輝度
        view.sunDisk = false;

        for (const bool lut : {true, false}) {
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                image.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                renderer.updateLuts(cmd, device.upload(), params, quality, view.cameraPosKm);
                if (lut) {
                    renderer.renderSkyFromLut(cmd, device.upload(), params, view, image.uavIndex(), n, n);
                } else {
                    atmosphere::RaymarchSettings settings;
                    settings.primarySteps = 2048;
                    settings.multipleScattering = true;
                    renderer.renderSkyRaymarch(cmd, device.upload(), params, view, settings, image.uavIndex(), n, n);
                }
            });
            io::saveExr(outDir / std::format("closeup_g{:g}_sz{:g}_{}.exr", c.g, c.sunZenithDeg, lut ? "lut" : "raymarch"),
                        n, n, toFloats(image.readback(device, 16)),
                        {{"units", "luminance per illuminance [sr^-1]"},
                         {"camera_yaw_deg", 90.0f},
                         {"camera_pitch_deg", 90.0f - c.sunZenithDeg},
                         {"vertical_fov_deg", kFovDeg},
                         {"camera_altitude_km", 0.2f},
                         {"path", lut ? "sky_view_lut" : "raymarch"},
                         {"atmosphere", atmosphere::toJson(params)}});
        }
    }
    gfx::log("sun_closeup: done");
    return 0;
}

} // namespace

int runPhase3Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    if (name == "multiscattering") {
        return runMultiScattering(device, pipelines, outDir);
    }
    if (name == "sky_lut") {
        return runSkyLut(device, pipelines, outDir);
    }
    if (name == "pathtrace") {
        return runPathTrace(device, pipelines, outDir);
    }
    if (name == "sun_closeup") {
        return runSunCloseup(device, pipelines, outDir);
    }
    if (name == "spectral") {
        return runSpectral(device, pipelines, outDir);
    }
    gfx::log("Unknown test: " + std::string(name));
    return 2;
}

} // namespace batch
