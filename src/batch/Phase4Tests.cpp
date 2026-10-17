#include "batch/Phase4Tests.h"

#include "batch/SceneTestCommon.h"
#include "atmosphere/AtmosphereRenderer.h"
#include "display/DisplayPass.h"
#include "gfx/GpuTimer.h"
#include "io/ExrWriter.h"
#include "render/SceneRenderer.h"
#include "render/ShadowRenderer.h"
#include "scene/Camera.h"

#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <map>

namespace batch {
namespace {

const std::filesystem::path& kCoefficientPath = kSceneTestCoefficientPath;

std::vector<float> toFloats(const std::vector<uint8_t>& bytes)
{
    return bytesToFloats(bytes);
}

// ---------------------------------------------------------------------------
// V3c: 地形の空気遠近法
// ---------------------------------------------------------------------------

int runAerialPerspective(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                         const std::filesystem::path& outDir)
{
    constexpr uint32_t w = 640;
    constexpr uint32_t h = 360;
    atmosphere::AtmosphereRenderer atmo(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    render::SceneRenderer sceneRenderer(device, pipelines);
    SceneTargets targets(device, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT);
    const atmosphere::AtmosphereQuality quality;
    const scene::Camera camera = testCamera();
    const float aspect = static_cast<float>(w) / static_cast<float>(h);

    struct Case {
        const char* name;
        float sunZenithDeg;
        float sunAzimuthDeg;  // −90° = カメラの正面（−Z）
        float g;
    };
    const Case cases[] = {
        {"day_front", 45.0f, -90.0f, 0.8f},
        {"day_front_g099", 45.0f, -90.0f, 0.99f},
        {"day_back", 45.0f, 90.0f, 0.8f},
        {"sunset_front", 88.0f, -90.0f, 0.8f},
    };
    for (const Case& c : cases) {
        atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
        params.sunZenithDeg = c.sunZenithDeg;
        params.sunAzimuthDeg = c.sunAzimuthDeg;
        params.miePhaseG = c.g;

        atmosphere::SkyView view;
        view.invViewProjRot = glm::inverse(camera.projection(aspect) * camera.viewRotation());
        view.cameraPosKm = atmosphere::worldToPlanetKm(params, camera.positionM);
        // [POINT] 露出を 1/E_sun にすると fb = 単位照度あたりの輝度 [sr⁻¹]（太陽の色が白の Hillaire2020 のとき）
        view.exposureScale = 1.0f / params.sunIlluminanceLux;
        view.sunDisk = false;
        view.depthSrv = targets.depth.srvIndex();
        render::SceneView sceneView;
        sceneView.viewProjRot = camera.projection(aspect) * camera.viewRotation();
        sceneView.cameraPosM = camera.positionM;
        sceneView.exposureScale = view.exposureScale;

        nlohmann::json meta = {{"units", "luminance per illuminance [sr^-1]"},
                               {"case", c.name},
                               {"camera_height_m", camera.positionM.y},
                               {"camera_pitch_deg", glm::degrees(camera.pitchRad)},
                               {"vertical_fov_deg", glm::degrees(camera.verticalFovRad)},
                               {"atmosphere", atmosphere::toJson(params)}};

        enum class Path { Geometry, Volume, RaymarchRgb, RaymarchMean };
        const std::pair<Path, const char*> paths[] = {{Path::Geometry, "geometry"},
                                                      {Path::Volume, "volume"},
                                                      {Path::RaymarchRgb, "raymarch_rgb"},
                                                      {Path::RaymarchMean, "raymarch_mean"}};
        for (const auto& [path, pathName] : paths) {
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                atmo.updateLuts(cmd, device.upload(), params, quality, view.cameraPosKm);
                atmo.updateAerialPerspective(cmd, device.upload(), params, view, quality.multipleScattering);
                atmo.updateSkyIrradiance(cmd, device.upload(), params, view.cameraPosKm);
                targets.drawScene(cmd, device.upload(), sceneRenderer, sceneView, params, atmo);
                if (path == Path::Volume) {
                    atmo.renderSkyFromLut(cmd, device.upload(), params, view, targets.color.uavIndex(), w, h);
                } else if (path != Path::Geometry) {
                    atmosphere::RaymarchSettings settings;
                    settings.primarySteps = 2048;
                    settings.multipleScattering = true;
                    settings.meanTransmittance = path == Path::RaymarchMean;
                    atmo.renderSkyRaymarch(cmd, device.upload(), params, view, settings, targets.color.uavIndex(), w, h);
                }
            });
            meta["path"] = pathName;
            io::saveExr(outDir / std::format("{}_{}.exr", c.name, pathName), w, h,
                        toFloats(targets.color.readback(device, 16)), meta);
        }
        targets.saveDepth(device, outDir / std::format("{}_depth.exr", c.name), camera.nearM);
        gfx::log(std::string("aerial_perspective: ") + c.name);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// V3d: 空からの照度
// ---------------------------------------------------------------------------

int runSkyIrradiance(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    constexpr uint32_t w = 2048;
    constexpr uint32_t h = 1024;
    atmosphere::AtmosphereRenderer atmo(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    gfx::Texture image(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"SkyEquirect"});
    const atmosphere::AtmosphereQuality quality;
    nlohmann::json all = nlohmann::json::array();
    for (const float g : {0.8f, 0.99f}) {
        for (const float sunZenithDeg : {30.0f, 80.0f}) {
            atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
            params.miePhaseG = g;
            params.sunZenithDeg = sunZenithDeg;
            const glm::vec3 camera(0.0f, params.bottomRadiusKm + 0.2f, 0.0f);
            atmosphere::SkyView view;
            view.cameraPosKm = camera;
            view.output = atmosphere::RaymarchOutput::TestPerLux;
            view.sunDisk = false;
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                image.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                atmo.updateLuts(cmd, device.upload(), params, quality, camera);
                atmo.updateSkyIrradiance(cmd, device.upload(), params, camera);
                atmo.renderSkyFromLut(cmd, device.upload(), params, view, image.uavIndex(), w, h);
            });
            const std::string stem = std::format("sky_g{:g}_sz{:g}", g, sunZenithDeg);
            io::saveExr(outDir / (stem + ".exr"), w, h, toFloats(image.readback(device, 16)),
                        {{"units", "luminance per illuminance [sr^-1]"}, {"atmosphere", atmosphere::toJson(params)}});
            const std::vector<float> sh = toFloats(atmo.skyIrradiance().readback(device));
            nlohmann::json coefficients = nlohmann::json::array();
            for (int i = 0; i < 9; ++i) {
                coefficients.push_back({sh[i * 4 + 0], sh[i * 4 + 1], sh[i * 4 + 2]});
            }
            all.push_back({{"image", stem + ".exr"},
                           {"mie_phase_g", g},
                           {"sun_zenith_deg", sunZenithDeg},
                           {"irradiance_sh_per_lux", coefficients}});
        }
    }
    std::ofstream(outDir / "sky_irradiance.json") << all.dump(2) << "\n";
    gfx::log("sky_irradiance: done");
    return 0;
}

// ---------------------------------------------------------------------------
// V8: 性能
// ---------------------------------------------------------------------------

int runPerf(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer atmo(device, pipelines);  // 実用と同じ RGBA16F
    render::SceneRenderer sceneRenderer(device, pipelines);
    render::ShadowRenderer shadows(device);
    const render::ShadowSettings shadowSettings;
    display::DisplayPass displayPass(pipelines);
    gfx::GpuTimer timer(device, 1);
    const scene::Camera camera = testCamera();
    atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kCoefficientPath);
    params.sunZenithDeg = 45.0f;
    params.sunAzimuthDeg = -90.0f;

    nlohmann::json report = nlohmann::json::object();
    report["adapter"] = device.adapterName();
#ifdef NDEBUG
    report["build"] = "Release";
#else
    report["build"] = "Debug (not representative)";
#endif
    for (const auto& [w, h] : {std::pair<uint32_t, uint32_t>{1920, 1080}, {3840, 2160}}) {
        SceneTargets targets(device, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT);
        gfx::Texture output(device, {w, h, DXGI_FORMAT_R10G10B10A2_UNORM, false, false, true, L"PerfOutput"});
        const float aspect = static_cast<float>(w) / static_cast<float>(h);
        atmosphere::SkyView view;
        view.invViewProjRot = glm::inverse(camera.projection(aspect) * camera.viewRotation());
        view.cameraPosKm = atmosphere::worldToPlanetKm(params, camera.positionM);
        view.exposureScale = 1.0f / (1.2f * std::exp2(15.0f));
        view.depthSrv = targets.depth.srvIndex();
        render::SceneView sceneView;
        sceneView.viewProjRot = camera.projection(aspect) * camera.viewRotation();
        sceneView.cameraPosM = camera.positionM;
        sceneView.exposureScale = view.exposureScale;

        std::map<std::string, std::vector<double>> samples;
        constexpr int kWarmup = 20;
        constexpr int kFrames = 100;
        for (int frame = 0; frame < kWarmup + kFrames; ++frame) {
            const atmosphere::AtmosphereQuality quality;
            // [POINT] Transmittance / Multi-scattering LUT は普段は媒質が変わったときしか作らないが、
            //         コストを測るために毎フレーム作り直させる（論文の表 2 と同じく LUT ごとの時間を出す）。
            atmo.invalidateLuts();
            device.upload().beginSlot(0);
            timer.beginFrame(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                uint32_t t = timer.begin(cmd, "transmittance_lut_500");
                atmo.updateTransmittanceLut(cmd, device.upload(), params, quality.transmittanceSteps);
                timer.end(cmd, t);
                t = timer.begin(cmd, "multiscattering_lut_20");
                atmo.updateMultiScatteringLut(cmd, device.upload(), params, quality.multiScatteringSteps,
                                              atmosphere::SunTransmittanceMode::Lut, 0, gfx::kInvalidDescriptorIndex,
                                              quality.multiScatteringUniformSteps);
                timer.end(cmd, t);
                t = timer.begin(cmd, "sky_view_lut_30");
                atmo.updateSkyViewLut(cmd, device.upload(), params, view.cameraPosKm, quality.skyViewSteps, true);
                timer.end(cmd, t);
                t = timer.begin(cmd, "aerial_perspective_volume");
                atmo.updateAerialPerspective(cmd, device.upload(), params, view, true);
                timer.end(cmd, t);
                t = timer.begin(cmd, "sky_irradiance_sh");
                atmo.updateSkyIrradiance(cmd, device.upload(), params, view.cameraPosKm);
                timer.end(cmd, t);
                // Phase 4.5: 太陽のシャドウマップ（全体は普段は太陽の向きの変更時だけ。測るために毎フレーム描く）と、
                // 影ありの空気遠近法のボリューム（影なしの上のボリュームとの差が影のコスト）
                t = timer.begin(cmd, "shadow_map_far_4096");
                shadows.updateFar(cmd, device.upload(), sceneRenderer, params.sunDirection(), true);
                timer.end(cmd, t);
                t = timer.begin(cmd, "shadow_map_near_2048");
                shadows.updateNear(cmd, device.upload(), sceneRenderer, camera, aspect, shadowSettings);
                timer.end(cmd, t);
                atmosphere::SkyView shadowView = view;
                shadowView.shadow = shadows.atmosphereShadow(camera.positionM, shadowSettings);
                t = timer.begin(cmd, "aerial_perspective_volume_shadow");
                atmo.updateAerialPerspective(cmd, device.upload(), params, shadowView, true);
                timer.end(cmd, t);
                // 空気遠近法の影の方式（design.md §5.6）: 1 = ボリュームでぼかす。5 のボリュームは影なし（上の
                // aerial_perspective_volume と同じ）で、合成のコストが増える（下の composite_per_pixel）
                atmosphere::SkyView filteredView = shadowView;
                filteredView.apShadow.method = atmosphere::AerialPerspectiveShadowMethod::VolumeFiltered;
                t = timer.begin(cmd, "aerial_perspective_volume_shadow_filtered");
                atmo.updateAerialPerspective(cmd, device.upload(), params, filteredView, true);
                timer.end(cmd, t);
                // 地形・テスト球（作例のシーン。大気のコストではない）。クリア / 描画 / 合成に渡す状態遷移に分けて測る
                t = timer.begin(cmd, "scene_clear");
                targets.color.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
                targets.depth.transition(cmd, D3D12_RESOURCE_STATE_DEPTH_WRITE);
                const float black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                cmd->ClearRenderTargetView(targets.color.rtv(), black, 0, nullptr);
                cmd->ClearDepthStencilView(targets.depth.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);
                timer.end(cmd, t);
                t = timer.begin(cmd, "scene_terrain_spheres");
                sceneRenderer.record(cmd, device.upload(), targets.color.rtv(), targets.color.desc().format,
                                     targets.depth.dsv(), w, h, sceneView, params, atmo);
                timer.end(cmd, t);
                // 同じ描画を影あり（2 段のカスケード、3×3 の PCF）で。上との差が地表の影のコスト
                cmd->ClearRenderTargetView(targets.color.rtv(), black, 0, nullptr);
                cmd->ClearDepthStencilView(targets.depth.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);
                render::SceneView shadowSceneView = sceneView;
                shadowSceneView.shadow = shadows.sceneShadow(camera.positionM, shadowSettings);
                t = timer.begin(cmd, "scene_terrain_spheres_shadow");
                sceneRenderer.record(cmd, device.upload(), targets.color.rtv(), targets.color.desc().format,
                                     targets.depth.dsv(), w, h, shadowSceneView, params, atmo);
                timer.end(cmd, t);
                // [POINT] 合成のコンピュートシェーダーが深度を読むために SRV にする遷移。GPU によっては深度の
                //         圧縮の展開が走るので、エンジンに組み込むときのコストとして別に記録する。
                t = timer.begin(cmd, "scene_depth_color_transitions");
                targets.depth.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
                targets.color.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                timer.end(cmd, t);
                t = timer.begin(cmd, "sky_and_aerial_perspective_composite");
                atmo.renderSkyFromLut(cmd, device.upload(), params, view, targets.color.uavIndex(), w, h);
                timer.end(cmd, t);
                // 方式 5 の合成（時間だけを測る。結果の画像は使わない）。視線上 32 点 + 境界の二分探索 6 回
                atmosphere::SkyView perPixelView = shadowView;
                perPixelView.depthSrv = view.depthSrv;
                perPixelView.apShadow.method = atmosphere::AerialPerspectiveShadowMethod::PerPixel;
                t = timer.begin(cmd, "sky_and_aerial_perspective_composite_per_pixel");
                atmo.renderSkyFromLut(cmd, device.upload(), params, perPixelView, targets.color.uavIndex(), w, h);
                timer.end(cmd, t);
                targets.color.transition(cmd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
                output.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
                display::DisplayPassInput input;
                input.sceneColorSrv = targets.color.srvIndex();
                input.encode = display::OutputEncode::Hdr10Pq;
                input.gt7 = display::makeGT7Constants(display::GT7Mode::HDR, 1000.0f);
                t = timer.begin(cmd, "display_gt7_pq");
                displayPass.record(cmd, device.upload(), output.rtv(), DXGI_FORMAT_R10G10B10A2_UNORM, w, h, input);
                timer.end(cmd, t);
                // Transmittance LUT（論文の PC 設定 40 区間）も参考に測る
                atmo.invalidateLuts();
                t = timer.begin(cmd, "transmittance_lut_40_info");
                atmo.updateTransmittanceLut(cmd, device.upload(), params, 40);
                timer.end(cmd, t);
                timer.resolve(cmd);
            });
            if (frame >= kWarmup) {
                for (const gfx::GpuTimer::Result& r : timer.read(0)) {
                    samples[r.name].push_back(r.milliseconds);
                }
            }
        }
        nlohmann::json passes = nlohmann::json::object();
        for (auto& [name, values] : samples) {
            std::sort(values.begin(), values.end());
            passes[name] = {{"median_ms", values[values.size() / 2]},
                            {"min_ms", values.front()},
                            {"max_ms", values.back()}};
        }
        report[std::format("{}x{}", w, h)] = passes;
        gfx::log(std::format("perf: {}x{} done", w, h));
    }
    std::filesystem::create_directories(outDir);
    std::ofstream(outDir / "perf.json") << report.dump(2) << "\n";
    return 0;
}

} // namespace

int runPhase4Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    if (name == "aerial_perspective") {
        return runAerialPerspective(device, pipelines, outDir);
    }
    if (name == "sky_irradiance") {
        return runSkyIrradiance(device, pipelines, outDir);
    }
    if (name == "perf") {
        return runPerf(device, pipelines, outDir);
    }
    gfx::log("Unknown test: " + std::string(name));
    return 2;
}

} // namespace batch
