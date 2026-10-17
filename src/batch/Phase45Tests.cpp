#include "batch/Phase45Tests.h"

#include "batch/SceneTestCommon.h"
#include "render/ShadowRenderer.h"

#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <nlohmann/json.hpp>

#include <format>

namespace batch {
namespace {

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 360;

nlohmann::json toJson(const glm::dvec3& v)
{
    return {v.x, v.y, v.z};
}

nlohmann::json toJson(const glm::vec3& v)
{
    return {v.x, v.y, v.z};
}

/// カスケードの範囲（Python で受け手の位置を (u, v) に移すため）
nlohmann::json cascadeJson(const render::ShadowCascade& c)
{
    const glm::dvec2 texel = c.texelSizeM();
    return {{"resolution", c.resolution},     {"axis_x", toJson(c.axisX)},
            {"axis_y", toJson(c.axisY)},      {"axis_z", toJson(c.axisZ)},
            {"min_light_m", toJson(c.minLight)}, {"max_light_m", toJson(c.maxLight)},
            {"texel_size_m", {texel.x, texel.y}}};
}

nlohmann::json sceneJson(render::SceneRenderer& scene, const scene::Camera& camera,
                         const render::ShadowRenderer& shadows, const render::ShadowSettings& settings)
{
    nlohmann::json spheres = nlohmann::json::array();
    for (const render::SceneObject& o : scene.objects()) {
        if (!o.terrain) {
            spheres.push_back({{"center_m", toJson(o.centerM)}, {"radius_m", o.radiusM}});
        }
    }
    return {{"camera_position_m", toJson(camera.positionM)},
            {"camera_yaw_deg", glm::degrees(camera.yawRad)},
            {"camera_pitch_deg", glm::degrees(camera.pitchRad)},
            {"vertical_fov_deg", glm::degrees(camera.verticalFovRad)},
            {"spheres", spheres},
            {"shadow_near", cascadeJson(shadows.nearCascade())},
            {"shadow_far", cascadeJson(shadows.farCascade())},
            {"shadow_settings",
             {{"near_range_m", settings.nearRangeM},
              {"normal_offset_texels", settings.normalOffsetTexels},
              {"blend_band", settings.blendBand}}}};
}

// ---------------------------------------------------------------------------
// V3e: 地表の影
// ---------------------------------------------------------------------------

int runShadowSurface(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer atmo(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    render::SceneRenderer sceneRenderer(device, pipelines);
    render::ShadowRenderer shadows(device);
    const render::ShadowSettings settings;
    SceneTargets targets(device, kWidth, kHeight, DXGI_FORMAT_R32G32B32A32_FLOAT);
    const atmosphere::AtmosphereQuality quality;
    const float aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);

    struct Case {
        const char* name;
        glm::vec3 cameraM;
        float pitchDeg;
        float sunZenithDeg;
        float sunAzimuthDeg;
    };
    // 高さ 300 m のカメラは全体のカスケード、地表 2 m のカメラ（近くの球の影）は近景のカスケードを主に使う
    const Case cases[] = {
        {"high_sz45", {0.0f, 300.0f, 0.0f}, -2.0f, 45.0f, -90.0f},
        {"low_sz80", {0.0f, 300.0f, 0.0f}, -2.0f, 80.0f, -90.0f},
        {"sunset_sz88", {0.0f, 300.0f, 0.0f}, -2.0f, 88.0f, -90.0f},
        {"near_sz60", {0.0f, 2.0f, 0.0f}, -10.0f, 60.0f, 0.0f},
    };
    for (const Case& c : cases) {
        atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kSceneTestCoefficientPath);
        params.sunZenithDeg = c.sunZenithDeg;
        params.sunAzimuthDeg = c.sunAzimuthDeg;
        scene::Camera camera = testCamera();
        camera.positionM = c.cameraM;
        camera.pitchRad = glm::radians(c.pitchDeg);

        render::SceneView view;
        view.viewProjRot = camera.projection(aspect) * camera.viewRotation();
        view.cameraPosM = camera.positionM;
        view.output = render::MeshOutput::ShadowDebug;
        device.upload().beginSlot(0);
        device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            atmo.updateLuts(cmd, device.upload(), params, quality, atmosphere::worldToPlanetKm(params, camera.positionM));
            atmo.updateSkyIrradiance(cmd, device.upload(), params,
                                     atmosphere::worldToPlanetKm(params, camera.positionM));
            shadows.updateFar(cmd, device.upload(), sceneRenderer, params.sunDirection(), true);
            shadows.updateNear(cmd, device.upload(), sceneRenderer, camera, aspect, settings);
            view.shadow = shadows.sceneShadow(camera.positionM, settings);
            targets.drawScene(cmd, device.upload(), sceneRenderer, view, params, atmo);
        });
        nlohmann::json meta = sceneJson(sceneRenderer, camera, shadows, settings);
        meta["units"] = "R = sun visibility [0,1], G = N.L, B = near cascade weight";
        meta["case"] = c.name;
        meta["atmosphere"] = atmosphere::toJson(params);
        io::saveExr(outDir / std::format("{}_visibility.exr", c.name), kWidth, kHeight,
                    bytesToFloats(targets.color.readback(device, 16)), meta);
        targets.saveDepth(device, outDir / std::format("{}_depth.exr", c.name), camera.nearM);
        gfx::log(std::string("shadow_surface: ") + c.name);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// V3f: 空気遠近法の影
// ---------------------------------------------------------------------------

int runShadowAerialPerspective(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                               const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer atmo(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    render::SceneRenderer sceneRenderer(device, pipelines);
    render::ShadowRenderer shadows(device);
    const render::ShadowSettings settings;
    SceneTargets targets(device, kWidth, kHeight, DXGI_FORMAT_R32G32B32A32_FLOAT);
    const atmosphere::AtmosphereQuality quality;
    const scene::Camera camera = testCamera();
    const float aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);

    struct Case {
        const char* name;
        float sunZenithDeg;
    };
    const Case cases[] = {{"day_front", 45.0f}, {"low_front", 80.0f}, {"sunset_front", 88.0f}};
    for (const Case& c : cases) {
        atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kSceneTestCoefficientPath);
        params.sunZenithDeg = c.sunZenithDeg;
        params.sunAzimuthDeg = -90.0f;  // カメラの正面（−Z）

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

        // 経路: 地形の面の影はすべて有り。空気遠近法の単散乱の影だけを切り替える
        enum class Path { Geometry, Volume, VolumeNoShadow, RaymarchRgb, RaymarchMean, RaymarchRgbNoShadow };
        const std::pair<Path, const char*> paths[] = {{Path::Geometry, "geometry"},
                                                      {Path::Volume, "volume"},
                                                      {Path::VolumeNoShadow, "volume_noshadow"},
                                                      {Path::RaymarchRgb, "raymarch_rgb"},
                                                      {Path::RaymarchMean, "raymarch_mean"},
                                                      {Path::RaymarchRgbNoShadow, "raymarch_rgb_noshadow"}};
        nlohmann::json meta;
        for (const auto& [path, pathName] : paths) {
            const bool apShadow = path != Path::VolumeNoShadow && path != Path::RaymarchRgbNoShadow;
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                atmo.updateLuts(cmd, device.upload(), params, quality, view.cameraPosKm);
                shadows.updateFar(cmd, device.upload(), sceneRenderer, params.sunDirection(), true);
                shadows.updateNear(cmd, device.upload(), sceneRenderer, camera, aspect, settings);
                atmosphere::SkyView pathView = view;
                if (apShadow) {
                    pathView.shadow = shadows.atmosphereShadow(camera.positionM, settings);
                }
                atmo.updateAerialPerspective(cmd, device.upload(), params, pathView, quality.multipleScattering);
                atmo.updateSkyIrradiance(cmd, device.upload(), params, view.cameraPosKm);
                sceneView.shadow = shadows.sceneShadow(camera.positionM, settings);
                targets.drawScene(cmd, device.upload(), sceneRenderer, sceneView, params, atmo);
                if (path == Path::Volume || path == Path::VolumeNoShadow) {
                    atmo.renderSkyFromLut(cmd, device.upload(), params, pathView, targets.color.uavIndex(), kWidth,
                                          kHeight);
                } else if (path != Path::Geometry) {
                    atmosphere::RaymarchSettings raymarch;
                    raymarch.primarySteps = 2048;
                    raymarch.multipleScattering = true;
                    raymarch.meanTransmittance = path == Path::RaymarchMean;
                    atmo.renderSkyRaymarch(cmd, device.upload(), params, pathView, raymarch, targets.color.uavIndex(),
                                           kWidth, kHeight);
                }
            });
            meta = sceneJson(sceneRenderer, camera, shadows, settings);
            meta["units"] = "luminance per illuminance [sr^-1]";
            meta["case"] = c.name;
            meta["path"] = pathName;
            meta["aerial_perspective_shadow"] = apShadow;
            meta["camera_height_m"] = camera.positionM.y;
            meta["atmosphere"] = atmosphere::toJson(params);
            io::saveExr(outDir / std::format("{}_{}.exr", c.name, pathName), kWidth, kHeight,
                        bytesToFloats(targets.color.readback(device, 16)), meta);
            if (path == Path::Volume) {
                // 空気遠近法のボリューム（Mie 側）。32 × (32 × 32) の画像（行 = スライス z × 32 + y）。
                // A = 視線上の点で太陽が見える割合の平均（Python のレイキャストとボクセルごとに比べる）
                constexpr uint32_t n = atmosphere::AtmosphereRenderer::kAerialPerspectiveSize;
                io::saveExr(outDir / std::format("{}_ap_mie_volume.exr", c.name), n, n * n,
                            bytesToFloats(atmo.aerialPerspectiveMie().readback(device, 16)),
                            {{"units", "RGB = Mie single scattering without phase, A = mean sun visibility"},
                             {"layout", "row = z * 32 + y"}});
            }
        }
        targets.saveDepth(device, outDir / std::format("{}_depth.exr", c.name), camera.nearM);
        gfx::log(std::string("shadow_aerial_perspective: ") + c.name);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// V3g: 空気遠近法の影の方式の比較（design.md §5.6）
// ---------------------------------------------------------------------------

int runShadowApMethods(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereRenderer atmo(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    render::SceneRenderer sceneRenderer(device, pipelines);
    render::ShadowRenderer shadows(device);
    const render::ShadowSettings settings;
    SceneTargets targets(device, kWidth, kHeight, DXGI_FORMAT_R32G32B32A32_FLOAT);
    const atmosphere::AtmosphereQuality quality;
    const float aspect = static_cast<float>(kWidth) / static_cast<float>(kHeight);

    struct Case {
        const char* name;
        glm::vec3 cameraM;
        float yawDeg;
        float pitchDeg;
        float sunZenithDeg;
        float sunAzimuthDeg;
        float g;
    };
    // ground_sz83: 2026-10-16 のユーザーの画面の条件（地表 1 m、近くの球の影がボリュームの格子のブロックに見えた）
    const Case cases[] = {
        {"ground_sz83", {0.0f, 1.0f, 0.0f}, 14.8f, 6.9f, 82.78f, -76.0f, 0.83f},
        {"low_front", {0.0f, 300.0f, 0.0f}, 0.0f, -2.0f, 80.0f, -90.0f, 0.8f},
        {"sunset_front", {0.0f, 300.0f, 0.0f}, 0.0f, -2.0f, 88.0f, -90.0f, 0.8f},
    };
    for (const Case& c : cases) {
        atmosphere::AtmosphereParams params = atmosphere::loadDefaultParams(kSceneTestCoefficientPath);
        params.sunZenithDeg = c.sunZenithDeg;
        params.sunAzimuthDeg = c.sunAzimuthDeg;
        params.miePhaseG = c.g;
        scene::Camera camera = testCamera();
        camera.positionM = c.cameraM;
        camera.yawRad = glm::radians(c.yawDeg);
        camera.pitchRad = glm::radians(c.pitchDeg);

        atmosphere::SkyView view;
        view.invViewProjRot = glm::inverse(camera.projection(aspect) * camera.viewRotation());
        view.cameraPosKm = atmosphere::worldToPlanetKm(params, camera.positionM);
        view.exposureScale = 1.0f / params.sunIlluminanceLux;  // fb = 単位照度あたりの輝度 [sr⁻¹]
        view.sunDisk = false;
        view.depthSrv = targets.depth.srvIndex();
        render::SceneView sceneView;
        sceneView.viewProjRot = camera.projection(aspect) * camera.viewRotation();
        sceneView.cameraPosM = camera.positionM;
        sceneView.exposureScale = view.exposureScale;

        // 経路: 地形の面の影はすべて有り。method_* は方式 0 / 1 / 5、raymarch_* は基準（各点で影を 1 回、2048 区間）
        enum class Path { Geometry, Method0, Method1, Method5, RaymarchMean, RaymarchMeanNoShadow };
        const std::pair<Path, const char*> paths[] = {{Path::Geometry, "geometry"},
                                                      {Path::Method0, "method0_volume_point"},
                                                      {Path::Method1, "method1_volume_filtered"},
                                                      {Path::Method5, "method5_per_pixel"},
                                                      {Path::RaymarchMean, "raymarch_mean"},
                                                      {Path::RaymarchMeanNoShadow, "raymarch_mean_noshadow"}};
        for (const auto& [path, pathName] : paths) {
            device.upload().beginSlot(0);
            device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                atmo.updateLuts(cmd, device.upload(), params, quality, view.cameraPosKm);
                shadows.updateFar(cmd, device.upload(), sceneRenderer, params.sunDirection(), true);
                shadows.updateNear(cmd, device.upload(), sceneRenderer, camera, aspect, settings);
                atmosphere::SkyView pathView = view;
                if (path != Path::RaymarchMeanNoShadow) {
                    pathView.shadow = shadows.atmosphereShadow(camera.positionM, settings);
                }
                pathView.apShadow.method = path == Path::Method1   ? atmosphere::AerialPerspectiveShadowMethod::VolumeFiltered
                                           : path == Path::Method5 ? atmosphere::AerialPerspectiveShadowMethod::PerPixel
                                                                   : atmosphere::AerialPerspectiveShadowMethod::VolumePoint;
                atmo.updateAerialPerspective(cmd, device.upload(), params, pathView, quality.multipleScattering);
                atmo.updateSkyIrradiance(cmd, device.upload(), params, view.cameraPosKm);
                sceneView.shadow = shadows.sceneShadow(camera.positionM, settings);
                targets.drawScene(cmd, device.upload(), sceneRenderer, sceneView, params, atmo);
                if (path == Path::Method0 || path == Path::Method1 || path == Path::Method5) {
                    atmo.renderSkyFromLut(cmd, device.upload(), params, pathView, targets.color.uavIndex(), kWidth,
                                          kHeight);
                } else if (path != Path::Geometry) {
                    atmosphere::RaymarchSettings raymarch;
                    raymarch.primarySteps = 2048;
                    raymarch.multipleScattering = true;
                    raymarch.meanTransmittance = true;  // ボリュームと同じ近似（差を影の扱いだけにする）
                    atmo.renderSkyRaymarch(cmd, device.upload(), params, pathView, raymarch, targets.color.uavIndex(),
                                           kWidth, kHeight);
                }
            });
            nlohmann::json meta = sceneJson(sceneRenderer, camera, shadows, settings);
            meta["units"] = "luminance per illuminance [sr^-1]";
            meta["case"] = c.name;
            meta["path"] = pathName;
            meta["ap_shadow_options"] = {{"filter_scale", view.apShadow.filterScale},
                                         {"per_pixel_steps", view.apShadow.perPixelSteps},
                                         {"per_pixel_bisect_steps", view.apShadow.perPixelBisectSteps}};
            meta["atmosphere"] = atmosphere::toJson(params);
            io::saveExr(outDir / std::format("{}_{}.exr", c.name, pathName), kWidth, kHeight,
                        bytesToFloats(targets.color.readback(device, 16)), meta);
        }
        targets.saveDepth(device, outDir / std::format("{}_depth.exr", c.name), camera.nearM);
        gfx::log(std::string("shadow_ap_methods: ") + c.name);
    }
    return 0;
}

} // namespace

int runPhase45Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                   const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    if (name == "shadow_surface") {
        return runShadowSurface(device, pipelines, outDir);
    }
    if (name == "shadow_aerial_perspective") {
        return runShadowAerialPerspective(device, pipelines, outDir);
    }
    if (name == "shadow_ap_methods") {
        return runShadowApMethods(device, pipelines, outDir);
    }
    gfx::log("Unknown Phase 4.5 test: " + std::string(name));
    return 2;
}

} // namespace batch
