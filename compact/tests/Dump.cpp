// Test-only adapter: compile once against compact and once against the untouched reference.
#include "atmosphere/AtmosphereRenderer.h"
#include "render/SceneRenderer.h"
#include "render/ShadowRenderer.h"
#include "scene/Camera.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <iostream>
#include <numbers>
#include <vector>
#ifdef COMPACT_REFERENCE
#include "atmosphere/MiePhaseTables.h"
namespace g = gfx;
namespace a = atmosphere;
namespace r = render;
namespace s = scene;
#else
#include "atmosphere/FixedData.h"
namespace g = compact::gfx;
namespace a = compact::atmosphere;
namespace r = compact::render;
namespace s = compact::scene;
#endif
namespace fs = std::filesystem;
#include "Readback.h"
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
void save(const fs::path &path, const std::vector<float> &data) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * sizeof(float)));
    require(bool(f), "Dump write failed");
}
void saveTexture(g::Device &device, g::Texture &texture, const fs::path &path) {
    bool half = texture.desc().format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    auto bytes = readTexture(device, texture, half ? 8 : 16);
    std::vector<float> result(bytes.size() / (half ? 2 : 4));
    if (half)
        for (size_t i = 0; i < result.size(); ++i) {
            uint16_t v;
            std::memcpy(&v, bytes.data() + i * 2, 2);
            result[i] = glm::unpackHalf1x16(v);
        }
    else
        std::memcpy(result.data(), bytes.data(), bytes.size());
    save(path, result);
}
struct Case {
    std::string name;
    float zenith = 45, altitude = 2, yaw = 0, aod = 0.1f, albedo = 0.3f, lux = -1, ev = 15, radius = 0.004675f;
    bool shadow = true;
};

int main(int argc, char **argv) {
    try {
        fs::path out = "dumps";
        bool debug = false, qualityTest = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--out" && i + 1 < argc)
                out = argv[++i];
            else if (arg == "--debug")
                debug = true;
            else if (arg == "--quality")
                qualityTest = true;
            else
                throw std::runtime_error("Unknown option");
        }
        g::Device device(debug);
        g::PipelineLibrary pipelines(device, COMPACT_SHADER_DIR);
#ifdef COMPACT_REFERENCE
        auto sets = a::loadCoefficientSets(COMPACT_COEFFICIENTS);
        a::MiePhaseTables phase(device, sets);
        {
            std::vector<float> v;
            for (const auto &row : sets[0].findAerosolPreset("C+A+B")->components[0].phaseMoments)
                for (int c = 0; c < 3; ++c)
                    v.push_back(row[c]);
            save(out / "phase_moments.f32", v);
        }
#else
        {
            std::vector<float> v;
            for (const auto &row : a::fixed::moments)
                for (int c = 0; c < 3; ++c)
                    v.push_back(row[c]);
            save(out / "phase_moments.f32", v);
        }
#endif
        std::vector<Case> cases;
        for (auto sun : {45.0f, 80.0f, 88.0f})
            for (auto altitude : {2.0f, 3000.0f})
                for (auto yaw : {-90.0f, 90.0f}) {
                    Case c;
                    c.name = "sun" + std::to_string(int(sun)) + "_h" + std::to_string(int(altitude)) + "_yaw" +
                             std::to_string(int(yaw));
                    c.zenith = sun;
                    c.altitude = altitude;
                    c.yaw = yaw;
                    cases.push_back(c);
                }
        for (auto aod : {0.001f, 1.0f}) {
            Case c;
            c.aod = aod;
            c.name = aod < 0.1f ? "aod_min" : "aod_max";
            cases.push_back(c);
        }
        for (auto albedo : {0.0f, 1.0f}) {
            Case c;
            c.albedo = albedo;
            c.name = albedo == 0 ? "albedo_zero" : "albedo_one";
            cases.push_back(c);
        }
        {
            Case c;
            c.lux = 0;
            c.name = "zero_lux";
            cases.push_back(c);
        }
        {
            Case c;
            c.ev = 12;
            c.name = "exposure";
            cases.push_back(c);
        }
        {
            Case c;
            c.radius = 0.02f;
            c.zenith = 80;
            c.yaw = 90;
            c.name = "large_disk";
            cases.push_back(c);
        }
        {
            Case c;
            c.shadow = false;
            c.zenith = 80;
            c.name = "no_shadow";
            cases.push_back(c);
        }
        {
            Case c;
            c.zenith = 80;
            c.yaw = 14.8f;
            c.altitude = 1;
            c.name = "near_shadow";
            cases.push_back(c);
        }
        {
            Case c;
            c.zenith = 80;
            c.yaw = 90;
            c.name = "sun_disk";
            cases.push_back(c);
        }
        {
            Case c;
            c.aod = 1;
            c.albedo = 1;
            c.zenith = 88;
            c.name = "dense_bright_ground";
            cases.push_back(c);
        }
        for (bool fp32 : {true, false}) {
            const auto format = fp32 ? DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
            a::AtmosphereRenderer atmosphere(device, pipelines, format);
            r::SceneRenderer scene(device, pipelines);
            r::ShadowRenderer shadow(device);
            r::ShadowSettings shadowSettings;
            g::Texture color(device, {320, 180, format, true, true, true, L"Compare scene"});
            g::Texture depth(device, {320, 180, DXGI_FORMAT_D32_FLOAT, true, false, false, L"Compare depth", 1, true});
            for (const auto &test : cases) {
                const auto folder = out / (fp32 ? "fp32" : "fp16") / test.name;
                s::Camera camera;
                camera.positionM = {0, test.altitude, 0};
                camera.yawRad = glm::radians(test.yaw);
                camera.pitchRad = glm::radians(-2.0f);
                const auto vp = camera.projection(320.0f / 180.0f) * camera.viewRotation();
                const float exposure = 1 / (1.2f * std::exp2(test.ev));
#ifdef COMPACT_REFERENCE
                auto settings = a::loadDefaultParams(COMPACT_COEFFICIENTS);
                settings.applyCoefficientSet(sets[0]);
                settings.applyAerosolPreset(*sets[0].findAerosolPreset("C+A+B"));
                settings.setAerosolAod550(*sets[0].findAerosolPreset("C+A+B"), test.aod);
                settings.groundAlbedo = glm::vec3(test.albedo);
                settings.sunZenithDeg = test.zenith;
                settings.sunAzimuthDeg = 0;
                settings.sunAngularRadiusRad = test.radius;
                if (test.lux >= 0)
                    settings.sunIlluminanceLux = test.lux;
                a::SkyView frame;
                frame.cameraPosKm = a::worldToPlanetKm(settings, camera.positionM);
                frame.invViewProjRot = glm::inverse(vp);
                frame.exposureScale = exposure;
                frame.apTransmittance = a::AerialPerspectiveTransmittanceMode::VolumeMean;
                frame.apShadow.method = a::AerialPerspectiveShadowMethod::PerPixel;
                a::AtmosphereQuality q;
                q.multiScatteringModel = a::MultiScatteringModel::ShL2Ground;
                auto direction = settings.sunDirection();
#else
                a::AtmosphereSettings settings;
                settings.aod550 = test.aod;
                settings.groundAlbedo = glm::vec3(test.albedo);
                settings.sunAngularRadiusRad = test.radius;
                float theta = test.zenith * std::numbers::pi_v<float> / 180.0f;
                settings.sunDirectionWs = {std::sin(theta), std::cos(theta), 0};
                if (test.lux >= 0)
                    settings.sunIlluminanceLux = test.lux;
                atmosphere.setSettings(settings);
                settings = atmosphere.settings();
                a::AtmosphereFrameInput frame;
                frame.cameraPosM = camera.positionM;
                frame.invViewProjRot = glm::inverse(vp);
                frame.exposureScale = exposure;
                auto direction = settings.sunDirectionWs;
#endif
                auto *cmd = device.beginFrame();
                auto &upload = device.upload();
#ifdef COMPACT_REFERENCE
                atmosphere.updateLuts(cmd, upload, settings, q, frame.cameraPosKm);
#else
                atmosphere.updateLuts(cmd, upload, frame);
#endif
                shadow.updateFar(cmd, upload, scene, direction);
                shadow.updateNear(cmd, upload, scene, camera, 320.0f / 180.0f, shadowSettings);
                if (test.shadow)
                    frame.shadow = shadow.atmosphereShadow(camera.positionM, shadowSettings);
#ifdef COMPACT_REFERENCE
                atmosphere.updateAerialPerspective(cmd, upload, settings, frame, true);
                atmosphere.updateSkyIrradiance(cmd, upload, settings, frame.cameraPosKm);
#else
                atmosphere.updateAerialPerspective(cmd, upload, frame);
                atmosphere.updateSkyIrradiance(cmd, upload, frame);
#endif
                color.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
                depth.transition(cmd, D3D12_RESOURCE_STATE_DEPTH_WRITE);
                const float clear[4] = {};
                cmd->ClearRenderTargetView(color.rtv(), clear, 0, nullptr);
                cmd->ClearDepthStencilView(depth.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0, 0, 0, nullptr);
                r::SceneView view;
                view.viewProjRot = vp;
                view.cameraPosM = camera.positionM;
                view.exposureScale = exposure;
                view.shadow = shadow.sceneShadow(camera.positionM, shadowSettings);
                scene.record(cmd, upload, color.rtv(), format, depth.dsv(), 320, 180, view, settings, atmosphere);
                depth.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
                color.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                frame.depthSrv = depth.srvIndex();
#ifdef COMPACT_REFERENCE
                atmosphere.renderSkyFromLut(cmd, upload, settings, frame, color.uavIndex(), 320, 180);
#else
                atmosphere.composite(cmd, upload, frame, color.uavIndex(), 320, 180);
#endif
                device.endFrame();
                device.waitIdle();
                saveTexture(device, color, folder / "scene.f32");
                saveTexture(device, atmosphere.transmittanceLut(), folder / "transmittance.f32");
                saveTexture(device, atmosphere.multiScatteringShLut(), folder / "ms.f32");
                saveTexture(device, atmosphere.skyViewLut(), folder / "sky.f32");
                saveTexture(device, atmosphere.skyViewMieLut(), folder / "sky_mie.f32");
                saveTexture(device, atmosphere.aerialPerspective(), folder / "ap.f32");
                saveTexture(device, atmosphere.aerialPerspectiveMie(), folder / "ap_mie.f32");
                saveTexture(device, atmosphere.aerialPerspectiveMultiple(), folder / "ap_ms.f32");
                auto sh = readBuffer(device, atmosphere.skyIrradiance());
                std::vector<float> shValues(sh.size() / 4);
                std::memcpy(shValues.data(), sh.data(), sh.size());
                save(folder / "irradiance.f32", shValues);
                auto db = readTexture(device, depth, 4);
                std::vector<float> depths(db.size() / 4);
                std::memcpy(depths.data(), db.data(), db.size());
                save(folder / "depth.f32", depths);
                std::cout << (fp32 ? "fp32 " : "fp16 ") << test.name << '\n';
            }
#ifndef COMPACT_REFERENCE
            if (qualityTest) {
                // Check resizing, parameter invalidation, every quality boundary and finite GPU output.
                a::AtmosphereFrameInput frame;
                s::Camera camera;
                camera.pitchRad = glm::radians(-2.0f);
                const auto vp = camera.projection(320.0f / 180.0f) * camera.viewRotation();
                frame.invViewProjRot = glm::inverse(vp);
                frame.exposureScale = 1 / (1.2f * std::exp2(15.0f));
                frame.depthSrv = depth.srvIndex();
                frame.shadow = shadow.atmosphereShadow(camera.positionM, shadowSettings);
                auto render = [&]() {
                    auto *cmd = device.beginFrame();
                    auto &upload = device.upload();
                    auto updated = atmosphere.updateLuts(cmd, upload, frame);
                    atmosphere.updateAerialPerspective(cmd, upload, frame);
                    atmosphere.updateSkyIrradiance(cmd, upload, frame);
                    color.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
                    depth.transition(cmd, D3D12_RESOURCE_STATE_DEPTH_WRITE);
                    const float clear[4] = {};
                    cmd->ClearRenderTargetView(color.rtv(), clear, 0, nullptr);
                    cmd->ClearDepthStencilView(depth.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0, 0, 0, nullptr);
                    r::SceneView view;
                    view.viewProjRot = vp;
                    view.cameraPosM = camera.positionM;
                    view.exposureScale = frame.exposureScale;
                    view.shadow = shadow.sceneShadow(camera.positionM, shadowSettings);
                    scene.record(cmd, upload, color.rtv(), format, depth.dsv(), 320, 180, view, atmosphere.settings(),
                                 atmosphere);
                    depth.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
                    color.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    atmosphere.composite(cmd, upload, frame, color.uavIndex(), 320, 180);
                    device.endFrame();
                    device.waitIdle();
                    return updated;
                };
                render();
                auto updates = render();
                require(!updates.transmittance && !updates.multiScattering, "Unchanged LUTs rebuilt");
                auto setting = atmosphere.settings();
                setting.sunIlluminanceLux += 1;
                setting.sunAngularRadiusRad = 0.01f;
                atmosphere.setSettings(setting);
                updates = render();
                require(!updates.transmittance && !updates.multiScattering, "Display-only change rebuilt medium LUTs");
                setting.groundAlbedo = glm::vec3(0.5f);
                atmosphere.setSettings(setting);
                updates = render();
                require(!updates.transmittance && updates.multiScattering, "Albedo invalidation incorrect");
                setting.aod550 = 0.02f;
                atmosphere.setSettings(setting);
                updates = render();
                require(updates.transmittance && updates.multiScattering, "AOD invalidation incorrect");
                {
                    auto q = atmosphere.quality();
                    q.aerialPerspectiveSize = 16;
                    atmosphere.setQuality(q);
                    updates = render();
                    require(!updates.transmittance && !updates.multiScattering, "AP resize rebuilt medium LUTs");
                }
                for (int level : {0, 2, 1, 0, 1}) {
                    a::AtmosphereQuality q;
                    q.transmittanceWidth = 128u << level;
                    q.transmittanceHeight = 32u << level;
                    q.multiScatteringSize = 16u << level;
                    q.skyViewWidth = 96u << level;
                    q.skyViewHeight = 54u << level;
                    q.aerialPerspectiveSize = 16u << level;
                    if (level == 0) {
                        q.transmittanceSteps = q.multiScatteringSteps = q.skyViewSteps = q.apMaxSteps =
                            q.perPixelSteps = 4;
                        q.bisectionSteps = 0;
                    }
                    if (level == 2) {
                        q.transmittanceSteps = 1000;
                        q.multiScatteringSteps = q.skyViewSteps = q.apMaxSteps = 256;
                        q.perPixelSteps = 128;
                        q.bisectionSteps = 12;
                    }
                    atmosphere.setQuality(q);
                    render();
                    for (auto *texture : {&atmosphere.transmittanceLut(), &atmosphere.multiScatteringShLut(),
                                          &atmosphere.skyViewLut(), &atmosphere.aerialPerspective(), &color}) {
                        auto bytes = readTexture(device, *texture, fp32 ? 16 : 8);
                        for (size_t i = 0; i < bytes.size(); i += fp32 ? 4 : 2) {
                            float v;
                            if (fp32)
                                std::memcpy(&v, bytes.data() + i, 4);
                            else {
                                uint16_t h;
                                std::memcpy(&h, bytes.data() + i, 2);
                                v = glm::unpackHalf1x16(h);
                            }
                            require(std::isfinite(v), "Nonfinite quality output");
                        }
                    }
                }
                auto invalid = atmosphere.quality();
                invalid.aerialPerspectiveSize = 0;
                bool rejected = false;
                try {
                    atmosphere.setQuality(invalid);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                require(rejected && atmosphere.quality().aerialPerspectiveSize == 32,
                        "Invalid quality changed live resources");
                std::cout << (fp32 ? "fp32 " : "fp16 ") << "quality/invalidation checks passed\n";
            }
#else
            (void)qualityTest;
#endif
            device.waitIdle();
        }
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> messages;
        if (SUCCEEDED(device.d3d()->QueryInterface(IID_PPV_ARGS(&messages))))
            for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
                SIZE_T n = 0;
                messages->GetMessage(i, nullptr, &n);
                std::vector<uint8_t> storage(n);
                auto *msg = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
                messages->GetMessage(i, msg, &n);
                require(msg->Severity > D3D12_MESSAGE_SEVERITY_ERROR, msg->pDescription);
            }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
