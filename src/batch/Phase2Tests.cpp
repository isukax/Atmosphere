#include "batch/Phase2Tests.h"

#include "atmosphere/AtmosphereRenderer.h"
#include "io/ExrWriter.h"

#include <cstring>
#include <format>

namespace batch {
namespace {

std::vector<float> toFloats(const std::vector<uint8_t>& bytes)
{
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

atmosphere::AtmosphereParams defaultParams()
{
    // 検証は論文値のセット（Hillaire2020、Rec.709）で行う。
    return atmosphere::loadDefaultParams(std::filesystem::path(ATMOS_DATA_DIR) / "atmosphere_coefficients.json");
}

int runTransmittance(gfx::Device& device, const gfx::PipelineLibrary& pipelines, const std::filesystem::path& outDir)
{
    const atmosphere::AtmosphereParams params = defaultParams();
    // [POINT] 数値検証では LUT を float32 で作る（RGBA16F の格納丸めは別問題として切り離す）。
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    constexpr uint32_t w = atmosphere::AtmosphereRenderer::kTransmittanceWidth;
    constexpr uint32_t h = atmosphere::AtmosphereRenderer::kTransmittanceHeight;
    gfx::Texture rMu(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"TransmittanceRMu"});

    for (const uint32_t steps : {500u, 40u}) {
        device.upload().beginSlot(0);
        device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
            rMu.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            renderer.updateTransmittanceLut(cmd, device.upload(), params, steps, rMu.uavIndex());
        });
        const std::vector<float> lut = toFloats(renderer.transmittanceLut().readback(device, 16));
        io::saveExr(outDir / std::format("transmittance_steps{}.exr", steps), w, h, lut,
                    {{"units", "transmittance (unitless)"},
                     {"color_space", display::toString(params.workingSpace)},
                     {"integration", "trapezoid"},
                     {"step_count", steps},
                     {"atmosphere", atmosphere::toJson(params)}});
    }
    io::saveExr(outDir / "transmittance_r_mu.exr", w, h, toFloats(rMu.readback(device, 16)),
                {{"units", "R = r [km] (planet center), G = mu (cos zenith)"},
                 {"atmosphere", atmosphere::toJson(params)}});
    gfx::log("transmittance: wrote LUTs");
    return 0;
}

int runSingleScattering(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                        const std::filesystem::path& outDir)
{
    atmosphere::AtmosphereParams params = defaultParams();
    atmosphere::AtmosphereRenderer renderer(device, pipelines, DXGI_FORMAT_R32G32B32A32_FLOAT);
    constexpr uint32_t w = 32;
    constexpr uint32_t h = 16;
    gfx::Texture radiance(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"SSRadiance"});
    gfx::Texture directions(device, {w, h, DXGI_FORMAT_R32G32B32A32_FLOAT, false, true, false, L"SSDirections"});

    // 太陽方向の透過率の求め方 3 通り:
    //   nested … LUT なしの入れ子レイマーチ（V2 の検証対象）
    //   lut40  … Transmittance LUT（40 区間 = 論文の PC 設定）を引く（参考値）
    //   lut500 … Transmittance LUT（500 区間）を引く（参考値。lut40 との差が LUT の積分誤差、
    //             nested との差が LUT の双線形補間の誤差）
    struct Variant {
        const char* name;
        atmosphere::SunTransmittanceMode mode;
        uint32_t lutSteps;
    };
    const Variant variants[] = {
        {"nested", atmosphere::SunTransmittanceMode::Nested, 0},
        {"lut40", atmosphere::SunTransmittanceMode::Lut, 40},
        {"lut500", atmosphere::SunTransmittanceMode::Lut, 500},
    };

    for (const float altitudeKm : {0.2f, 10.0f}) {
        for (const float sunZenithDeg : {30.0f, 60.0f, 85.0f}) {
            params.sunZenithDeg = sunZenithDeg;
            for (const Variant& variant : variants) {
                const atmosphere::SunTransmittanceMode mode = variant.mode;
                if (variant.lutSteps > 0) {
                    device.upload().beginSlot(0);
                    device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                        renderer.updateTransmittanceLut(cmd, device.upload(), params, variant.lutSteps);
                    });
                }
                // [POINT] リファレンス設定: 視線 2048 区間（中点則）、太陽方向 256 区間（台形則）。
                //         1 回のディスパッチが長すぎると GPU のタイムアウト（TDR）になるので、この程度に抑える。
                atmosphere::RaymarchSettings settings;
                settings.primarySteps = 2048;
                settings.sunMode = mode;
                settings.nestedSteps = 256;
                settings.multipleScattering = false;  // V2 は単散乱のみ
                atmosphere::SkyView view;
                view.cameraPosKm = glm::vec3(0.0f, params.bottomRadiusKm + altitudeKm, 0.0f);
                view.output = atmosphere::RaymarchOutput::TestPerLux;
                view.directionUav = directions.uavIndex();
                view.sunDisk = false;

                device.upload().beginSlot(0);
                device.executeImmediate([&](ID3D12GraphicsCommandList* cmd) {
                    radiance.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    directions.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    renderer.renderSkyRaymarch(cmd, device.upload(), params, view, settings, radiance.uavIndex(), w, h);
                });
                const std::string stem = std::format("ss_h{:g}_sz{:g}", altitudeKm, sunZenithDeg);
                io::saveExr(outDir / (stem + "_" + variant.name + ".exr"), w, h,
                            toFloats(radiance.readback(device, 16)),
                            {{"units", "luminance per illuminance [sr^-1] (= cd/m2 per lux)"},
                             {"color_space", display::toString(params.workingSpace)},
                             {"camera_altitude_km", altitudeKm},
                             {"sun_transmittance", variant.name},
                             {"primary_steps", settings.primarySteps},
                             {"nested_steps", settings.nestedSteps},
                             {"transmittance_lut_steps", variant.lutSteps},
                             {"atmosphere", atmosphere::toJson(params)}});
                if (mode == atmosphere::SunTransmittanceMode::Nested) {
                    io::saveExr(outDir / (stem + "_directions.exr"), w, h,
                                toFloats(directions.readback(device, 16)),
                                {{"units", "unit view direction (planet frame, +Y = zenith)"}});
                }
            }
        }
    }
    gfx::log("single_scattering: wrote results");
    return 0;
}

} // namespace

int runPhase2Test(gfx::Device& device, const gfx::PipelineLibrary& pipelines, std::string_view name,
                  const std::filesystem::path& outDir)
{
    std::filesystem::create_directories(outDir);
    if (name == "transmittance") {
        return runTransmittance(device, pipelines, outDir);
    }
    if (name == "single_scattering") {
        return runSingleScattering(device, pipelines, outDir);
    }
    gfx::log("Unknown test: " + std::string(name));
    return 2;
}

} // namespace batch
