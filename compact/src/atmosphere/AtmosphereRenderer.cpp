#include "atmosphere/AtmosphereRenderer.h"
#include "atmosphere/FixedData.h"
#include "atmosphere/MultiScatteringConstants.h"
#include "atmosphere/TransmittanceConstants.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/geometric.hpp>
#include <stdexcept>

namespace compact::atmosphere {
namespace {
void store(float (&dst)[3], const glm::vec3 &v) {
    std::memcpy(dst, &v.x, sizeof(dst));
}
void matrix(float (&dst)[4][4], const glm::mat4 &m) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            dst[r][c] = m[c][r];
}
bool finite(const glm::vec3 &v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
void require(bool value, const char *message) {
    if (!value)
        throw std::invalid_argument(message);
}
} // namespace
float defaultSolarIlluminanceLux() {
    return fixed::solarLux;
}
glm::vec3 AtmosphereRenderer::sunColor() const {
    return {fixed::sunColor[0], fixed::sunColor[1], fixed::sunColor[2]};
}

AtmosphereRenderer::AtmosphereRenderer(gfx::Device &device, const gfx::PipelineLibrary &pipelines, DXGI_FORMAT format)
    : m_device(device), m_pipelines(pipelines), m_format(format),
      m_transmittancePso(pipelines.createCompute("TransmittanceLut_cs")),
      m_transferPso(pipelines.createCompute("MultiScatteringTransfer_cs")),
      m_multiScatteringPso(pipelines.createCompute("MultiScatteringLut_cs")),
      m_skyPso(pipelines.createCompute("SkyViewLut_cs")), m_apPso(pipelines.createCompute("AerialPerspective_cs")),
      m_irradiancePso(pipelines.createCompute("SkyIrradiance_cs")),
      m_compositePso(pipelines.createCompute("SkyFromLut_cs")),
      m_phase(device, {sizeof(fixed::phase), 16, true, false, L"CAB phase"}),
      m_irradiance(device, {9 * 16, 16, true, true, L"Sky irradiance"}) {
    require(format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R32G32B32A32_FLOAT,
            "Unsupported LUT format");
    m_phase.upload(device, fixed::phase, sizeof(fixed::phase), D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    createResources();
}

void AtmosphereRenderer::createResources() {
    const auto &q = m_quality;
    auto texture = [&](gfx::Texture &target, uint32_t w, uint32_t h, const wchar_t *name, uint32_t d = 1,
                       DXGI_FORMAT f = DXGI_FORMAT_UNKNOWN) {
        if (target && target.desc().width == w && target.desc().height == h && target.desc().depth == d)
            return false;
        target = gfx::Texture(m_device, {w, h, f == DXGI_FORMAT_UNKNOWN ? m_format : f, true, true, false, name, d});
        return true;
    };
    if (texture(m_transmittance, q.transmittanceWidth, q.transmittanceHeight, L"Transmittance"))
        m_transmittanceDirty = m_multiScatteringDirty = true;
    texture(m_transfer, 17, q.multiScatteringSize, L"SH transfer", 1, DXGI_FORMAT_R32G32B32A32_FLOAT);
    if (texture(m_multiScattering, 6 * q.multiScatteringSize, q.multiScatteringSize, L"SH L2+ground"))
        m_multiScatteringDirty = true;
    texture(m_sky, q.skyViewWidth, q.skyViewHeight, L"Sky");
    texture(m_skyMie, q.skyViewWidth, q.skyViewHeight, L"Sky Mie");
    texture(m_ap, q.aerialPerspectiveSize, q.aerialPerspectiveSize, L"AP Rayleigh / mean T", q.aerialPerspectiveSize);
    texture(m_apMie, q.aerialPerspectiveSize, q.aerialPerspectiveSize, L"AP Mie", q.aerialPerspectiveSize);
    texture(m_apMs, q.aerialPerspectiveSize, q.aerialPerspectiveSize, L"AP multiple", q.aerialPerspectiveSize);
}

void AtmosphereRenderer::setSettings(const AtmosphereSettings &input) {
    require(std::isfinite(input.aod550) && input.aod550 >= 0.001f && input.aod550 <= 1.0f, "AOD must be in [0.001, 1]");
    require(finite(input.sunDirectionWs) && std::isfinite(glm::length(input.sunDirectionWs)) &&
                glm::length(input.sunDirectionWs) > 1e-6f,
            "Sun direction must be finite and nonzero");
    require(std::isfinite(input.sunIlluminanceLux) && input.sunIlluminanceLux >= 0,
            "Solar illuminance must be finite and nonnegative");
    require(finite(input.groundAlbedo) && glm::all(glm::greaterThanEqual(input.groundAlbedo, glm::vec3(0))) &&
                glm::all(glm::lessThanEqual(input.groundAlbedo, glm::vec3(1))),
            "Albedo must be in [0,1]");
    require(std::isfinite(input.sunAngularRadiusRad) && input.sunAngularRadiusRad >= 0.0008f &&
                input.sunAngularRadiusRad <= 0.1f,
            "Sun radius must be in [0.0008,0.1] rad");
    if (input.aod550 != m_settings.aod550)
        m_transmittanceDirty = m_multiScatteringDirty = true;
    if (input.groundAlbedo != m_settings.groundAlbedo)
        m_multiScatteringDirty = true;
    m_settings = input;
    // [POINT] Preserve already-unit float vectors: a needless normalization can shift hard shadow edges.
    if (std::abs(glm::length(input.sunDirectionWs) - 1.0f) > 1e-6f)
        m_settings.sunDirectionWs = glm::normalize(input.sunDirectionWs);
}

void AtmosphereRenderer::setQuality(const AtmosphereQuality &q) {
    require((q.transmittanceWidth == 128 && q.transmittanceHeight == 32) ||
                (q.transmittanceWidth == 256 && q.transmittanceHeight == 64) ||
                (q.transmittanceWidth == 512 && q.transmittanceHeight == 128),
            "Invalid transmittance size");
    require(q.multiScatteringSize == 16 || q.multiScatteringSize == 32 || q.multiScatteringSize == 64,
            "Invalid MS size");
    require((q.skyViewWidth == 96 && q.skyViewHeight == 54) || (q.skyViewWidth == 192 && q.skyViewHeight == 108) ||
                (q.skyViewWidth == 384 && q.skyViewHeight == 216),
            "Invalid sky size");
    require(q.aerialPerspectiveSize == 16 || q.aerialPerspectiveSize == 32 || q.aerialPerspectiveSize == 64,
            "Invalid AP size");
    require(q.transmittanceSteps >= 4 && q.transmittanceSteps <= 1000 && q.multiScatteringSteps >= 4 &&
                q.multiScatteringSteps <= 256 && q.skyViewSteps >= 4 && q.skyViewSteps <= 256 && q.apMaxSteps >= 4 &&
                q.apMaxSteps <= 256 && q.perPixelSteps >= 4 && q.perPixelSteps <= 128 && q.bisectionSteps <= 12,
            "Invalid integration quality");
    const auto old = m_quality;
    const bool resize = q.transmittanceWidth != old.transmittanceWidth ||
                        q.transmittanceHeight != old.transmittanceHeight ||
                        q.multiScatteringSize != old.multiScatteringSize || q.skyViewWidth != old.skyViewWidth ||
                        q.skyViewHeight != old.skyViewHeight || q.aerialPerspectiveSize != old.aerialPerspectiveSize;
    if (resize)
        m_device.waitIdle();
    m_quality = q;
    if (resize)
        createResources();
    if (q.transmittanceSteps != old.transmittanceSteps)
        m_transmittanceDirty = m_multiScatteringDirty = true;
    if (q.multiScatteringSteps != old.multiScatteringSteps)
        m_multiScatteringDirty = true;
}

AtmosphereGpu AtmosphereRenderer::gpuParameters() const {
    AtmosphereGpu a{};
    a.bottomRadiusKm = 6360;
    a.topRadiusKm = 6460;
    a.rayleighScaleHeightKm = 8;
    a.mieScaleHeightKm = 1.2f;
    std::memcpy(a.rayleighScatteringPerKm, fixed::rayleigh, sizeof(a.rayleighScatteringPerKm));
    std::memcpy(a.ozoneAbsorptionPerKm, fixed::ozone, sizeof(a.ozoneAbsorptionPerKm));
    const float scale = m_settings.aod550 / 0.1f;
    for (int i = 0; i < 3; ++i) {
        a.mieScatteringPerKm[i] = fixed::mieScattering[i] * scale;
        a.mieExtinctionPerKm[i] = a.mieScatteringPerKm[i] + fixed::mieAbsorption[i] * scale;
    }
    a.ozoneCenterKm = 25;
    a.ozoneHalfWidthKm = 15;
    a.sunAngularRadiusRad = m_settings.sunAngularRadiusRad;
    a.miePhaseTableSrv = m_phase.srvIndex();
    store(a.groundAlbedo, m_settings.groundAlbedo);
    return a;
}

SkyPassConstants AtmosphereRenderer::constants(const AtmosphereFrameInput &view, uint32_t output, uint32_t w,
                                               uint32_t h) const {
    SkyPassConstants c{};
    const auto &q = m_quality;
    c.atmosphere = gpuParameters();
    matrix(c.invViewProjRot, view.invViewProjRot);
    store(c.cameraPosKm, worldToPlanetKm(view.cameraPosM));
    c.outputUav = output;
    store(c.sunDirection, m_settings.sunDirectionWs);
    c.transmittanceLutSrv = m_transmittance.srvIndex();
    store(c.sunIlluminanceLux, m_settings.sunIlluminanceLux * sunColor());
    c.exposureScale = view.exposureScale;
    c.width = w;
    c.height = h;
    c.multiScatteringLutSrv = m_multiScattering.srvIndex();
    c.transmittanceLutSize[0] = float(q.transmittanceWidth);
    c.transmittanceLutSize[1] = float(q.transmittanceHeight);
    c.multiScatteringLutSize[0] = c.multiScatteringLutSize[1] = float(q.multiScatteringSize);
    c.skyViewLutSrv = m_sky.srvIndex();
    c.skyViewMieLutUav = m_skyMie.uavIndex();
    c.skyViewMieLutSrv = m_skyMie.srvIndex();
    c.skyIrradianceUav = m_irradiance.uavIndex();
    c.skyViewLutSize[0] = float(q.skyViewWidth);
    c.skyViewLutSize[1] = float(q.skyViewHeight);
    c.aerialPerspectiveUav = m_ap.uavIndex();
    c.aerialPerspectiveMieUav = m_apMie.uavIndex();
    c.aerialPerspectiveMsUav = m_apMs.uavIndex();
    c.aerialPerspectiveSrv = m_ap.srvIndex();
    c.aerialPerspectiveMieSrv = m_apMie.srvIndex();
    c.aerialPerspectiveMsSrv = m_apMs.srvIndex();
    c.depthSrv = view.depthSrv;
    c.apSliceCount = float(q.aerialPerspectiveSize);
    c.apKmPerSlice = 128.0f / c.apSliceCount;
    c.apMaxSteps = q.apMaxSteps;
    matrix(c.shadowFarToUvz, view.shadow.cameraRelativeToUvz);
    matrix(c.shadowNearToUvz, view.shadow.nearCameraRelativeToUvz);
    c.shadowFarSrv = view.shadow.mapSrv;
    c.shadowNearSrv = view.shadow.valid() ? view.shadow.nearMapSrv : gfx::kInvalidDescriptorIndex;
    c.apPerPixelSteps = q.perPixelSteps;
    c.apPerPixelBisectSteps = q.bisectionSteps;
    std::memcpy(c.msPhaseMoment0, fixed::moments[0], 16);
    std::memcpy(c.msPhaseMoment1, fixed::moments[1], 16);
    std::memcpy(c.msPhaseMoment2, fixed::moments[2], 16);
    return c;
}
void AtmosphereRenderer::dispatch(ID3D12GraphicsCommandList *cmd, gfx::UploadRing &upload, ID3D12PipelineState *pso,
                                  const SkyPassConstants &c, uint32_t x, uint32_t y, uint32_t z) {
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(pso);
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch(x, y, z);
}

AtmosphereRenderer::Updates AtmosphereRenderer::updateLuts(ID3D12GraphicsCommandList *cmd, gfx::UploadRing &upload,
                                                           const AtmosphereFrameInput &view) {
    const Updates updated{m_transmittanceDirty, m_multiScatteringDirty};
    const auto &q = m_quality;
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    if (m_transmittanceDirty) {
        TransmittanceConstants c{};
        c.atmosphere = gpuParameters();
        c.outputUav = m_transmittance.uavIndex();
        c.width = q.transmittanceWidth;
        c.height = q.transmittanceHeight;
        c.stepCount = q.transmittanceSteps;
        m_transmittance.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->SetPipelineState(m_transmittancePso.Get());
        cmd->SetComputeRootConstantBufferView(0, upload.push(c));
        cmd->Dispatch((c.width + 7) / 8, (c.height + 7) / 8, 1);
        m_transmittance.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        m_transmittanceDirty = false;
    }
    if (m_multiScatteringDirty) {
        MultiScatteringConstants c{};
        c.atmosphere = gpuParameters();
        c.outputUav = m_multiScattering.uavIndex();
        c.width = c.height = q.multiScatteringSize;
        c.stepCount = q.multiScatteringSteps;
        c.transmittanceLutSize[0] = float(q.transmittanceWidth);
        c.transmittanceLutSize[1] = float(q.transmittanceHeight);
        c.transmittanceLutSrv = m_transmittance.srvIndex();
        c.transferUav = m_transfer.uavIndex();
        c.transferSrv = m_transfer.srvIndex();
        std::memcpy(c.phaseMoment0, fixed::moments[0], 16);
        std::memcpy(c.phaseMoment1, fixed::moments[1], 16);
        std::memcpy(c.phaseMoment2, fixed::moments[2], 16);
        m_transfer.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->SetPipelineState(m_transferPso.Get());
        cmd->SetComputeRootConstantBufferView(0, upload.push(c));
        cmd->Dispatch(q.multiScatteringSize, 1, 1);
        m_transfer.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        m_multiScattering.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmd->SetPipelineState(m_multiScatteringPso.Get());
        cmd->SetComputeRootConstantBufferView(0, upload.push(c));
        cmd->Dispatch(q.multiScatteringSize, q.multiScatteringSize, 1);
        m_multiScattering.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        m_multiScatteringDirty = false;
    }
    auto c = constants(view, m_sky.uavIndex(), q.skyViewWidth, q.skyViewHeight);
    c.stepCount = q.skyViewSteps;
    m_sky.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_skyMie.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch(cmd, upload, m_skyPso.Get(), c, (c.width + 7) / 8, (c.height + 7) / 8, 1);
    m_sky.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_skyMie.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    return updated;
}
void AtmosphereRenderer::updateAerialPerspective(ID3D12GraphicsCommandList *cmd, gfx::UploadRing &upload,
                                                 const AtmosphereFrameInput &view) {
    const auto n = m_quality.aerialPerspectiveSize;
    const auto c = constants(view, gfx::kInvalidDescriptorIndex, n, n);
    for (auto *t : {&m_ap, &m_apMie, &m_apMs})
        t->transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch(cmd, upload, m_apPso.Get(), c, n / 4, n / 4, n / 4);
    for (auto *t : {&m_ap, &m_apMie, &m_apMs})
        t->transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}
void AtmosphereRenderer::updateSkyIrradiance(ID3D12GraphicsCommandList *cmd, gfx::UploadRing &upload,
                                             const AtmosphereFrameInput &view) {
    m_irradiance.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch(cmd, upload, m_irradiancePso.Get(), constants(view, gfx::kInvalidDescriptorIndex, 1, 1), 1, 1, 1);
    m_irradiance.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}
void AtmosphereRenderer::composite(ID3D12GraphicsCommandList *cmd, gfx::UploadRing &upload,
                                   const AtmosphereFrameInput &view, uint32_t output, uint32_t w, uint32_t h) {
    dispatch(cmd, upload, m_compositePso.Get(), constants(view, output, w, h), (w + 7) / 8, (h + 7) / 8, 1);
}
} // namespace compact::atmosphere
