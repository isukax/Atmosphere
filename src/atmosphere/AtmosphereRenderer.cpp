#include "atmosphere/AtmosphereRenderer.h"

#include <cstddef>

namespace atmosphere {
namespace {

struct TransmittanceLutConstants {
    AtmosphereGpu atmosphere;  // c0-c5
    uint32_t outputUav;        // c6.x
    uint32_t width;            // c6.y
    uint32_t height;           // c6.z
    uint32_t stepCount;        // c6.w
    uint32_t debugRMuUav;      // c7.x
    uint32_t pad[3];
};
static_assert(offsetof(TransmittanceLutConstants, outputUav) == 6 * 16);
static_assert(offsetof(TransmittanceLutConstants, debugRMuUav) == 7 * 16);

struct MultiScatteringLutConstants {
    AtmosphereGpu atmosphere;           // c0-c5
    uint32_t outputUav;                 // c6.x
    uint32_t width;                     // c6.y
    uint32_t height;                    // c6.z
    uint32_t stepCount;                 // c6.w
    uint32_t transmittanceLutSrv;       // c7.x
    uint32_t sunTransmittanceMode;      // c7.y
    uint32_t nestedSteps;               // c7.z
    uint32_t debugUav;                  // c7.w
    float transmittanceLutSize[2];      // c8.xy
    uint32_t uniformSteps;              // c8.z
    float pad;
};
static_assert(offsetof(MultiScatteringLutConstants, outputUav) == 6 * 16);
static_assert(offsetof(MultiScatteringLutConstants, transmittanceLutSrv) == 7 * 16);
static_assert(offsetof(MultiScatteringLutConstants, transmittanceLutSize) == 8 * 16);
static_assert(offsetof(MultiScatteringLutConstants, uniformSteps) == 8 * 16 + 8);

void store(float (&dst)[3], const glm::vec3& v)
{
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
}

/// glm（列優先）→ HLSL の row_major float4x4（行優先）
void storeRowMajor(float (&dst)[4][4], const glm::mat4& m)
{
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            dst[row][col] = m[col][row];
        }
    }
}

} // namespace

AtmosphereRenderer::AtmosphereRenderer(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                                       DXGI_FORMAT lutFormat)
    : m_pipelines(pipelines),
      m_transmittancePso(pipelines.createCompute("TransmittanceLut_cs")),
      m_multiScatteringPso(pipelines.createCompute("MultiScatteringLut_cs")),
      m_skyViewPso(pipelines.createCompute("SkyViewLut_cs")),
      m_skyFromLutPso(pipelines.createCompute("SkyFromLut_cs")),
      m_raymarchPso(pipelines.createCompute("SkyRaymarch_cs")),
      m_transmittanceLut(device, {kTransmittanceWidth, kTransmittanceHeight, lutFormat, true, true, false,
                                  L"TransmittanceLut"}),
      m_multiScatteringLut(device, {kMultiScatteringSize, kMultiScatteringSize, lutFormat, true, true, false,
                                    L"MultiScatteringLut"}),
      m_skyViewLut(device, {kSkyViewWidth, kSkyViewHeight, lutFormat, true, true, false, L"SkyViewLut"}),
      m_skyViewMieLut(device, {kSkyViewWidth, kSkyViewHeight, lutFormat, true, true, false, L"SkyViewMieLut"}),
      m_aerialPerspective(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                   L"AerialPerspective", kAerialPerspectiveSize}),
      m_aerialPerspectiveMie(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                      L"AerialPerspectiveMie", kAerialPerspectiveSize}),
      m_aerialPerspectiveMs(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                     L"AerialPerspectiveMultiple", kAerialPerspectiveSize}),
      m_skyIrradiance(device, {9 * 16, 16, true, true, L"SkyIrradianceSH"})
{
    m_aerialPerspectivePso = pipelines.createCompute("AerialPerspective_cs");
    m_skyIrradiancePso = pipelines.createCompute("SkyIrradiance_cs");
}

bool AtmosphereRenderer::updateTransmittanceLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                                const AtmosphereParams& params, uint32_t stepCount,
                                                uint32_t debugRMuUav)
{
    const AtmosphereGpu gpu = toGpu(params);
    if (m_transmittanceParams == gpu && m_transmittanceSteps == stepCount &&
        debugRMuUav == gfx::kInvalidDescriptorIndex) {
        return false;
    }
    m_transmittanceParams = gpu;
    m_transmittanceSteps = stepCount;
    ++m_transmittanceVersion;

    TransmittanceLutConstants c{};
    c.atmosphere = gpu;
    c.outputUav = m_transmittanceLut.uavIndex();
    c.width = kTransmittanceWidth;
    c.height = kTransmittanceHeight;
    c.stepCount = stepCount;
    c.debugRMuUav = debugRMuUav;

    m_transmittanceLut.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_transmittancePso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch((kTransmittanceWidth + 7) / 8, (kTransmittanceHeight + 7) / 8, 1);
    m_transmittanceLut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    return true;
}

bool AtmosphereRenderer::updateMultiScatteringLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                                  const AtmosphereParams& params, uint32_t stepCount,
                                                  SunTransmittanceMode sunMode, uint32_t nestedSteps,
                                                  uint32_t debugUav, bool uniformSteps)
{
    const AtmosphereGpu gpu = toGpu(params);
    // [POINT] Multi-scattering LUT は媒質と地表アルベドで決まり、太陽方向の透過率に Transmittance LUT を使う。
    //         どちらかが変わったときだけ作り直す（毎フレームの更新は不要）。
    if (m_multiScatteringParams == gpu && m_multiScatteringSteps == stepCount && m_multiScatteringSunMode == sunMode &&
        m_multiScatteringUniformSteps == uniformSteps &&
        m_multiScatteringTransmittanceVersion == m_transmittanceVersion && debugUav == gfx::kInvalidDescriptorIndex) {
        return false;
    }
    m_multiScatteringParams = gpu;
    m_multiScatteringSteps = stepCount;
    m_multiScatteringSunMode = sunMode;
    m_multiScatteringUniformSteps = uniformSteps;
    m_multiScatteringTransmittanceVersion = m_transmittanceVersion;

    MultiScatteringLutConstants c{};
    c.atmosphere = gpu;
    c.outputUav = m_multiScatteringLut.uavIndex();
    c.width = kMultiScatteringSize;
    c.height = kMultiScatteringSize;
    c.stepCount = stepCount;
    c.transmittanceLutSrv = m_transmittanceLut.srvIndex();
    c.sunTransmittanceMode = static_cast<uint32_t>(sunMode);
    c.nestedSteps = nestedSteps;
    c.debugUav = debugUav;
    c.transmittanceLutSize[0] = static_cast<float>(kTransmittanceWidth);
    c.transmittanceLutSize[1] = static_cast<float>(kTransmittanceHeight);
    c.uniformSteps = uniformSteps ? 1u : 0u;

    m_multiScatteringLut.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_multiScatteringPso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    // 1 テクセル = 1 スレッドグループ（グループ内 64 スレッド = 64 方向）
    cmd->Dispatch(kMultiScatteringSize, kMultiScatteringSize, 1);
    m_multiScatteringLut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    return true;
}

SkyPassConstants AtmosphereRenderer::makeSkyPassConstants(const AtmosphereParams& params, const SkyView& view,
                                                          uint32_t outputUav, uint32_t width, uint32_t height) const
{
    SkyPassConstants c{};
    c.atmosphere = toGpu(params);
    storeRowMajor(c.invViewProjRot, view.invViewProjRot);
    store(c.cameraPosKm, view.cameraPosKm);
    c.outputUav = outputUav;
    store(c.sunDirection, params.sunDirection());
    c.transmittanceLutSrv = m_transmittanceLut.srvIndex();
    // [POINT] 太陽の照度 [lx] × 太陽の色（Y = 1 の RGB）= RGB の照度。Y 成分が E_sun になる。
    store(c.sunIlluminanceLux, params.sunIlluminanceLux * params.sunColor);
    c.exposureScale = view.exposureScale;
    c.width = width;
    c.height = height;
    c.outputMode = static_cast<uint32_t>(view.output);
    c.directionUav = view.directionUav;
    c.multiScatteringLutSrv = m_multiScatteringLut.srvIndex();
    c.transmittanceLutSize[0] = static_cast<float>(kTransmittanceWidth);
    c.transmittanceLutSize[1] = static_cast<float>(kTransmittanceHeight);
    c.multiScatteringLutSize[0] = static_cast<float>(kMultiScatteringSize);
    c.multiScatteringLutSize[1] = static_cast<float>(kMultiScatteringSize);
    c.skyViewLutSrv = m_skyViewLut.srvIndex();
    c.skyViewMieLutUav = m_skyViewMieLut.uavIndex();
    c.skyViewMieLutSrv = m_skyViewMieLut.srvIndex();
    c.skyViewLutSize[0] = static_cast<float>(kSkyViewWidth);
    c.skyViewLutSize[1] = static_cast<float>(kSkyViewHeight);
    c.flags = view.sunDisk ? kSkyFlagSunDisk : 0u;
    c.sunTransmittanceMode = static_cast<uint32_t>(SunTransmittanceMode::Lut);
    c.accumulationUavA = gfx::kInvalidDescriptorIndex;
    c.accumulationUavB = gfx::kInvalidDescriptorIndex;
    c.skyIrradianceUav = m_skyIrradiance.uavIndex();
    c.aerialPerspectiveUav = m_aerialPerspective.uavIndex();
    c.aerialPerspectiveMieUav = m_aerialPerspectiveMie.uavIndex();
    c.aerialPerspectiveSrv = m_aerialPerspective.srvIndex();
    c.aerialPerspectiveMieSrv = m_aerialPerspectiveMie.srvIndex();
    c.depthSrv = view.depthSrv;
    c.apKmPerSlice = kAerialPerspectiveKmPerSlice;
    c.apSliceCount = static_cast<float>(kAerialPerspectiveSize);
    c.raymarchMeanTransmittance = 0;
    storeRowMajor(c.shadowFarToUvz, view.shadow.cameraRelativeToUvz);
    c.shadowFarSrv = view.shadow.mapSrv;
    storeRowMajor(c.shadowNearToUvz, view.shadow.nearCameraRelativeToUvz);
    c.shadowNearSrv = view.shadow.valid() ? view.shadow.nearMapSrv : gfx::kInvalidDescriptorIndex;
    c.apShadowMethod = static_cast<uint32_t>(view.apShadow.method);
    c.apPerPixelSteps = view.apShadow.perPixelSteps;
    c.apShadowFilterScale = view.apShadow.filterScale;
    c.apPerPixelBisectSteps = view.apShadow.perPixelBisectSteps;
    c.aerialPerspectiveMsUav = m_aerialPerspectiveMs.uavIndex();
    c.aerialPerspectiveMsSrv = m_aerialPerspectiveMs.srvIndex();
    return c;
}

void AtmosphereRenderer::dispatchSkyPass(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                         ID3D12PipelineState* pso, const SkyPassConstants& constants, uint32_t width,
                                         uint32_t height)
{
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(pso);
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(constants));
    cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

void AtmosphereRenderer::updateSkyViewLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                          const AtmosphereParams& params, const glm::vec3& cameraPosKm,
                                          uint32_t stepCount, bool multipleScattering)
{
    SkyView view;
    view.cameraPosKm = cameraPosKm;
    SkyPassConstants c = makeSkyPassConstants(params, view, m_skyViewLut.uavIndex(), kSkyViewWidth, kSkyViewHeight);
    c.stepCount = stepCount;
    c.flags = multipleScattering ? kSkyFlagMultipleScattering : 0u;
    m_skyViewLut.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_skyViewMieLut.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatchSkyPass(cmd, upload, m_skyViewPso.Get(), c, kSkyViewWidth, kSkyViewHeight);
    m_skyViewLut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_skyViewMieLut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}

void AtmosphereRenderer::updateLuts(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                    const AtmosphereParams& params, const AtmosphereQuality& quality,
                                    const glm::vec3& cameraPosKm)
{
    updateTransmittanceLut(cmd, upload, params, quality.transmittanceSteps);
    updateMultiScatteringLut(cmd, upload, params, quality.multiScatteringSteps, SunTransmittanceMode::Lut, 0,
                             gfx::kInvalidDescriptorIndex, quality.multiScatteringUniformSteps);
    updateSkyViewLut(cmd, upload, params, cameraPosKm, quality.skyViewSteps, quality.multipleScattering);
}

void AtmosphereRenderer::renderSkyFromLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                          const AtmosphereParams& params, const SkyView& view, uint32_t outputUav,
                                          uint32_t width, uint32_t height)
{
    dispatchSkyPass(cmd, upload, m_skyFromLutPso.Get(), makeSkyPassConstants(params, view, outputUav, width, height),
                    width, height);
}

void AtmosphereRenderer::renderSkyRaymarch(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                           const AtmosphereParams& params, const SkyView& view,
                                           const RaymarchSettings& settings, uint32_t outputUav, uint32_t width,
                                           uint32_t height)
{
    SkyPassConstants c = makeSkyPassConstants(params, view, outputUav, width, height);
    c.stepCount = settings.primarySteps;
    c.sunTransmittanceMode = static_cast<uint32_t>(settings.sunMode);
    c.nestedSteps = settings.nestedSteps;
    if (settings.multipleScattering) {
        c.flags |= kSkyFlagMultipleScattering;
    }
    c.raymarchMeanTransmittance = settings.meanTransmittance ? 1u : 0u;
    dispatchSkyPass(cmd, upload, m_raymarchPso.Get(), c, width, height);
}

void AtmosphereRenderer::updateAerialPerspective(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                                 const AtmosphereParams& params, const SkyView& view,
                                                 bool multipleScattering)
{
    SkyPassConstants c = makeSkyPassConstants(params, view, gfx::kInvalidDescriptorIndex, kAerialPerspectiveSize,
                                              kAerialPerspectiveSize);
    c.flags = multipleScattering ? kSkyFlagMultipleScattering : 0u;
    m_aerialPerspective.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_aerialPerspectiveMie.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_aerialPerspectiveMs.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_aerialPerspectivePso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    constexpr uint32_t groups = kAerialPerspectiveSize / 4;  // numthreads(4, 4, 4)
    cmd->Dispatch(groups, groups, groups);
    m_aerialPerspective.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_aerialPerspectiveMie.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_aerialPerspectiveMs.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}

void AtmosphereRenderer::updateSkyIrradiance(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                             const AtmosphereParams& params, const glm::vec3& cameraPosKm)
{
    SkyView view;
    view.cameraPosKm = cameraPosKm;
    const SkyPassConstants c = makeSkyPassConstants(params, view, gfx::kInvalidDescriptorIndex, 1, 1);
    m_skyIrradiance.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_skyIrradiancePso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch(1, 1, 1);
    m_skyIrradiance.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}

} // namespace atmosphere
