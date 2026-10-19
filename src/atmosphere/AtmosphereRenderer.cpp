#include "atmosphere/AtmosphereRenderer.h"

#include <cstddef>

namespace atmosphere {
namespace {

struct TransmittanceLutConstants {
    AtmosphereGpu atmosphere;  // c0-c9
    uint32_t outputUav;        // c10.x
    uint32_t width;            // c10.y
    uint32_t height;           // c10.z
    uint32_t stepCount;        // c10.w
    uint32_t debugRMuUav;      // c11.x
    uint32_t pad[3];
};
static_assert(offsetof(TransmittanceLutConstants, outputUav) == 10 * 16);
static_assert(offsetof(TransmittanceLutConstants, debugRMuUav) == 11 * 16);

struct MultiScatteringLutConstants {
    AtmosphereGpu atmosphere;           // c0-c9
    uint32_t outputUav;                 // c10.x
    uint32_t width;                     // c10.y
    uint32_t height;                    // c10.z
    uint32_t stepCount;                 // c10.w
    uint32_t transmittanceLutSrv;       // c11.x
    uint32_t sunTransmittanceMode;      // c11.y
    uint32_t nestedSteps;               // c11.z
    uint32_t debugUav;                  // c11.w
    float transmittanceLutSize[2];      // c12.xy
    uint32_t uniformSteps;              // c12.z
    float pad;
};
static_assert(offsetof(MultiScatteringLutConstants, outputUav) == 10 * 16);
static_assert(offsetof(MultiScatteringLutConstants, transmittanceLutSrv) == 11 * 16);
static_assert(offsetof(MultiScatteringLutConstants, transmittanceLutSize) == 12 * 16);
static_assert(offsetof(MultiScatteringLutConstants, uniformSteps) == 12 * 16 + 8);

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
      m_transmittancePso(pipelines, "TransmittanceLut"),
      m_multiScatteringPso(pipelines, "MultiScatteringLut"),
      m_skyViewPso(pipelines, "SkyViewLut"),
      m_skyFromLutPso(pipelines, "SkyFromLut"),
      m_raymarchPso(pipelines, "SkyRaymarch"),
      m_aerialPerspectivePso(pipelines, "AerialPerspective"),
      m_skyIrradiancePso(pipelines, "SkyIrradiance"),
      m_transmittanceLut(device, {kTransmittanceWidth, kTransmittanceHeight, lutFormat, true, true, false,
                                  L"TransmittanceLut"}),
      m_multiScatteringLut(device, {kMultiScatteringSize, kMultiScatteringSize, lutFormat, true, true, false,
                                    L"MultiScatteringLut"}),
      m_skyViewLut(device, {kSkyViewWidth, kSkyViewHeight, lutFormat, true, true, false, L"SkyViewLut"}),
      m_skyViewMieLut(device, {kSkyViewWidth, kSkyViewHeight, lutFormat, true, true, false, L"SkyViewMieLut"}),
      m_skyViewMie1Lut(device, {kSkyViewWidth, kSkyViewHeight, lutFormat, true, true, false, L"SkyViewMie1Lut"}),
      m_aerialPerspective(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                   L"AerialPerspective", kAerialPerspectiveSize}),
      m_aerialPerspectiveMie(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                      L"AerialPerspectiveMie", kAerialPerspectiveSize}),
      m_aerialPerspectiveMie1(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                       L"AerialPerspectiveMie1", kAerialPerspectiveSize}),
      m_aerialPerspectiveMs(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                     L"AerialPerspectiveMultiple", kAerialPerspectiveSize}),
      m_aerialPerspectiveT(device, {kAerialPerspectiveSize, kAerialPerspectiveSize, lutFormat, true, true, false,
                                    L"AerialPerspectiveTransmittance", kAerialPerspectiveSize}),
      m_skyIrradiance(device, {9 * 16, 16, true, true, L"SkyIrradianceSH"})
{
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
    cmd->SetPipelineState(m_transmittancePso.get(params));
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
    cmd->SetPipelineState(m_multiScatteringPso.get(params));
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
    // [POINT] Mie の成分 1 の LUT・ボリュームは常に確保し、記述子も常に渡す。書き込み・読み出しはシェーダーが
    //         成分の数（gAtmosphere.mieComponentCount）で分岐する（1 成分の描画と GPU 時間は変わらない）
    c.skyViewMie1LutUav = m_skyViewMie1Lut.uavIndex();
    c.skyViewMie1LutSrv = m_skyViewMie1Lut.srvIndex();
    c.skyViewLutSize[0] = static_cast<float>(kSkyViewWidth);
    c.skyViewLutSize[1] = static_cast<float>(kSkyViewHeight);
    c.flags = (view.sunDisk ? kSkyFlagSunDisk : 0u) | (view.writeApTransmittance ? kSkyFlagWriteApTransmittance : 0u);
    c.apTransmittanceMode = static_cast<uint32_t>(view.apTransmittance);
    c.sunTransmittanceMode = static_cast<uint32_t>(SunTransmittanceMode::Lut);
    c.accumulationUavA = gfx::kInvalidDescriptorIndex;
    c.accumulationUavB = gfx::kInvalidDescriptorIndex;
    c.skyIrradianceUav = m_skyIrradiance.uavIndex();
    c.aerialPerspectiveUav = m_aerialPerspective.uavIndex();
    c.aerialPerspectiveMieUav = m_aerialPerspectiveMie.uavIndex();
    c.aerialPerspectiveSrv = m_aerialPerspective.srvIndex();
    c.aerialPerspectiveMieSrv = m_aerialPerspectiveMie.srvIndex();
    c.aerialPerspectiveMie1Uav = m_aerialPerspectiveMie1.uavIndex();
    c.aerialPerspectiveMie1Srv = m_aerialPerspectiveMie1.srvIndex();
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
    c.aerialPerspectiveTransmittanceUav = m_aerialPerspectiveT.uavIndex();
    c.aerialPerspectiveTransmittanceSrv = m_aerialPerspectiveT.srvIndex();
    c.pathTraceMediumSrv = gfx::kInvalidDescriptorIndex;  // 表の媒質は PathTracer が設定する
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
    m_skyViewMie1Lut.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatchSkyPass(cmd, upload, m_skyViewPso.get(params), c, kSkyViewWidth, kSkyViewHeight);
    m_skyViewLut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_skyViewMieLut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_skyViewMie1Lut.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
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
    dispatchSkyPass(cmd, upload, m_skyFromLutPso.get(params), makeSkyPassConstants(params, view, outputUav, width, height),
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
    dispatchSkyPass(cmd, upload, m_raymarchPso.get(params), c, width, height);
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
    m_aerialPerspectiveMie1.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_aerialPerspectiveMs.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_aerialPerspectiveT.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_aerialPerspectivePso.get(params));
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    constexpr uint32_t groups = kAerialPerspectiveSize / 4;  // numthreads(4, 4, 4)
    cmd->Dispatch(groups, groups, groups);
    m_aerialPerspective.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_aerialPerspectiveMie.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_aerialPerspectiveMie1.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_aerialPerspectiveMs.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_aerialPerspectiveT.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}

void AtmosphereRenderer::updateSkyIrradiance(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                             const AtmosphereParams& params, const glm::vec3& cameraPosKm)
{
    SkyView view;
    view.cameraPosKm = cameraPosKm;
    const SkyPassConstants c = makeSkyPassConstants(params, view, gfx::kInvalidDescriptorIndex, 1, 1);
    m_skyIrradiance.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_skyIrradiancePso.get(params));
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch(1, 1, 1);
    m_skyIrradiance.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}

} // namespace atmosphere
