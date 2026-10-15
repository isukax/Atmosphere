#include "atmosphere/AtmosphereRenderer.h"

#include <glm/mat4x4.hpp>

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

struct SkyRaymarchConstants {
    AtmosphereGpu atmosphere;        // c0-c5
    float invViewProjRot[4][4];      // c6-c9（行優先）
    float cameraPosKm[3];            // c10.xyz
    uint32_t outputUav;              // c10.w
    float sunDirection[3];           // c11.xyz
    uint32_t transmittanceLutSrv;    // c11.w
    float sunIlluminanceLux[3];      // c12.xyz
    float exposureScale;             // c12.w
    uint32_t width;                  // c13.x
    uint32_t height;                 // c13.y
    uint32_t primarySteps;           // c13.z
    uint32_t sunTransmittanceMode;   // c13.w
    uint32_t nestedSteps;            // c14.x
    uint32_t outputMode;             // c14.y
    uint32_t directionUav;           // c14.z
    uint32_t pad14;                  // c14.w
    float transmittanceLutSize[2];   // c15.xy
    float pad15[2];
};
static_assert(offsetof(SkyRaymarchConstants, invViewProjRot) == 6 * 16);
static_assert(offsetof(SkyRaymarchConstants, cameraPosKm) == 10 * 16);
static_assert(offsetof(SkyRaymarchConstants, outputUav) == 10 * 16 + 12);
static_assert(offsetof(SkyRaymarchConstants, sunDirection) == 11 * 16);
static_assert(offsetof(SkyRaymarchConstants, sunIlluminanceLux) == 12 * 16);
static_assert(offsetof(SkyRaymarchConstants, width) == 13 * 16);
static_assert(offsetof(SkyRaymarchConstants, nestedSteps) == 14 * 16);
static_assert(offsetof(SkyRaymarchConstants, transmittanceLutSize) == 15 * 16);

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
                                       DXGI_FORMAT transmittanceFormat)
    : m_pipelines(pipelines),
      m_transmittancePso(pipelines.createCompute("TransmittanceLut_cs")),
      m_raymarchPso(pipelines.createCompute("SkyRaymarch_cs")),
      m_transmittanceLut(device, {kTransmittanceWidth, kTransmittanceHeight, transmittanceFormat, true, true, false,
                                  L"TransmittanceLut"})
{
}

bool AtmosphereRenderer::updateTransmittanceLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                                const AtmosphereParams& params, uint32_t stepCount,
                                                uint32_t debugRMuUav)
{
    const AtmosphereGpu gpu = toGpu(params);
    if (m_lutParams == gpu && m_lutSteps == stepCount && debugRMuUav == gfx::kInvalidDescriptorIndex) {
        return false;
    }
    m_lutParams = gpu;
    m_lutSteps = stepCount;

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

void AtmosphereRenderer::renderSkyRaymarch(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                           const AtmosphereParams& params, const RaymarchView& view,
                                           const RaymarchSettings& settings, uint32_t outputUav, uint32_t width,
                                           uint32_t height)
{
    SkyRaymarchConstants c{};
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
    c.primarySteps = settings.primarySteps;
    c.sunTransmittanceMode = static_cast<uint32_t>(settings.sunMode);
    c.nestedSteps = settings.nestedSteps;
    c.outputMode = static_cast<uint32_t>(view.output);
    c.directionUav = view.directionUav;
    c.transmittanceLutSize[0] = static_cast<float>(kTransmittanceWidth);
    c.transmittanceLutSize[1] = static_cast<float>(kTransmittanceHeight);

    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_raymarchPso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

} // namespace atmosphere
