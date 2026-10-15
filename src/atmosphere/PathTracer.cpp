#include "atmosphere/PathTracer.h"

namespace atmosphere {

PathTracer::PathTracer(gfx::Device& device, const gfx::PipelineLibrary& pipelines)
    : m_device(device), m_pipelines(pipelines), m_pso(pipelines.createCompute("PathTrace_cs"))
{
}

void PathTracer::resize(uint32_t width, uint32_t height)
{
    if (width == m_width && height == m_height && m_accumulationA) {
        reset();
        return;
    }
    m_width = width;
    m_height = height;
    // [POINT] 累積は float32。数千サンプルを足すので、FP16 では和の精度が足りない。
    m_accumulationA = gfx::Texture(m_device, {width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, true, true, false,
                                              L"PathTraceAccumulationA"});
    m_accumulationB = gfx::Texture(m_device, {width, height, DXGI_FORMAT_R32G32B32A32_FLOAT, true, true, false,
                                              L"PathTraceAccumulationB"});
    reset();
}

void PathTracer::render(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereRenderer& luts,
                        const AtmosphereParams& params, const SkyView& view, uint32_t maxDepth,
                        uint32_t samplesPerDispatch, uint32_t outputUav)
{
    if (m_frameIndex == 0) {
        m_samples = 0;
    }
    SkyPassConstants c = luts.makeSkyPassConstants(params, view, outputUav, m_width, m_height);
    c.frameIndex = m_frameIndex;
    c.maxDepth = maxDepth;
    c.samplesPerDispatch = samplesPerDispatch;
    c.accumulationUavA = m_accumulationA.uavIndex();
    c.accumulationUavB = m_accumulationB.uavIndex();

    m_accumulationA.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_accumulationB.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_pso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch((m_width + 7) / 8, (m_height + 7) / 8, 1);
    ++m_frameIndex;
    m_samples += samplesPerDispatch;
}

} // namespace atmosphere
