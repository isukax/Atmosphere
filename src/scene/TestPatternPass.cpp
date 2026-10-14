#include "scene/TestPatternPass.h"

#include <cstddef>

namespace scene {
namespace {

struct TestPatternConstants {
    uint32_t outputIndex;    // c0.x
    uint32_t width;          // c0.y
    uint32_t height;         // c0.z
    uint32_t pad0;           // c0.w
    float exposureScale;     // c1.x
    float log10MinNits;      // c1.y
    float log10MaxNits;      // c1.z
    float pad1;              // c1.w
    float luminanceWeights[3];  // c2.xyz
    float pad2;
};
static_assert(offsetof(TestPatternConstants, exposureScale) == 1 * 16);
static_assert(offsetof(TestPatternConstants, luminanceWeights) == 2 * 16);

} // namespace

TestPatternPass::TestPatternPass(const gfx::PipelineLibrary& pipelines)
    : m_pipelines(pipelines), m_pso(pipelines.createCompute("TestPattern_cs"))
{
}

void TestPatternPass::record(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, uint32_t sceneColorUav,
                             uint32_t width, uint32_t height, float exposureScale,
                             display::WorkingColorSpace workingSpace, const TestPatternSettings& settings)
{
    const glm::vec3 weights = display::luminanceWeights(workingSpace);
    TestPatternConstants c{};
    c.outputIndex = sceneColorUav;
    c.width = width;
    c.height = height;
    c.exposureScale = exposureScale;
    c.log10MinNits = settings.log10MinNits;
    c.log10MaxNits = settings.log10MaxNits;
    c.luminanceWeights[0] = weights.x;
    c.luminanceWeights[1] = weights.y;
    c.luminanceWeights[2] = weights.z;

    cmd->SetComputeRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_pso.Get());
    cmd->SetComputeRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

} // namespace scene
