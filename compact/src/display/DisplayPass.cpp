#include "display/DisplayPass.h"

#include <cstddef>

namespace compact::display {
namespace {

/// HLSL の `row_major float3x3` 1 行 = 16 バイト（float3 + 余り 4 バイト）。
struct alignas(16) CbRow {
    float v[3];
    float pad;
};

/// Display.hlsl の cbuffer DisplayConstants と同じ並び。
struct DisplayConstants {
    CbRow workingToRec2020[3];  // c0
    CbRow rec2020ToRec709[3];   // c3
    CbRow rec709ToRec2020[3];   // c6
    uint32_t sceneColorIndex;   // c9.x
    uint32_t uiIndex;           // c9.y
    uint32_t outputEncode;      // c9.z
    uint32_t toneMapOperator;   // c9.w
    float exposureRescale;      // c10.x
    float uiWhiteNits;          // c10.y
    uint32_t frameIndex;        // c10.z
    float pad10;                // c10.w
    float gt7TargetFb;          // c11.x
    float gt7SdrCorrection;     // c11.y
    float gt7MidPoint;          // c11.z
    float gt7LinearSection;     // c11.w
    float gt7ToeStrength;       // c12.x
    float gt7K;                 // c12.y
    float gt7KA;                // c12.z
    float gt7TargetUcsI;        // c12.w
    float gt7BlendRatio;        // c13.x
    float gt7FadeStart;         // c13.y
    float gt7FadeEnd;           // c13.z
    float pad13;                // c13.w
};

// [POINT] HLSL 側の packoffset と 1 対 1 で照合する（cN.x = N*16 バイト目）。
static_assert(offsetof(DisplayConstants, rec2020ToRec709) == 3 * 16);
static_assert(offsetof(DisplayConstants, rec709ToRec2020) == 6 * 16);
static_assert(offsetof(DisplayConstants, sceneColorIndex) == 9 * 16);
static_assert(offsetof(DisplayConstants, toneMapOperator) == 9 * 16 + 12);
static_assert(offsetof(DisplayConstants, exposureRescale) == 10 * 16);
static_assert(offsetof(DisplayConstants, frameIndex) == 10 * 16 + 8);
static_assert(offsetof(DisplayConstants, gt7TargetFb) == 11 * 16);
static_assert(offsetof(DisplayConstants, gt7ToeStrength) == 12 * 16);
static_assert(offsetof(DisplayConstants, gt7BlendRatio) == 13 * 16);
static_assert(sizeof(DisplayConstants) == 14 * 16);

void store(CbRow (&dst)[3], const Mat3& src)
{
    for (int r = 0; r < 3; ++r) {
        dst[r] = {{src.m[r][0], src.m[r][1], src.m[r][2]}, 0.0f};
    }
}

} // namespace

DisplayPass::DisplayPass(const gfx::PipelineLibrary& pipelines)
    : m_pipelines(pipelines)
{
}

ID3D12PipelineState* DisplayPass::pso(DXGI_FORMAT format)
{
    auto it = m_psos.find(format);
    if (it == m_psos.end()) {
        it = m_psos.emplace(format, m_pipelines.createFullscreen("Display_vs", "Display_ps", format)).first;
    }
    return it->second.Get();
}

void DisplayPass::record(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, D3D12_CPU_DESCRIPTOR_HANDLE rtv,
                         DXGI_FORMAT rtvFormat, uint32_t width, uint32_t height, const DisplayPassInput& input)
{
    DisplayConstants c{};
    store(c.workingToRec2020, workingToRec2020(input.workingSpace));
    store(c.rec2020ToRec709, kRec2020ToRec709);
    store(c.rec709ToRec2020, kRec709ToRec2020);
    c.sceneColorIndex = input.sceneColorSrv;
    c.uiIndex = input.uiSrv;
    c.outputEncode = static_cast<uint32_t>(input.encode);
    c.toneMapOperator = static_cast<uint32_t>(input.toneMap);
    c.exposureRescale = input.exposureRescale;
    c.uiWhiteNits = input.uiWhiteNits;
    c.frameIndex = input.frameIndex;
    c.gt7TargetFb = input.gt7.targetFb;
    c.gt7SdrCorrection = input.gt7.sdrCorrection;
    c.gt7MidPoint = input.gt7.midPoint;
    c.gt7LinearSection = input.gt7.linearSection;
    c.gt7ToeStrength = input.gt7.toeStrength;
    c.gt7K = input.gt7.k;
    c.gt7KA = input.gt7.kA;
    c.gt7TargetUcsI = input.gt7.targetUcsI;
    c.gt7BlendRatio = input.gt7.blendRatio;
    c.gt7FadeStart = input.gt7.fadeStart;
    c.gt7FadeEnd = input.gt7.fadeEnd;

    cmd->SetGraphicsRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(pso(rtvFormat));
    cmd->SetGraphicsRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);
    cmd->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd->DrawInstanced(3, 1, 0, 0);
}

} // namespace display
