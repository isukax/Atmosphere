#pragma once

#include "display/ColorSpace.h"
#include "gfx/Pipeline.h"

namespace scene {

struct TestPatternSettings {
    float log10MinNits = -1.0f;  // 左端の輝度 10^-1 = 0.1 cd/m²
    float log10MaxNits = 6.0f;   // 右端の輝度 10^6 cd/m²
};

/// Phase 1 用のテストパターン（物理輝度の対数ランプ）をシーンカラーに書くコンピュートパス。
class TestPatternPass {
public:
    explicit TestPatternPass(const gfx::PipelineLibrary& pipelines);

    /// `sceneColorUav` は UNORDERED_ACCESS 状態であること。
    /// `exposureScale` は 1/(1.2·2^EV100) [1/(cd/m²)]。
    void record(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, uint32_t sceneColorUav, uint32_t width,
                uint32_t height, float exposureScale, display::WorkingColorSpace workingSpace,
                const TestPatternSettings& settings);

private:
    const gfx::PipelineLibrary& m_pipelines;
    ComPtr<ID3D12PipelineState> m_pso;
};

} // namespace scene
