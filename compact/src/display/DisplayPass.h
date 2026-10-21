#pragma once

#include "display/ColorSpace.h"
#include "display/GT7ToneMapping.h"
#include "gfx/Pipeline.h"

#include <unordered_map>

namespace compact::display {

/// 表示パスの出力形式（Display.hlsl の OUTPUT_* と一致させる）。
enum class OutputEncode : uint32_t {
    SdrSrgb = 0,        // R8G8B8A8_UNORM、sRGB、Rec.709
    Hdr10Pq = 1,        // R10G10B10A2_UNORM、PQ、Rec.2020
    LinearRec2020 = 2,  // 検証用: GT7 の生出力（R32G32B32A32_FLOAT）
};

/// トーンマップ演算子（Display.hlsl の TONEMAP_* と一致させる）。
enum class ToneMapOperator : uint32_t {
    GT7 = 0,
    None = 1,  // 検証用（PQ 単体の検証）
};

struct DisplayPassInput {
    uint32_t sceneColorSrv = gfx::kInvalidDescriptorIndex;
    uint32_t uiSrv = gfx::kInvalidDescriptorIndex;
    WorkingColorSpace workingSpace = WorkingColorSpace::Rec709;
    OutputEncode encode = OutputEncode::SdrSrgb;
    ToneMapOperator toneMap = ToneMapOperator::GT7;
    GT7Constants gt7{};
    float exposureRescale = 1.0f;
    float uiWhiteNits = 200.0f;
    uint32_t frameIndex = 0;
};

/// シーンカラー → GT7 → エンコード → 出力 RT。全画面三角形 1 回。
class DisplayPass {
public:
    explicit DisplayPass(const gfx::PipelineLibrary& pipelines);

    /// `rtv` は `rtvFormat` の RT。呼び出し側で RENDER_TARGET 状態にしておくこと。
    /// シーンカラーと UI は PIXEL_SHADER_RESOURCE 状態であること。
    void record(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, D3D12_CPU_DESCRIPTOR_HANDLE rtv,
                DXGI_FORMAT rtvFormat, uint32_t width, uint32_t height, const DisplayPassInput& input);

private:
    ID3D12PipelineState* pso(DXGI_FORMAT format);

    const gfx::PipelineLibrary& m_pipelines;
    std::unordered_map<DXGI_FORMAT, ComPtr<ID3D12PipelineState>> m_psos;
};

} // namespace display
