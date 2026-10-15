#pragma once

#include "atmosphere/AtmosphereRenderer.h"

namespace atmosphere {

/// 大気のパストレーサー（多重散乱の真値、V4 の比較対象）。フレームごとにサンプルを足して収束させる。
/// 偶数・奇数フレームを別の累積テクスチャに足し、2 つの差からノイズを見積もれるようにしている。
class PathTracer {
public:
    PathTracer(gfx::Device& device, const gfx::PipelineLibrary& pipelines);

    /// 累積テクスチャを作り直して累積をリセットする。GPU が空であること。
    void resize(uint32_t width, uint32_t height);
    /// 累積をリセットする（次の render から数え直す）。
    void reset() { m_frameIndex = 0; }

    /// 1 ディスパッチ分（チャンネルごとに `samplesPerDispatch` 本の経路）を追加し、平均を `outputUav` に書く。
    /// Transmittance LUT（太陽ディスク用）は `luts` のものを使う。
    void render(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereRenderer& luts,
                const AtmosphereParams& params, const SkyView& view, uint32_t maxDepth, uint32_t samplesPerDispatch,
                uint32_t outputUav);

    /// これまでに累積したサンプル数（チャンネルあたり）
    uint64_t samplesPerChannel() const { return m_samples; }
    gfx::Texture& accumulationA() { return m_accumulationA; }
    gfx::Texture& accumulationB() { return m_accumulationB; }

private:
    gfx::Device& m_device;
    const gfx::PipelineLibrary& m_pipelines;
    ComPtr<ID3D12PipelineState> m_pso;
    gfx::Texture m_accumulationA;  // RGBA32F: rgb = 単位照度あたり輝度の和、a = サンプル数
    gfx::Texture m_accumulationB;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_frameIndex = 0;
    uint64_t m_samples = 0;
};

} // namespace atmosphere
