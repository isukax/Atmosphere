#pragma once

#include "gfx/Device.h"

#include <filesystem>
#include <vector>

namespace gfx {

/// 全パス共通のルートシグネチャと、.cso からの PSO 生成。
///
/// [POINT] ルートシグネチャは 1 種類だけ:
///           [0] ルート CBV (b0) … パスごとの定数（Bindless インデックスもここに入れる）
///           静的サンプラー s0 = 線形/クランプ, s1 = ポイント/クランプ
///         テクスチャ・バッファは `ResourceDescriptorHeap[index]` で引くのでテーブルは持たない。
///         Shoga の「Bindless 前提・最小 API」に合わせた形。
class PipelineLibrary {
public:
    static constexpr UINT kRootPassConstants = 0;

    PipelineLibrary(Device& device, std::filesystem::path shaderDir);

    ID3D12RootSignature* rootSignature() const { return m_rootSignature.Get(); }

    ComPtr<ID3D12PipelineState> createCompute(const char* csName) const;
    /// 全画面三角形（頂点バッファなし）を 1 枚の RT に描く PSO。深度なし・ブレンドなし。
    ComPtr<ID3D12PipelineState> createFullscreen(const char* vsName, const char* psName, DXGI_FORMAT rtvFormat) const;

private:
    std::vector<uint8_t> load(const char* name) const;

    Device& m_device;
    std::filesystem::path m_shaderDir;
    ComPtr<ID3D12RootSignature> m_rootSignature;
};

} // namespace gfx
