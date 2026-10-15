#pragma once

#include "atmosphere/AtmosphereParams.h"
#include "gfx/Pipeline.h"
#include "gfx/Texture.h"

#include <glm/mat4x4.hpp>

#include <optional>

namespace atmosphere {

/// 太陽方向の透過率の求め方（SkyRaymarch.hlsl の SUN_TRANSMITTANCE_* と一致させる）。
enum class SunTransmittanceMode : uint32_t {
    Lut = 0,     // Transmittance LUT を引く（実用の方法）
    Nested = 1,  // サンプル点ごとに太陽方向へ入れ子レイマーチ（LUT を使わないリファレンス）
};

/// SkyRaymarch.hlsl の OUTPUT_* と一致させる。
enum class RaymarchOutput : uint32_t {
    CameraFb = 0,     // カメラ視点、露出済み fb
    TestPerLux = 1,   // 検証用: 正距円筒の方向グリッド、単位照度あたりの輝度 [sr⁻¹]
};

struct RaymarchSettings {
    uint32_t primarySteps = 64;
    SunTransmittanceMode sunMode = SunTransmittanceMode::Lut;
    uint32_t nestedSteps = 64;
};

struct RaymarchView {
    glm::mat4 invViewProjRot{1.0f};  // 回転のみのビュー × 射影 の逆行列（Reverse-Z）
    glm::vec3 cameraPosKm{0.0f};     // 惑星座標 [km]
    float exposureScale = 1.0f;      // 1/(1.2·2^EV100) [1/(cd/m²)]
    RaymarchOutput output = RaymarchOutput::CameraFb;
    uint32_t directionUav = gfx::kInvalidDescriptorIndex;  // TestPerLux のときの方向出力
};

/// 大気の LUT 生成と描画（Phase 2: Transmittance LUT + 単散乱レイマーチ）。
/// Shoga へ移植する単位。D3D12 への依存は gfx の薄い層（コマンドリスト・Bindless インデックス）だけ。
class AtmosphereRenderer {
public:
    static constexpr uint32_t kTransmittanceWidth = 256;
    static constexpr uint32_t kTransmittanceHeight = 64;

    /// `transmittanceFormat`: 実用は RGBA16F。数値検証では RGBA32F（格納の丸めを測らないため）。
    AtmosphereRenderer(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                       DXGI_FORMAT transmittanceFormat = DXGI_FORMAT_R16G16B16A16_FLOAT);

    /// [POINT] Transmittance LUT は大気の媒質だけで決まる。媒質パラメータかステップ数が
    ///         前回と同じなら何もしない（作り直したら true）。LUT は終了時に全シェーダーから読める状態。
    bool updateTransmittanceLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                                uint32_t stepCount, uint32_t debugRMuUav = gfx::kInvalidDescriptorIndex);

    /// 単散乱のレイマーチ。`outputUav` は UNORDERED_ACCESS 状態であること。
    void renderSkyRaymarch(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                           const RaymarchView& view, const RaymarchSettings& settings, uint32_t outputUav,
                           uint32_t width, uint32_t height);

    gfx::Texture& transmittanceLut() { return m_transmittanceLut; }

private:
    const gfx::PipelineLibrary& m_pipelines;
    ComPtr<ID3D12PipelineState> m_transmittancePso;
    ComPtr<ID3D12PipelineState> m_raymarchPso;
    gfx::Texture m_transmittanceLut;
    std::optional<AtmosphereGpu> m_lutParams;
    uint32_t m_lutSteps = 0;
};

} // namespace atmosphere
