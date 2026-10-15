#pragma once

#include "atmosphere/AtmosphereParams.h"
#include "atmosphere/SkyPassConstants.h"
#include "gfx/Pipeline.h"
#include "gfx/Texture.h"

#include <glm/mat4x4.hpp>

#include <optional>

namespace atmosphere {

/// 太陽方向の透過率の求め方（AtmosphereIntegrate.hlsli の SUN_TRANSMITTANCE_* と一致させる）。
enum class SunTransmittanceMode : uint32_t {
    Lut = 0,     // Transmittance LUT を引く（実用の方法）
    Nested = 1,  // サンプル点ごとに太陽方向へ入れ子レイマーチ（LUT を使わないリファレンス）
};

/// SkyPassConstants.hlsli の OUTPUT_* と一致させる。
enum class RaymarchOutput : uint32_t {
    CameraFb = 0,    // カメラ視点、露出済み fb
    TestPerLux = 1,  // 検証用: 正距円筒の方向グリッド、単位照度あたりの輝度 [sr⁻¹]
};

/// LUT の品質設定（ステップ数）と、多重散乱を入れるか。
struct AtmosphereQuality {
    // [POINT] Transmittance LUT の積分区間数。論文の PC 設定は 40 だが、Phase 2 の検証（V1/V2）で
    //         地平線付近の誤差が最大 1.8%（空の輝度で 1.4%）と分かった。LUT は大気パラメータの変更時しか
    //         作り直さないので、既定を 500（誤差 0.09%）にする（design.md §14）。
    uint32_t transmittanceSteps = 500;
    uint32_t multiScatteringSteps = 20;  // 論文の PC 設定（64 方向 × 20 区間）
    // [POINT] Multi-scattering LUT は光線上の区間を等間隔に置く（参照実装と同じ）。Sky-View LUT・レイマーチの
    //         2 次配置をここにも使うと、20 区間で最大誤差が 2.7% → 4.9% に悪化した（design.md §14 Phase 3）。
    //         全高度の点から下向きを含む全方向へ積分するので、濃い空気が光線の遠い側に来る光線が多く、
    //         カメラ側を細かくする 2 次配置が裏目に出ると考えている（理由は未検証）。
    bool multiScatteringUniformSteps = true;
    uint32_t skyViewSteps = 30;          // 論文の PC 設定
    bool multipleScattering = true;
};

/// リファレンスのレイマーチの設定。
struct RaymarchSettings {
    uint32_t primarySteps = 64;
    SunTransmittanceMode sunMode = SunTransmittanceMode::Lut;
    uint32_t nestedSteps = 64;
    bool multipleScattering = true;
};

/// 空を描くときの視点・出力。
struct SkyView {
    glm::mat4 invViewProjRot{1.0f};  // 回転のみのビュー × 射影 の逆行列（Reverse-Z）
    glm::vec3 cameraPosKm{0.0f};     // 惑星座標 [km]
    float exposureScale = 1.0f;      // 1/(1.2·2^EV100) [1/(cd/m²)]
    RaymarchOutput output = RaymarchOutput::CameraFb;
    uint32_t directionUav = gfx::kInvalidDescriptorIndex;  // TestPerLux のときの方向出力（不要なら無効値）
    bool sunDisk = true;
};

/// 大気の LUT 生成と空の描画。Shoga へ移植する単位。
/// D3D12 への依存は gfx の薄い層（コマンドリスト・Bindless インデックス）だけ。
class AtmosphereRenderer {
public:
    static constexpr uint32_t kTransmittanceWidth = 256;
    static constexpr uint32_t kTransmittanceHeight = 64;
    static constexpr uint32_t kMultiScatteringSize = 32;
    static constexpr uint32_t kSkyViewWidth = 192;
    static constexpr uint32_t kSkyViewHeight = 108;

    /// `lutFormat`: 実用は RGBA16F。数値検証では RGBA32F（格納の丸めを測らないため）。
    AtmosphereRenderer(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                       DXGI_FORMAT lutFormat = DXGI_FORMAT_R16G16B16A16_FLOAT);

    /// [POINT] Transmittance LUT は大気の媒質だけで決まる。媒質パラメータかステップ数が
    ///         前回と同じなら何もしない（作り直したら true）。LUT は終了時に全シェーダーから読める状態。
    bool updateTransmittanceLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                                uint32_t stepCount, uint32_t debugRMuUav = gfx::kInvalidDescriptorIndex);
    /// Multi-scattering LUT。媒質・ステップ数・Transmittance LUT のどれかが変わったときだけ作り直す。
    /// `uniformSteps`: 光線上の区間を等間隔に置く（既定）。false で 2 次配置（比較検証用）。
    bool updateMultiScatteringLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                  const AtmosphereParams& params, uint32_t stepCount,
                                  SunTransmittanceMode sunMode = SunTransmittanceMode::Lut, uint32_t nestedSteps = 0,
                                  uint32_t debugUav = gfx::kInvalidDescriptorIndex, bool uniformSteps = true);
    /// Sky-View LUT。カメラの高度と太陽の向きで変わるので毎フレーム作る。
    void updateSkyViewLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                          const glm::vec3& cameraPosKm, uint32_t stepCount, bool multipleScattering);
    /// 3 つの LUT をまとめて更新する（対話モード用）。
    void updateLuts(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                    const AtmosphereQuality& quality, const glm::vec3& cameraPosKm);

    /// Sky-View LUT から空を描く（実用パス）。`outputUav` は UNORDERED_ACCESS 状態であること。
    void renderSkyFromLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                          const SkyView& view, uint32_t outputUav, uint32_t width, uint32_t height);
    /// Sky-View LUT を使わずピクセルごとにレイマーチする（リファレンス）。
    void renderSkyRaymarch(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                           const SkyView& view, const RaymarchSettings& settings, uint32_t outputUav,
                           uint32_t width, uint32_t height);

    /// 空のパス共通の定数（パストレーサーも使う）。LUT のインデックス・サイズを埋める。
    SkyPassConstants makeSkyPassConstants(const AtmosphereParams& params, const SkyView& view, uint32_t outputUav,
                                          uint32_t width, uint32_t height) const;

    gfx::Texture& transmittanceLut() { return m_transmittanceLut; }
    gfx::Texture& multiScatteringLut() { return m_multiScatteringLut; }
    gfx::Texture& skyViewLut() { return m_skyViewLut; }
    /// Sky-View LUT の 2 枚目: Mie の単散乱から位相関数を除いた積分（描画時に p_M(θ) を掛ける）
    gfx::Texture& skyViewMieLut() { return m_skyViewMieLut; }

private:
    void dispatchSkyPass(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, ID3D12PipelineState* pso,
                         const SkyPassConstants& constants, uint32_t width, uint32_t height);

    const gfx::PipelineLibrary& m_pipelines;
    ComPtr<ID3D12PipelineState> m_transmittancePso;
    ComPtr<ID3D12PipelineState> m_multiScatteringPso;
    ComPtr<ID3D12PipelineState> m_skyViewPso;
    ComPtr<ID3D12PipelineState> m_skyFromLutPso;
    ComPtr<ID3D12PipelineState> m_raymarchPso;
    gfx::Texture m_transmittanceLut;
    gfx::Texture m_multiScatteringLut;
    gfx::Texture m_skyViewLut;     // Rayleigh の単散乱 + 多重散乱 [sr⁻¹]
    gfx::Texture m_skyViewMieLut;  // Mie の単散乱（位相関数なし）[無次元]

    std::optional<AtmosphereGpu> m_transmittanceParams;
    uint32_t m_transmittanceSteps = 0;
    uint64_t m_transmittanceVersion = 0;  // Transmittance LUT を作り直すたびに増える
    std::optional<AtmosphereGpu> m_multiScatteringParams;
    uint32_t m_multiScatteringSteps = 0;
    SunTransmittanceMode m_multiScatteringSunMode = SunTransmittanceMode::Lut;
    bool m_multiScatteringUniformSteps = false;
    uint64_t m_multiScatteringTransmittanceVersion = 0;
};

} // namespace atmosphere
