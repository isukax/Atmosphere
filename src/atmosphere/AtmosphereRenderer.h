#pragma once

#include "atmosphere/AtmosphereParams.h"
#include "atmosphere/ShadowInput.h"
#include "atmosphere/SkyPassConstants.h"
#include "gfx/Buffer.h"
#include "gfx/Pipeline.h"
#include "gfx/Texture.h"

#include <glm/mat4x4.hpp>

#include <optional>
#include <stdexcept>
#include <string>

namespace atmosphere {

/// 多重散乱の近似（design.md §5.11）。シェーダーの版 MS_SH = 0 / 1（AtmosphereCommon.hlsli）に対応する。
enum class MultiScatteringModel : uint32_t {
    Isotropic = 0,   // Hillaire 2020 の等方の Ψ_ms（Multi-scattering LUT は RGB 1 つ）
    ShL2Ground = 1,  // 入ってくる散乱光の場を l ≤ 2 の球面調和（6 係数）で持つ。閉じ方に地表との往復を含める
};

/// 媒質を扱うパスの、シェーダーの 2 つの版（MIE_GENERAL = 0 / 1、CMakeLists.txt・AtmosphereCommon.hlsli）。
/// [POINT] 1 成分・背景なしのプリセットで成分 1 の分岐を残すと GPU 時間が 10〜30% 増えたので、版を分けて
///         媒質に合わせて選ぶ（usesGeneralMieMedium）。どちらの版でも結果は同じ（0 の版が扱える媒質では）
///
/// 多重散乱を扱うパス（Multi-scattering / Sky-View LUT・空気遠近法・レイマーチ）は、多重散乱の近似の 2 つの版
/// （MS_SH = 0 / 1、*_sh_cs / *_mie_sh_cs）も持つ（`shVariants`）。
struct MediumPso {
    MediumPso(const gfx::PipelineLibrary& pipelines, const std::string& pass, bool shVariants = false)
        : basic(pipelines.createCompute((pass + "_cs").c_str())),
          general(pipelines.createCompute((pass + "_mie_cs").c_str()))
    {
        if (shVariants) {
            basicSh = pipelines.createCompute((pass + "_sh_cs").c_str());
            generalSh = pipelines.createCompute((pass + "_mie_sh_cs").c_str());
        }
    }
    ID3D12PipelineState* get(const AtmosphereParams& params,
                             MultiScatteringModel model = MultiScatteringModel::Isotropic) const
    {
        const bool generalMie = usesGeneralMieMedium(params);
        if (model == MultiScatteringModel::ShL2Ground) {
            if (!basicSh) {
                throw std::runtime_error("MediumPso: this pass has no spherical-harmonics variant");
            }
            return generalMie ? generalSh.Get() : basicSh.Get();
        }
        return generalMie ? general.Get() : basic.Get();
    }

    ComPtr<ID3D12PipelineState> basic;      // 1 成分・背景なし（Phase 4.8 と同じコード）
    ComPtr<ID3D12PipelineState> general;    // 2 成分・自由大気の背景
    ComPtr<ID3D12PipelineState> basicSh;    // 多重散乱の SH 近似の版（shVariants のときだけ）
    ComPtr<ID3D12PipelineState> generalSh;
};

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

/// 空気遠近法の透過率の求め方（SkyPassConstants.hlsli の AP_TRANSMITTANCE_* と一致させる）。design.md §5.7。
enum class AerialPerspectiveTransmittanceMode : uint32_t {
    VolumeMean = 0,  // ボリュームの A（RGB の平均。論文・参照実装と同じ近似）
    LutRatio = 1,    // Transmittance LUT の比（RGB）
    VolumeRgb = 2,   // ボリュームに RGB の透過率を持つ（もう 1 枚の 3D テクスチャ）
};

/// LUT の品質設定（ステップ数）と、多重散乱を入れるか。
struct AtmosphereQuality {
    // [POINT] Transmittance LUT の積分区間数。論文の PC 設定は 40 だが、Phase 2 の検証（V1/V2）で
    //         地平線付近の誤差が最大 1.8%（空の輝度で 1.4%）と分かった。LUT は大気パラメータの変更時しか
    //         作り直さないので、既定を 500（誤差 0.09%）にする（design.md §14）。
    uint32_t transmittanceSteps = 500;
    // [POINT] Multi-scattering LUT は 40 区間・2 次配置（design.md §12 Q8、2026-10-18 に更新）。
    //         論文の PC 設定（20 区間・等間隔）は、今の大気（AOD ≈ 0.005）では最大誤差 2.7% で足りたが、
    //         エアロゾルのプリセット（AOD 0.1）では 24〜34% になり、空の見た目で ΔE_ITP 0.5〜4.7 ずれた。
    //         地表付近の濃い層を、カメラ側（LUT の点の側）を細かくする 2 次配置で拾うと、40 区間で ΔE ≤ 0.05、
    //         今の大気でも最大誤差 1.3% になる（§14 Phase 3・Phase 4.8）。LUT は媒質の変更時しか作らない。
    uint32_t multiScatteringSteps = 40;
    bool multiScatteringUniformSteps = false;
    uint32_t skyViewSteps = 30;          // 論文の PC 設定
    bool multipleScattering = true;
    // [POINT] 多重散乱の近似（design.md §5.11、§12 Q28）。SH の版は地上・低い太陽・夕方の空と、地面を照らした
    //         画面を大きく改善するが、LUT の作り直しが重い（媒質を変えたときだけ）
    MultiScatteringModel multiScatteringModel = MultiScatteringModel::Isotropic;
};

/// リファレンスのレイマーチの設定。
struct RaymarchSettings {
    uint32_t primarySteps = 64;
    SunTransmittanceMode sunMode = SunTransmittanceMode::Lut;
    uint32_t nestedSteps = 64;
    bool multipleScattering = true;
    bool meanTransmittance = false;  // 地形に掛ける透過率を RGB 平均にする（V3c の比較用）
};

/// 空を描くときの視点・出力。
struct SkyView {
    glm::mat4 invViewProjRot{1.0f};  // 回転のみのビュー × 射影 の逆行列（Reverse-Z）
    glm::vec3 cameraPosKm{0.0f};     // 惑星座標 [km]
    float exposureScale = 1.0f;      // 1/(1.2·2^EV100) [1/(cd/m²)]
    RaymarchOutput output = RaymarchOutput::CameraFb;
    uint32_t directionUav = gfx::kInvalidDescriptorIndex;  // TestPerLux のときの方向出力（不要なら無効値）
    bool sunDisk = true;
    uint32_t depthSrv = gfx::kInvalidDescriptorIndex;  // 深度バッファ（地形があるとき）。空気遠近法を掛ける
    /// 地形の影（空気遠近法と、レイマーチの地形のピクセルの単散乱に掛ける）。既定は影なし
    ShadowInput shadow;
    /// 空気遠近法の影の方式（design.md §5.6）。updateAerialPerspective と renderSkyFromLut で同じ値を渡すこと
    AerialPerspectiveShadowOptions apShadow;
    /// 空気遠近法の透過率の求め方（design.md §5.7）。renderSkyFromLut だけが使う。
    /// [POINT] 既定はボリュームの RGB（方式 2）。RGB 平均（方式 0）は遠景で R/B が ±6%（高い視点で B 13%）ずれ、
    ///         Transmittance LUT の比（方式 1）は地平線付近で崩れた（V3h、design.md §14 Phase 4.6）。
    AerialPerspectiveTransmittanceMode apTransmittance = AerialPerspectiveTransmittanceMode::VolumeRgb;
    /// 検証用（V3h）: 地形のピクセルに、合成の代わりに透過率（RGB）を書く。renderSkyFromLut / renderSkyRaymarch
    bool writeApTransmittance = false;
};

/// 大気の LUT 生成と空の描画。Shoga へ移植する単位。
/// D3D12 への依存は gfx の薄い層（コマンドリスト・Bindless インデックス）だけ。
class AtmosphereRenderer {
public:
    static constexpr uint32_t kTransmittanceWidth = 256;
    static constexpr uint32_t kTransmittanceHeight = 64;
    static constexpr uint32_t kMultiScatteringSize = 32;
    static constexpr uint32_t kMultiScatteringShCoefficients = 6;  // l ≤ 2 の鏡映対称の球面調和（design.md §5.11）
    static constexpr uint32_t kMultiScatteringShTransferCount = 17;  // F の表の 1 行（F の 3 つのブロック + 地表の G）
    static constexpr uint32_t kSkyViewWidth = 192;
    static constexpr uint32_t kSkyViewHeight = 108;
    static constexpr uint32_t kAerialPerspectiveSize = 32;   // 32×32×32
    static constexpr float kAerialPerspectiveKmPerSlice = 4.0f;  // スライス座標 1 あたりの距離（参照実装と同じ）

    /// `lutFormat`: 実用は RGBA16F。数値検証では RGBA32F（格納の丸めを測らないため）。
    AtmosphereRenderer(gfx::Device& device, const gfx::PipelineLibrary& pipelines,
                       DXGI_FORMAT lutFormat = DXGI_FORMAT_R16G16B16A16_FLOAT);

    /// [POINT] Transmittance LUT は大気の媒質だけで決まる。媒質パラメータかステップ数が
    ///         前回と同じなら何もしない（作り直したら true）。LUT は終了時に全シェーダーから読める状態。
    bool updateTransmittanceLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                                uint32_t stepCount, uint32_t debugRMuUav = gfx::kInvalidDescriptorIndex);
    /// Multi-scattering LUT。媒質・ステップ数・Transmittance LUT・近似のモデルのどれかが変わったときだけ作り直す。
    /// `uniformSteps`: true で光線上の区間を等間隔に置く（引数の既定。比較検証用）、false で 2 次配置
    /// （実用の設定 AtmosphereQuality の既定）。
    /// `model`: 多重散乱の近似。これ以降の Sky-View LUT・空気遠近法・レイマーチは、最後に作った LUT のモデルの版で描く。
    bool updateMultiScatteringLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                  const AtmosphereParams& params, uint32_t stepCount,
                                  SunTransmittanceMode sunMode = SunTransmittanceMode::Lut, uint32_t nestedSteps = 0,
                                  uint32_t debugUav = gfx::kInvalidDescriptorIndex, bool uniformSteps = true,
                                  MultiScatteringModel model = MultiScatteringModel::Isotropic);
    /// Sky-View LUT。カメラの高度と太陽の向きで変わるので毎フレーム作る。
    void updateSkyViewLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                          const glm::vec3& cameraPosKm, uint32_t stepCount, bool multipleScattering);
    /// 3 つの LUT をまとめて更新する（対話モード用）。
    void updateLuts(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                    const AtmosphereQuality& quality, const glm::vec3& cameraPosKm);
    /// 空気遠近法のボリューム（カメラの視錐台）。毎フレーム作る。Sky-View LUT と同じく LUT 3 枚の後に呼ぶ。
    void updateAerialPerspective(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                 const AtmosphereParams& params, const SkyView& view, bool multipleScattering);
    /// 空からの拡散照度（SH 係数）。Sky-View LUT の後に呼ぶ。
    void updateSkyIrradiance(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                             const glm::vec3& cameraPosKm);

    /// Sky-View LUT から空を描く（実用パス）。`outputUav` は UNORDERED_ACCESS 状態であること。
    void renderSkyFromLut(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                          const SkyView& view, uint32_t outputUav, uint32_t width, uint32_t height);
    /// Sky-View LUT を使わずピクセルごとにレイマーチする（リファレンス）。
    void renderSkyRaymarch(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, const AtmosphereParams& params,
                           const SkyView& view, const RaymarchSettings& settings, uint32_t outputUav,
                           uint32_t width, uint32_t height);

    /// 次の update で Transmittance / Multi-scattering LUT を必ず作り直させる（性能計測用）。
    void invalidateLuts()
    {
        m_transmittanceParams.reset();
        m_multiScatteringParams.reset();
    }

    /// 空のパス共通の定数（パストレーサーも使う）。LUT のインデックス・サイズを埋める。
    SkyPassConstants makeSkyPassConstants(const AtmosphereParams& params, const SkyView& view, uint32_t outputUav,
                                          uint32_t width, uint32_t height) const;

    gfx::Texture& transmittanceLut() { return m_transmittanceLut; }
    gfx::Texture& multiScatteringLut() { return m_multiScatteringLut; }
    /// 多重散乱の SH 近似の係数のアトラス（192×32 = 32×32 のタイル 6 枚、RGB。design.md §5.11）
    gfx::Texture& multiScatteringShLut() { return m_multiScatteringShLut; }
    /// 最後に作った Multi-scattering LUT のモデル（描画のパスはこの版を使う）
    MultiScatteringModel multiScatteringModel() const { return m_multiScatteringModel; }
    gfx::Texture& skyViewLut() { return m_skyViewLut; }
    /// Sky-View LUT の 2 枚目: Mie の単散乱から位相関数を除いた積分（描画時に p_M(θ) を掛ける）
    gfx::Texture& skyViewMieLut() { return m_skyViewMieLut; }
    /// Sky-View LUT の 3 枚目: Mie の成分 1（2 成分のプリセットだけ。design.md §5.9）
    gfx::Texture& skyViewMie1Lut() { return m_skyViewMie1Lut; }
    gfx::Texture& aerialPerspective() { return m_aerialPerspective; }
    gfx::Texture& aerialPerspectiveMie() { return m_aerialPerspectiveMie; }
    gfx::Texture& aerialPerspectiveMie1() { return m_aerialPerspectiveMie1; }
    gfx::Texture& aerialPerspectiveMultiple() { return m_aerialPerspectiveMs; }
    gfx::Texture& aerialPerspectiveTransmittance() { return m_aerialPerspectiveT; }
    /// 空からの照度の SH 係数（9 × float4、単位照度あたり、余弦の畳み込み済み）
    gfx::Buffer& skyIrradiance() { return m_skyIrradiance; }

private:
    void dispatchSkyPass(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, ID3D12PipelineState* pso,
                         const SkyPassConstants& constants, uint32_t width, uint32_t height);

    const gfx::PipelineLibrary& m_pipelines;
    MediumPso m_transmittancePso;
    MediumPso m_multiScatteringPso;
    MediumPso m_skyViewPso;
    MediumPso m_skyFromLutPso;
    MediumPso m_raymarchPso;
    MediumPso m_aerialPerspectivePso;
    MediumPso m_skyIrradiancePso;
    // SH の Multi-scattering LUT の F の表（MultiScatteringLut.hlsl の CSTransfer）。1 成分・背景なし / 一般形
    ComPtr<ID3D12PipelineState> m_multiScatteringShTransferPso;
    ComPtr<ID3D12PipelineState> m_multiScatteringShTransferMiePso;
    gfx::Texture m_transmittanceLut;
    gfx::Texture m_multiScatteringLut;
    gfx::Texture m_multiScatteringShLut;  // SH の係数のアトラス（MultiScatteringModel::ShL2Ground）
    gfx::Texture m_multiScatteringShTransfer;  // SH の版の F の表（17×32、行 = 高度。CSTransfer → CSMain）
    gfx::Texture m_skyViewLut;     // Rayleigh の単散乱 + 多重散乱 [sr⁻¹]
    gfx::Texture m_skyViewMieLut;  // Mie の単散乱（位相関数なし）[無次元]
    gfx::Texture m_skyViewMie1Lut; // Mie の成分 1 の同じ量（2 成分のプリセットだけ書く。design.md §5.9）
    gfx::Texture m_aerialPerspective;     // 3D: Rayleigh の単散乱の内散乱 [sr⁻¹]、A = 平均透過率
    gfx::Texture m_aerialPerspectiveMie;  // 3D: Mie の単散乱（位相関数なし）[無次元]
    gfx::Texture m_aerialPerspectiveMie1; // 3D: Mie の成分 1 の同じ量（2 成分のプリセットだけ書く）
    gfx::Texture m_aerialPerspectiveMs;   // 3D: 多重散乱の内散乱 [sr⁻¹]（影の対象外。design.md §5.6）
    gfx::Texture m_aerialPerspectiveT;    // 3D: 透過率（RGB）[無次元]（design.md §5.7 の方式 2）
    gfx::Buffer m_skyIrradiance;          // SH 係数 9 × float4

    std::optional<AtmosphereGpu> m_transmittanceParams;
    uint32_t m_transmittanceSteps = 0;
    uint64_t m_transmittanceVersion = 0;  // Transmittance LUT を作り直すたびに増える
    std::optional<AtmosphereGpu> m_multiScatteringParams;
    uint32_t m_multiScatteringSteps = 0;
    SunTransmittanceMode m_multiScatteringSunMode = SunTransmittanceMode::Lut;
    bool m_multiScatteringUniformSteps = false;
    uint64_t m_multiScatteringTransmittanceVersion = 0;
    MultiScatteringModel m_multiScatteringModel = MultiScatteringModel::Isotropic;
};

} // namespace atmosphere
