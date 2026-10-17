#pragma once

#include "atmosphere/ShadowInput.h"
#include "gfx/Texture.h"
#include "render/SceneRenderer.h"
#include "scene/Camera.h"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace render {

struct ShadowSettings {
    bool surface = true;            // 地形・球の直達光に影を掛ける
    bool aerialPerspective = true;  // 空気遠近法の単散乱に影を掛ける
    bool nearCascade = true;        // 近景のカスケードを使う（切ると全体だけ）
    float nearRangeM = 500.0f;      // 近景のカスケードが覆う視錐台の奥行き [m]
    float normalOffsetTexels = 1.0f;  // 法線オフセット（テクセル数。テクセルの小さい方の辺に掛ける、design.md §14 Phase 4.5）
    float blendBand = 0.1f;         // 近景の端で全体と混ぜる帯の幅（近景の半幅に対する割合）
};

/// 1 段のカスケード（正射影）。光源空間 = 太陽へ向かう方向を Z とする右手系 [m]。
/// [POINT] シャドウマップの深度は光源に近いほど大きい（Reverse。クリア値 0 = 遮蔽物なし）:
///           u = (x − x_min) / (x_max − x_min)、v = (y_max − y) / (y_max − y_min)、深度 = (z − z_min) / (z_max − z_min)
struct ShadowCascade {
    uint32_t resolution = 0;
    glm::dvec3 axisX{1.0, 0.0, 0.0};
    glm::dvec3 axisY{0.0, 1.0, 0.0};
    glm::dvec3 axisZ{0.0, 0.0, 1.0};  // 太陽へ向かう方向
    glm::dvec3 minLight{0.0};         // 光源空間の範囲 [m]
    glm::dvec3 maxLight{1.0};
    glm::vec3 originM{0.0f};          // 描画時の原点（頂点はこの点からの相対位置で変換する）

    /// 1 テクセルの大きさ [m]（光源空間の x, y）
    glm::dvec2 texelSizeM() const;
    /// カメラ相対の位置 [m] → (u, v, 深度)
    glm::mat4 cameraRelativeToUvz(const glm::vec3& cameraPosM) const;
    /// 原点相対の位置 [m] → クリップ座標（シャドウマップを描く頂点シェーダー用）
    glm::mat4 originRelativeToClip() const;
};

/// 太陽のシャドウマップ（2 段のカスケード）。design.md §5.5。
///   近景（0）: 視錐台の 0〜nearRangeM を囲む球。毎フレーム、テクセル単位でスナップして描く
///   全体（1）: 全物体を囲む箱。太陽の向きが変わったときだけ描き直す
class ShadowRenderer {
public:
    static constexpr uint32_t kNearResolution = 2048;
    static constexpr uint32_t kFarResolution = 4096;

    explicit ShadowRenderer(gfx::Device& device);

    /// 全体のカスケード。太陽の向きが前回と違うか `force` なら描き直す（描いたら true）。
    bool updateFar(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, SceneRenderer& scene,
                   const glm::vec3& sunDirection, bool force = false);
    /// 近景のカスケード（毎フレーム）。全体のカスケードの後に呼ぶ（奥行きの範囲を全体に合わせるため）。
    void updateNear(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, SceneRenderer& scene,
                    const scene::Camera& camera, float aspect, const ShadowSettings& settings);

    /// 地表の影の入力（settings.surface が false なら影なし）
    SceneShadow sceneShadow(const glm::vec3& cameraPosM, const ShadowSettings& settings) const;
    /// 空気遠近法の影の入力（全体と近景のカスケード。settings.aerialPerspective が false なら影なし）
    atmosphere::ShadowInput atmosphereShadow(const glm::vec3& cameraPosM, const ShadowSettings& settings) const;

    const ShadowCascade& nearCascade() const { return m_near; }
    const ShadowCascade& farCascade() const { return m_far; }
    gfx::Texture& nearMap() { return m_nearMap; }
    gfx::Texture& farMap() { return m_farMap; }

private:
    void draw(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, SceneRenderer& scene, gfx::Texture& map,
              const ShadowCascade& cascade);

    gfx::Texture m_nearMap;
    gfx::Texture m_farMap;
    ShadowCascade m_near;
    ShadowCascade m_far;
    bool m_farValid = false;
    bool m_nearValid = false;
    glm::vec3 m_farSunDirection{0.0f};
};

} // namespace render
