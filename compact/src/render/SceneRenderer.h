#pragma once

#include "atmosphere/AtmosphereRenderer.h"
#include "gfx/Buffer.h"
#include "gfx/Pipeline.h"
#include "scene/Meshes.h"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <unordered_map>
#include <vector>

namespace compact::render {

/// 描画する物体（地形、または単位球を配置したもの）
struct SceneObject {
    bool terrain = false;
    glm::vec3 centerM{0.0f};  // 球の中心 [m]（地形は無視）
    float radiusM = 1.0f;     // 球の半径 [m]
    glm::vec3 albedo{0.18f};
};

/// 地表の影（2 段のカスケード、design.md §5.5）。全体の SRV が無効値なら影なし。
struct SceneShadow {
    uint32_t nearSrv = gfx::kInvalidDescriptorIndex;
    uint32_t farSrv = gfx::kInvalidDescriptorIndex;
    glm::mat4 nearCameraRelativeToUvz{1.0f};  // カメラ相対の位置 [m] → (u, v, 深度)
    glm::mat4 farCameraRelativeToUvz{1.0f};
    float nearNormalOffsetM = 0.0f;  // 法線オフセット [m]
    float farNormalOffsetM = 0.0f;
    float blendBand = 0.1f;          // 近景の端で全体と混ぜる帯の幅（近景の半幅に対する割合）
};

/// Mesh.hlsl の MESH_OUTPUT_*


struct SceneView {
    glm::mat4 viewProjRot{1.0f};  // 回転のみのビュー × 射影（カメラ相対の座標用）
    glm::vec3 cameraPosM{0.0f};   // ワールド [m]
    float exposureScale = 1.0f;   // 1/(1.2·2^EV100) [1/(cd/m²)]
    bool skyIrradiance = true;    // 空からの拡散照度を加える
    SceneShadow shadow;

};

/// 地形・テスト球を、シーンカラー（fb）と深度（Reverse-Z）に描く。design.md §6。
class SceneRenderer {
public:
    SceneRenderer(gfx::Device& device, const gfx::PipelineLibrary& pipelines);

    std::vector<SceneObject>& objects() { return m_objects; }
    float terrainAlbedo = 0.18f;
    bool showTerrain = true;
    bool showSpheres = true;

    /// `colorRtv` と `dsv` は RENDER_TARGET / DEPTH_WRITE 状態であること。深度は 0 でクリアしておくこと。
    void record(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, D3D12_CPU_DESCRIPTOR_HANDLE colorRtv,
                DXGI_FORMAT colorFormat, D3D12_CPU_DESCRIPTOR_HANDLE dsv, uint32_t width, uint32_t height,
                const SceneView& view, const atmosphere::AtmosphereSettings& params,
                atmosphere::AtmosphereRenderer& atmosphere);
    /// 深度だけを描く（シャドウマップ）。地形のスカートも描く。`originRelativeToClip` は「`originM` からの
    /// 相対位置 [m] → クリップ」。
    /// `dsv` は DEPTH_WRITE 状態で、0 でクリアしておくこと。
    void recordDepth(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, D3D12_CPU_DESCRIPTOR_HANDLE dsv,
                     uint32_t width, uint32_t height, const glm::mat4& originRelativeToClip, const glm::vec3& originM);
    /// 全物体を囲む箱 [m]（地形の範囲と最大標高、スカートの下端、球）。シャドウマップの範囲の当てはめに使う。
    void bounds(glm::vec3& minM, glm::vec3& maxM) const;

private:
    struct GpuMesh {
        gfx::Buffer vertices;
        gfx::Buffer indices;
        uint32_t indexCount = 0;
    };
    GpuMesh upload(gfx::Device& device, const scene::MeshData& mesh, const wchar_t* name);
    ID3D12PipelineState* pso(DXGI_FORMAT format);

    const gfx::PipelineLibrary& m_pipelines;
    GpuMesh m_terrain;
    GpuMesh m_terrainSkirt;  // シャドウマップにだけ描く（地面を中身の詰まった地盤として扱う）
    GpuMesh m_sphere;
    std::vector<SceneObject> m_objects;
    std::unordered_map<DXGI_FORMAT, ComPtr<ID3D12PipelineState>> m_psos;
    ComPtr<ID3D12PipelineState> m_depthPso;
};

} // namespace render
