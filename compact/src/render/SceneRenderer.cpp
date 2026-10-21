#include "render/SceneRenderer.h"

#include <cstddef>

namespace compact::render {
namespace {

/// shaders/scene/Mesh.hlsl の cbuffer MeshConstants と同じ並び
struct MeshConstants {
    float viewProjRot[4][4];             // c0-c3（行優先）
    float cameraPosM[3];                 // c4.xyz
    uint32_t vertexBufferSrv;            // c4.w
    float objectOffsetM[3];              // c5.xyz
    float objectScale;                   // c5.w
    float albedo[3];                     // c6.xyz
    uint32_t skyIrradianceSrv;           // c6.w
    float sunDirection[3];               // c7.xyz
    uint32_t transmittanceLutSrv;        // c7.w
    float sunIlluminanceLux[3];          // c8.xyz
    float exposureScale;                 // c8.w
    float planetOffsetKm[3];             // c9.xyz
    float pad9;                          // c9.w
    float transmittanceLutSize[2];       // c10.xy
    uint32_t useSkyIrradiance;           // c10.z
    float pad10;                         // c10.w
    atmosphere::AtmosphereGpu atmosphere;  // c11-c16
    float shadowNearToUvz[4][4];         // c17-c20（行優先）
    float shadowFarToUvz[4][4];          // c21-c24
    uint32_t shadowNearSrv;              // c25.x
    uint32_t shadowFarSrv;               // c25.y
    float shadowBlendBand;               // c25.z
    uint32_t outputMode;                 // c25.w
    float shadowNearNormalOffsetM;       // c26.x
    float shadowFarNormalOffsetM;        // c26.y
    float pad26[2];
};
static_assert(offsetof(MeshConstants, cameraPosM) == 4 * 16);
static_assert(offsetof(MeshConstants, vertexBufferSrv) == 4 * 16 + 12);
static_assert(offsetof(MeshConstants, objectOffsetM) == 5 * 16);
static_assert(offsetof(MeshConstants, albedo) == 6 * 16);
static_assert(offsetof(MeshConstants, sunDirection) == 7 * 16);
static_assert(offsetof(MeshConstants, sunIlluminanceLux) == 8 * 16);
static_assert(offsetof(MeshConstants, planetOffsetKm) == 9 * 16);
static_assert(offsetof(MeshConstants, transmittanceLutSize) == 10 * 16);
static_assert(offsetof(MeshConstants, atmosphere) == 11 * 16);
static_assert(offsetof(MeshConstants, shadowNearToUvz) == 17 * 16);
static_assert(offsetof(MeshConstants, shadowFarToUvz) == 21 * 16);
static_assert(offsetof(MeshConstants, shadowNearSrv) == 25 * 16);
static_assert(offsetof(MeshConstants, outputMode) == 25 * 16 + 12);
static_assert(offsetof(MeshConstants, shadowNearNormalOffsetM) == 26 * 16);
static_assert(sizeof(MeshConstants) == 27 * 16);

// [POINT] 地形のスカートの下端 [m]。平面の世界と惑星の曲率の差（地形の角 28 km で 62 m）より深くする。
constexpr float kTerrainSkirtDepthM = 500.0f;

// [POINT] シャドウマップの傾斜スケールのデプスバイアス。深度は光源に近いほど大きいので、負の値で遮蔽物を
//         光源から遠ざける。受け手側の法線オフセット（Mesh.hlsl）と併用する。
constexpr float kShadowSlopeScaledDepthBias = -1.5f;

void store(float (&dst)[3], const glm::vec3& v)
{
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
}

void storeRowMajor(float (&dst)[4][4], const glm::mat4& m)
{
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            dst[row][col] = m[col][row];  // glm（列優先）→ row_major
        }
    }
}

} // namespace

SceneRenderer::SceneRenderer(gfx::Device& device, const gfx::PipelineLibrary& pipelines)
    : m_pipelines(pipelines)
{
    m_terrain = upload(device, scene::makeTerrain(scene::TerrainSettings{}), L"Terrain");
    m_terrainSkirt = upload(device, scene::makeTerrainSkirt(scene::TerrainSettings{}, kTerrainSkirtDepthM),
                            L"TerrainSkirt");
    m_depthPso = pipelines.createDepthOnly("Mesh_vs", DXGI_FORMAT_D32_FLOAT, kShadowSlopeScaledDepthBias);
    m_sphere = upload(device, scene::makeUnitSphere(48, 24), L"Sphere");

    // テスト球（design.md §6）: 近く（半径 2 m、アルベド 0.18 / 0.8）と、遠く（2 km・10 km 先の大きな球）。
    // カメラの初期位置（原点、高さ 2 m、−Z 向き）から見える位置に置く。原点の周り 2 km は標高 0 の平地。
    const scene::TerrainSettings terrain;
    auto onGround = [&](float x, float z, float radius) {
        return glm::vec3(x, scene::terrainHeightM(terrain, x, z) + radius, z);  // 地面に接するように置く
    };
    m_objects.push_back({true, {}, 1.0f, glm::vec3(0.18f)});
    m_objects.push_back({false, onGround(-3.0f, -12.0f, 2.0f), 2.0f, glm::vec3(0.18f)});
    m_objects.push_back({false, onGround(3.0f, -12.0f, 2.0f), 2.0f, glm::vec3(0.8f)});
    m_objects.push_back({false, onGround(-300.0f, -2000.0f, 200.0f), 200.0f, glm::vec3(0.8f)});
    m_objects.push_back({false, onGround(1500.0f, -10000.0f, 500.0f), 500.0f, glm::vec3(0.8f)});
}

SceneRenderer::GpuMesh SceneRenderer::upload(gfx::Device& device, const scene::MeshData& mesh, const wchar_t* name)
{
    GpuMesh gpu;
    const uint64_t vertexBytes = mesh.vertices.size() * sizeof(scene::Vertex);
    const uint64_t indexBytes = mesh.indices.size() * sizeof(uint32_t);
    gpu.vertices = gfx::Buffer(device, {vertexBytes, sizeof(scene::Vertex), true, false, name});
    gpu.indices = gfx::Buffer(device, {indexBytes, 0, false, false, name});
    gpu.vertices.upload(device, mesh.vertices.data(), vertexBytes, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    gpu.indices.upload(device, mesh.indices.data(), indexBytes, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    gpu.indexCount = static_cast<uint32_t>(mesh.indices.size());
    return gpu;
}

ID3D12PipelineState* SceneRenderer::pso(DXGI_FORMAT format)
{
    auto it = m_psos.find(format);
    if (it == m_psos.end()) {
        it = m_psos.emplace(format, m_pipelines.createMesh("Mesh_vs", "Mesh_ps", format, DXGI_FORMAT_D32_FLOAT)).first;
    }
    return it->second.Get();
}

void SceneRenderer::record(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                           D3D12_CPU_DESCRIPTOR_HANDLE colorRtv, DXGI_FORMAT colorFormat,
                           D3D12_CPU_DESCRIPTOR_HANDLE dsv, uint32_t width, uint32_t height, const SceneView& view,
                           const atmosphere::AtmosphereSettings& params, atmosphere::AtmosphereRenderer& atmosphere)
{
    MeshConstants base{};
    storeRowMajor(base.viewProjRot, view.viewProjRot);
    store(base.cameraPosM, view.cameraPosM);
    base.skyIrradianceSrv = atmosphere.skyIrradiance().srvIndex();
    store(base.sunDirection, params.sunDirectionWs);
    base.transmittanceLutSrv = atmosphere.transmittanceLut().srvIndex();
    store(base.sunIlluminanceLux, params.sunIlluminanceLux * atmosphere.sunColor());
    base.exposureScale = view.exposureScale;
    // [POINT] ワールド原点（地表）の惑星座標 [km]。シェーダーでは p_km = p_m × 0.001 + このオフセット。
    store(base.planetOffsetKm, atmosphere::worldToPlanetKm(glm::vec3(0.0f)));
    base.transmittanceLutSize[0] = static_cast<float>(atmosphere.quality().transmittanceWidth);
    base.transmittanceLutSize[1] = static_cast<float>(atmosphere.quality().transmittanceHeight);
    base.useSkyIrradiance = view.skyIrradiance ? 1u : 0u;
    base.atmosphere = atmosphere.gpuParameters();
    storeRowMajor(base.shadowNearToUvz, view.shadow.nearCameraRelativeToUvz);
    storeRowMajor(base.shadowFarToUvz, view.shadow.farCameraRelativeToUvz);
    base.shadowNearSrv = view.shadow.nearSrv;
    base.shadowFarSrv = view.shadow.farSrv;
    base.shadowBlendBand = view.shadow.blendBand;
    base.outputMode = 0;
    base.shadowNearNormalOffsetM = view.shadow.nearNormalOffsetM;
    base.shadowFarNormalOffsetM = view.shadow.farNormalOffsetM;

    cmd->SetGraphicsRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(pso(colorFormat));
    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);
    cmd->OMSetRenderTargets(1, &colorRtv, FALSE, &dsv);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    for (const SceneObject& object : m_objects) {
        if (object.terrain ? !showTerrain : !showSpheres) {
            continue;
        }
        const GpuMesh& mesh = object.terrain ? m_terrain : m_sphere;
        MeshConstants c = base;
        c.vertexBufferSrv = mesh.vertices.srvIndex();
        store(c.objectOffsetM, object.terrain ? glm::vec3(0.0f) : object.centerM);
        c.objectScale = object.terrain ? 1.0f : object.radiusM;
        store(c.albedo, object.terrain ? glm::vec3(terrainAlbedo) : object.albedo);
        cmd->SetGraphicsRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
        // [POINT] 頂点バッファは使わず、インデックスバッファだけ IA に渡す。SV_VertexID がインデックスの値になるので、
        //         頂点シェーダーが構造化バッファ（Bindless）から頂点を読む。
        const D3D12_INDEX_BUFFER_VIEW ibv{mesh.indices.resource()->GetGPUVirtualAddress(),
                                          static_cast<UINT>(mesh.indexCount * sizeof(uint32_t)), DXGI_FORMAT_R32_UINT};
        cmd->IASetIndexBuffer(&ibv);
        cmd->DrawIndexedInstanced(mesh.indexCount, 1, 0, 0, 0);
    }
}

void SceneRenderer::recordDepth(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload,
                                D3D12_CPU_DESCRIPTOR_HANDLE dsv, uint32_t width, uint32_t height,
                                const glm::mat4& originRelativeToClip, const glm::vec3& originM)
{
    // 頂点シェーダーは通常の描画と同じ: クリップ = gViewProjRot × (頂点 − gCameraPosM)
    MeshConstants base{};
    storeRowMajor(base.viewProjRot, originRelativeToClip);
    store(base.cameraPosM, originM);

    cmd->SetGraphicsRootSignature(m_pipelines.rootSignature());
    cmd->SetPipelineState(m_depthPso.Get());
    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);
    cmd->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    auto draw = [&](const GpuMesh& mesh, const glm::vec3& offsetM, float scale) {
        MeshConstants c = base;
        c.vertexBufferSrv = mesh.vertices.srvIndex();
        store(c.objectOffsetM, offsetM);
        c.objectScale = scale;
        cmd->SetGraphicsRootConstantBufferView(gfx::PipelineLibrary::kRootPassConstants, upload.push(c));
        const D3D12_INDEX_BUFFER_VIEW ibv{mesh.indices.resource()->GetGPUVirtualAddress(),
                                          static_cast<UINT>(mesh.indexCount * sizeof(uint32_t)), DXGI_FORMAT_R32_UINT};
        cmd->IASetIndexBuffer(&ibv);
        cmd->DrawIndexedInstanced(mesh.indexCount, 1, 0, 0, 0);
    };
    for (const SceneObject& object : m_objects) {
        if (object.terrain ? !showTerrain : !showSpheres) {
            continue;
        }
        if (object.terrain) {
            draw(m_terrain, glm::vec3(0.0f), 1.0f);
            // [POINT] 地形は端の開いた高さマップなので、太陽が低いと光源側の端で地表より低い所から入る光線が
            //         どの面にも当たらず、地形の下（山の中）の点が日なたになる。空気遠近法のボクセルの光線は、
            //         地形より奥のスライスで山の中を通るので、その光が手前の地形のピクセルに補間で漏れた
            //         （design.md §14 Phase 4.5）。四辺から下へ壁を垂らし、地面を中身の詰まった地盤として扱う。
            draw(m_terrainSkirt, glm::vec3(0.0f), 1.0f);
        } else {
            draw(m_sphere, object.centerM, object.radiusM);
        }
    }
}

void SceneRenderer::bounds(glm::vec3& minM, glm::vec3& maxM) const
{
    // Keep a stable, nonempty shadow volume even when every object is hidden.
    const scene::TerrainSettings terrain;
    minM = glm::vec3(-0.5f * terrain.sizeM, -kTerrainSkirtDepthM, -0.5f * terrain.sizeM);
    maxM = glm::vec3(0.5f * terrain.sizeM, terrain.maxHeightM, 0.5f * terrain.sizeM);
    for (const SceneObject& object : m_objects) {
        if (!object.terrain) {
            minM = glm::min(minM, object.centerM - glm::vec3(object.radiusM));
            maxM = glm::max(maxM, object.centerM + glm::vec3(object.radiusM));
        }
    }
}

} // namespace render
