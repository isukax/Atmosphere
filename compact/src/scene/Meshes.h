#pragma once

#include <cstdint>
#include <vector>

namespace compact::scene {

/// 頂点（構造化バッファの 1 要素。HLSL の struct Vertex { float3 position; float3 normal; } と同じ 24 バイト）
struct Vertex {
    float position[3];  // [m]
    float normal[3];
};
static_assert(sizeof(Vertex) == 24);

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct TerrainSettings {
    float sizeM = 40000.0f;          // 一辺 [m]（原点中心）
    uint32_t cells = 512;            // 一辺の区間数
    float maxHeightM = 3000.0f;      // fBm の最大標高 [m]
    float flatRadiusM = 2000.0f;     // この半径の内側は標高 0 の平地
    float blendRadiusM = 8000.0f;    // 平地から山へつなぐ外側の半径
};

/// fBm の高さマップの地形（design.md §6）。
MeshData makeTerrain(const TerrainSettings& settings);
/// 地形の四辺から真下へ垂らす壁（スカート）。シャドウマップにだけ描く（design.md §5.5）。
/// 上端は地形の端の格子点と同じ高さ、下端は −depthM [m]。
MeshData makeTerrainSkirt(const TerrainSettings& settings, float depthM);
/// 単位球（半径 1）。描画時に中心と半径で配置する。
MeshData makeUnitSphere(uint32_t segments, uint32_t rings);
/// 地形の高さ [m]（地形と同じ関数。物を地面に置くときに使う）
float terrainHeightM(const TerrainSettings& settings, float xM, float zM);

} // namespace scene
