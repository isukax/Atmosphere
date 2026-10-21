#include "scene/Meshes.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace compact::scene {
namespace {

uint32_t hash2(int32_t x, int32_t y)
{
    uint32_t h = static_cast<uint32_t>(x) * 0x8DA6B343u ^ static_cast<uint32_t>(y) * 0xD8163841u;
    h ^= h >> 13;
    h *= 0x85EBCA6Bu;
    h ^= h >> 16;
    return h;
}

float lattice(int32_t x, int32_t y)
{
    return static_cast<float>(hash2(x, y) & 0xFFFFFFu) / 16777215.0f;
}

/// 値ノイズ（格子点の乱数を 5 次のフェードで補間）。出力 [0, 1]。
float valueNoise(float x, float y)
{
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const int32_t ix = static_cast<int32_t>(fx);
    const int32_t iy = static_cast<int32_t>(fy);
    auto fade = [](float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); };
    const float u = fade(x - fx);
    const float v = fade(y - fy);
    const float a = lattice(ix, iy);
    const float b = lattice(ix + 1, iy);
    const float c = lattice(ix, iy + 1);
    const float d = lattice(ix + 1, iy + 1);
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v;
}

float fbm(float x, float y)
{
    float sum = 0.0f;
    float amplitude = 0.5f;
    float norm = 0.0f;
    for (int octave = 0; octave < 6; ++octave) {
        sum += amplitude * valueNoise(x, y);
        norm += amplitude;
        x *= 2.03f;
        y *= 2.03f;
        amplitude *= 0.5f;
    }
    return sum / norm;
}

float smoothStep(float edge0, float edge1, float x)
{
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

} // namespace

float terrainHeightM(const TerrainSettings& s, float xM, float zM)
{
    // 原点の周りは平地（カメラの初期位置・テスト球の置き場所）。外側へなめらかに山へつなぐ。
    const float mask = smoothStep(s.flatRadiusM, s.blendRadiusM, std::sqrt(xM * xM + zM * zM));
    // 山の起伏は 0〜1 の fBm を 2 乗して谷を広く、峰を鋭くする
    const float n = fbm(xM / 9000.0f + 17.3f, zM / 9000.0f - 4.1f);
    return s.maxHeightM * n * n * mask;
}

MeshData makeTerrain(const TerrainSettings& s)
{
    MeshData mesh;
    const uint32_t n = s.cells + 1;
    const float step = s.sizeM / static_cast<float>(s.cells);
    const float origin = -0.5f * s.sizeM;
    mesh.vertices.resize(static_cast<size_t>(n) * n);
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < n; ++i) {
            const float x = origin + step * static_cast<float>(i);
            const float z = origin + step * static_cast<float>(j);
            const float y = terrainHeightM(s, x, z);
            // 法線: 高さの中心差分（格子間隔の半分）。n = normalize(−∂h/∂x, 1, −∂h/∂z)
            const float e = 0.5f * step;
            const float dhdx = (terrainHeightM(s, x + e, z) - terrainHeightM(s, x - e, z)) / (2.0f * e);
            const float dhdz = (terrainHeightM(s, x, z + e) - terrainHeightM(s, x, z - e)) / (2.0f * e);
            const float len = std::sqrt(dhdx * dhdx + 1.0f + dhdz * dhdz);
            mesh.vertices[static_cast<size_t>(j) * n + i] = {{x, y, z}, {-dhdx / len, 1.0f / len, -dhdz / len}};
        }
    }
    mesh.indices.reserve(static_cast<size_t>(s.cells) * s.cells * 6);
    for (uint32_t j = 0; j < s.cells; ++j) {
        for (uint32_t i = 0; i < s.cells; ++i) {
            const uint32_t a = j * n + i;
            const uint32_t b = a + 1;
            const uint32_t c = a + n;
            const uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
        }
    }
    return mesh;
}

MeshData makeTerrainSkirt(const TerrainSettings& s, float depthM)
{
    MeshData mesh;
    const uint32_t n = s.cells + 1;
    const float step = s.sizeM / static_cast<float>(s.cells);
    const float origin = -0.5f * s.sizeM;
    // 四辺それぞれ: 端の格子点（上）と、その真下 −depthM の点（下）を交互に並べ、隣どうしを 2 枚の三角形で結ぶ。
    // 法線は外向きの水平（シャドウマップの深度だけに使うので、陰影には使わない）。
    const float edges[4][4] = {
        // 始点 x, z、進む向き dx, dz
        {origin, origin, 1.0f, 0.0f},
        {origin, -origin, 1.0f, 0.0f},
        {origin, origin, 0.0f, 1.0f},
        {-origin, origin, 0.0f, 1.0f},
    };
    const float normals[4][3] = {{0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    for (int e = 0; e < 4; ++e) {
        const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        for (uint32_t i = 0; i < n; ++i) {
            const float x = edges[e][0] + edges[e][2] * step * static_cast<float>(i);
            const float z = edges[e][1] + edges[e][3] * step * static_cast<float>(i);
            const float top = terrainHeightM(s, x, z);
            const float* nrm = normals[e];
            mesh.vertices.push_back({{x, top, z}, {nrm[0], nrm[1], nrm[2]}});
            mesh.vertices.push_back({{x, -depthM, z}, {nrm[0], nrm[1], nrm[2]}});
        }
        for (uint32_t i = 0; i + 1 < n; ++i) {
            const uint32_t a = base + 2 * i;
            mesh.indices.insert(mesh.indices.end(), {a, a + 1, a + 2, a + 2, a + 1, a + 3});
        }
    }
    return mesh;
}

MeshData makeUnitSphere(uint32_t segments, uint32_t rings)
{
    MeshData mesh;
    for (uint32_t r = 0; r <= rings; ++r) {
        const float theta = std::numbers::pi_v<float> * static_cast<float>(r) / static_cast<float>(rings);
        for (uint32_t s = 0; s <= segments; ++s) {
            const float phi = 2.0f * std::numbers::pi_v<float> * static_cast<float>(s) / static_cast<float>(segments);
            const float x = std::sin(theta) * std::cos(phi);
            const float y = std::cos(theta);
            const float z = std::sin(theta) * std::sin(phi);
            mesh.vertices.push_back({{x, y, z}, {x, y, z}});
        }
    }
    for (uint32_t r = 0; r < rings; ++r) {
        for (uint32_t s = 0; s < segments; ++s) {
            const uint32_t a = r * (segments + 1) + s;
            const uint32_t b = a + 1;
            const uint32_t c = a + segments + 1;
            const uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
        }
    }
    return mesh;
}

} // namespace scene
