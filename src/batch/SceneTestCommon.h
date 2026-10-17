#pragma once

// 地形ありのバッチ検証（Phase 4 / 4.5）で共通に使うカメラ・描画先。

#include "atmosphere/AtmosphereRenderer.h"
#include "gfx/Texture.h"
#include "io/ExrWriter.h"
#include "render/SceneRenderer.h"
#include "scene/Camera.h"

#include <glm/trigonometric.hpp>

#include <cstring>
#include <filesystem>
#include <vector>

namespace batch {

inline const std::filesystem::path kSceneTestCoefficientPath =
    std::filesystem::path(ATMOS_DATA_DIR) / "atmosphere_coefficients.json";

inline std::vector<float> bytesToFloats(const std::vector<uint8_t>& bytes)
{
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

/// 検証・計測用のカメラ: 高さ 300 m から −Z（山の方向）を少し見下ろす
inline scene::Camera testCamera()
{
    scene::Camera camera;
    camera.positionM = glm::vec3(0.0f, 300.0f, 0.0f);
    camera.yawRad = 0.0f;
    camera.pitchRad = glm::radians(-2.0f);
    camera.verticalFovRad = glm::radians(60.0f);
    return camera;
}

/// シーンカラー・深度とその描画（地形・球 → 合成の前まで）
struct SceneTargets {
    gfx::Texture color;
    gfx::Texture depth;
    SceneTargets(gfx::Device& device, uint32_t w, uint32_t h, DXGI_FORMAT colorFormat)
        : color(device, {w, h, colorFormat, true, true, true, L"TestSceneColor"}),
          depth(device, {w, h, DXGI_FORMAT_D32_FLOAT, true, false, false, L"TestSceneDepth", 1, true})
    {
    }

    void drawScene(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, render::SceneRenderer& sceneRenderer,
                   const render::SceneView& view, const atmosphere::AtmosphereParams& params,
                   atmosphere::AtmosphereRenderer& atmosphere)
    {
        const uint32_t w = color.desc().width;
        const uint32_t h = color.desc().height;
        color.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
        depth.transition(cmd, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        const float black[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        cmd->ClearRenderTargetView(color.rtv(), black, 0, nullptr);
        cmd->ClearDepthStencilView(depth.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);
        sceneRenderer.record(cmd, upload, color.rtv(), color.desc().format, depth.dsv(), w, h, view, params, atmosphere);
        depth.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        color.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    /// 深度（Reverse-Z の値を R に。ビュー空間の奥行き = near / depth [m]）を EXR で保存する
    void saveDepth(gfx::Device& device, const std::filesystem::path& path, float nearM)
    {
        const uint32_t w = color.desc().width;
        const uint32_t h = color.desc().height;
        const std::vector<float> values = bytesToFloats(depth.readback(device, 4));
        std::vector<float> rgba(static_cast<size_t>(w) * h * 4, 0.0f);
        for (size_t i = 0; i < values.size(); ++i) {
            rgba[i * 4 + 0] = values[i];
        }
        io::saveExr(path, w, h, rgba, {{"units", "reverse-Z depth (R). view depth [m] = near / depth"}, {"near_m", nearM}});
    }
};

} // namespace batch
