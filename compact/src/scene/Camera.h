#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace compact::scene {

/// フライカメラ。ワールドは右手系・+Y 上・単位 m（docs/design.md D10）。
struct Camera {
    glm::vec3 positionM{0.0f, 2.0f, 0.0f};
    float yawRad = 0.0f;     // +Y まわり。0 で −Z を向く
    float pitchRad = 0.15f;  // 上向きが正
    float verticalFovRad = 1.0471976f;  // 60°
    float nearM = 0.1f;

    glm::vec3 forward() const;
    glm::vec3 right() const;

    /// 平行移動を含まないビュー行列（回転のみ）。
    glm::mat4 viewRotation() const;
    /// [POINT] Reverse-Z・無限遠ファーの射影（D3D の深度 [0,1]）。深度 = near / 距離 で、
    ///         近平面が 1、無限遠が 0。float の精度が遠方で落ちにくい（数十 km の遠景向け）。
    glm::mat4 projection(float aspect) const;
};

} // namespace scene
