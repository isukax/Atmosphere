#include "scene/Camera.h"

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace scene {

glm::vec3 Camera::forward() const
{
    return {std::sin(yawRad) * std::cos(pitchRad), std::sin(pitchRad), -std::cos(yawRad) * std::cos(pitchRad)};
}

glm::vec3 Camera::right() const
{
    return glm::normalize(glm::cross(forward(), glm::vec3(0.0f, 1.0f, 0.0f)));
}

glm::mat4 Camera::viewRotation() const
{
    return glm::lookAtRH(glm::vec3(0.0f), forward(), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::projection(float aspect) const
{
    const float f = 1.0f / std::tan(0.5f * verticalFovRad);
    // 列優先（glm）: m[列][行]。クリップ座標 = P · ビュー座標。
    //   x_clip = f/aspect · x, y_clip = f · y, z_clip = near, w_clip = −z_view
    //   → 深度 = z_clip / w_clip = near / (−z_view)
    glm::mat4 p(0.0f);
    p[0][0] = f / aspect;
    p[1][1] = f;
    p[3][2] = nearM;
    p[2][3] = -1.0f;
    return p;
}

} // namespace scene
