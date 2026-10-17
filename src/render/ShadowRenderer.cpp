#include "render/ShadowRenderer.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace render {
namespace {

/// 光源空間の軸。Z = 太陽へ向かう方向。太陽が真上に近いと +Y との外積が退化するので +Z を基準にする。
void lightBasis(const glm::vec3& sunDirection, ShadowCascade& c)
{
    c.axisZ = glm::normalize(glm::dvec3(sunDirection));
    const glm::dvec3 reference = std::abs(c.axisZ.y) < 0.99 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0);
    c.axisX = glm::normalize(glm::cross(reference, c.axisZ));
    c.axisY = glm::cross(c.axisZ, c.axisX);
}

glm::dvec3 toLight(const ShadowCascade& c, const glm::dvec3& p)
{
    return {glm::dot(c.axisX, p), glm::dot(c.axisY, p), glm::dot(c.axisZ, p)};
}

/// 箱の 8 頂点を光源空間に移した範囲
void lightBounds(const ShadowCascade& c, const glm::vec3& minM, const glm::vec3& maxM, glm::dvec3& lo, glm::dvec3& hi)
{
    lo = glm::dvec3(1e300);
    hi = glm::dvec3(-1e300);
    for (int i = 0; i < 8; ++i) {
        const glm::dvec3 corner((i & 1) ? maxM.x : minM.x, (i & 2) ? maxM.y : minM.y, (i & 4) ? maxM.z : minM.z);
        const glm::dvec3 l = toLight(c, corner);
        lo = glm::min(lo, l);
        hi = glm::max(hi, l);
    }
}

/// 行ごとに (係数ベクトル, 平行移動) を与えて 4×4 行列（glm は列優先: m[列][行]）を作る
glm::mat4 affineRows(const glm::dvec3 (&rows)[3], const double (&translation)[3])
{
    glm::mat4 m(0.0f);
    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 3; ++k) {
            m[k][r] = static_cast<float>(rows[r][k]);
        }
        m[3][r] = static_cast<float>(translation[r]);
    }
    m[3][3] = 1.0f;
    return m;
}

} // namespace

glm::dvec2 ShadowCascade::texelSizeM() const
{
    return {(maxLight.x - minLight.x) / resolution, (maxLight.y - minLight.y) / resolution};
}

glm::mat4 ShadowCascade::cameraRelativeToUvz(const glm::vec3& cameraPosM) const
{
    // uvz = D·L·(相対位置 + カメラ) + b。平行移動の部分（D·L·カメラ + b）は倍精度で計算してから float にする
    const glm::dvec3 size = maxLight - minLight;
    const glm::dvec3 cam = toLight(*this, glm::dvec3(cameraPosM));
    const glm::dvec3 rows[3] = {axisX / size.x, -axisY / size.y, axisZ / size.z};
    const double translation[3] = {(cam.x - minLight.x) / size.x, (maxLight.y - cam.y) / size.y,
                                   (cam.z - minLight.z) / size.z};
    return affineRows(rows, translation);
}

glm::mat4 ShadowCascade::originRelativeToClip() const
{
    // クリップ = (2u − 1, 1 − 2v, 深度)
    const glm::dvec3 size = maxLight - minLight;
    const glm::dvec3 o = toLight(*this, glm::dvec3(originM));
    const glm::dvec3 rows[3] = {2.0 * axisX / size.x, 2.0 * axisY / size.y, axisZ / size.z};
    const double translation[3] = {2.0 * (o.x - minLight.x) / size.x - 1.0, 2.0 * (o.y - maxLight.y) / size.y + 1.0,
                                   (o.z - minLight.z) / size.z};
    return affineRows(rows, translation);
}

ShadowRenderer::ShadowRenderer(gfx::Device& device)
    : m_nearMap(device, {kNearResolution, kNearResolution, DXGI_FORMAT_D32_FLOAT, true, false, false,
                         L"ShadowMapNear", 1, true}),
      m_farMap(device, {kFarResolution, kFarResolution, DXGI_FORMAT_D32_FLOAT, true, false, false, L"ShadowMapFar",
                        1, true})
{
    m_near.resolution = kNearResolution;
    m_far.resolution = kFarResolution;
}

void ShadowRenderer::draw(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, SceneRenderer& scene,
                          gfx::Texture& map, const ShadowCascade& cascade)
{
    map.transition(cmd, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    cmd->ClearDepthStencilView(map.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);  // 0 = 遮蔽物なし
    scene.recordDepth(cmd, upload, map.dsv(), cascade.resolution, cascade.resolution, cascade.originRelativeToClip(),
                      cascade.originM);
    map.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
}

bool ShadowRenderer::updateFar(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, SceneRenderer& scene,
                               const glm::vec3& sunDirection, bool force)
{
    if (m_farValid && !force && sunDirection == m_farSunDirection) {
        return false;
    }
    m_farValid = true;
    m_farSunDirection = sunDirection;

    // [POINT] 全物体を囲む箱を光源空間に移し、その範囲にぴったり合わせる（縦横で別の倍率）。太陽が低いと
    //         光源空間の縦の幅は 40 km × sin(仰角) + 標高差 程度に縮むので、縦のテクセルが細かくなる
    //         （天頂角 88° で約 1.2 m。横は太陽の方位で 40 km / 4096 ≈ 10 m 〜 40 km × √2 / 4096 ≈ 14 m）。
    glm::vec3 minM;
    glm::vec3 maxM;
    scene.bounds(minM, maxM);
    lightBasis(sunDirection, m_far);
    glm::dvec3 lo;
    glm::dvec3 hi;
    lightBounds(m_far, minM, maxM, lo, hi);
    const glm::dvec3 pad = (hi - lo) * 0.005;
    m_far.minLight = lo - pad;
    m_far.maxLight = hi + pad;
    m_far.originM = 0.5f * (minM + maxM);
    draw(cmd, upload, scene, m_farMap, m_far);
    return true;
}

void ShadowRenderer::updateNear(ID3D12GraphicsCommandList* cmd, gfx::UploadRing& upload, SceneRenderer& scene,
                                const scene::Camera& camera, float aspect, const ShadowSettings& settings)
{
    if (!settings.nearCascade) {
        m_nearValid = false;
        return;
    }
    m_nearValid = true;
    // 視錐台の 0〜range を囲む球: 中心は視線上 range/2、半径は遠い側の角までの距離
    const double range = settings.nearRangeM;
    const double halfHeight = range * std::tan(0.5 * camera.verticalFovRad);
    const double halfWidth = halfHeight * aspect;
    const double radius = std::sqrt(0.25 * range * range + halfWidth * halfWidth + halfHeight * halfHeight);
    const glm::dvec3 center = glm::dvec3(camera.positionM) + glm::dvec3(camera.forward()) * (0.5 * range);

    m_near.axisX = m_far.axisX;
    m_near.axisY = m_far.axisY;
    m_near.axisZ = m_far.axisZ;
    // [POINT] 中心を光源空間でテクセル単位に丸める。カメラが動いてもテクセルの格子がワールドに対して動かず、
    //         影の輪郭がちらつかない（半径は向きによらず一定なので、回転してもテクセルの大きさが変わらない）。
    const double texel = 2.0 * radius / m_near.resolution;
    const glm::dvec3 c = toLight(m_near, center);
    const double cx = std::floor(c.x / texel) * texel;
    const double cy = std::floor(c.y / texel) * texel;
    m_near.minLight = glm::dvec3(cx - radius, cy - radius, c.z - radius);
    // 奥行きは光源の側へ全物体の端まで伸ばす（範囲の外にある山の影も落とすため）
    m_near.maxLight = glm::dvec3(cx + radius, cy + radius, std::max(c.z + radius, m_far.maxLight.z));
    m_near.originM = camera.positionM;
    draw(cmd, upload, scene, m_nearMap, m_near);
}

SceneShadow ShadowRenderer::sceneShadow(const glm::vec3& cameraPosM, const ShadowSettings& settings) const
{
    SceneShadow s;
    if (!settings.surface || !m_farValid) {
        return s;
    }
    s.farSrv = m_farMap.srvIndex();
    s.farCameraRelativeToUvz = m_far.cameraRelativeToUvz(cameraPosM);
    const glm::dvec2 farTexel = m_far.texelSizeM();
    s.farNormalOffsetM = static_cast<float>(settings.normalOffsetTexels * std::min(farTexel.x, farTexel.y));
    if (m_nearValid) {
        s.nearSrv = m_nearMap.srvIndex();
        s.nearCameraRelativeToUvz = m_near.cameraRelativeToUvz(cameraPosM);
        const glm::dvec2 nearTexel = m_near.texelSizeM();
        s.nearNormalOffsetM = static_cast<float>(settings.normalOffsetTexels * std::min(nearTexel.x, nearTexel.y));
    }
    s.blendBand = settings.blendBand;
    return s;
}

atmosphere::ShadowInput ShadowRenderer::atmosphereShadow(const glm::vec3& cameraPosM,
                                                         const ShadowSettings& settings) const
{
    atmosphere::ShadowInput s;
    if (settings.aerialPerspective && m_farValid) {
        s.mapSrv = m_farMap.srvIndex();
        s.cameraRelativeToUvz = m_far.cameraRelativeToUvz(cameraPosM);
        if (m_nearValid) {
            s.nearMapSrv = m_nearMap.srvIndex();
            s.nearCameraRelativeToUvz = m_near.cameraRelativeToUvz(cameraPosM);
        }
    }
    return s;
}

} // namespace render
