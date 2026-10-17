#pragma once

#include "gfx/Descriptors.h"

#include <glm/mat4x4.hpp>

#include <cstdint>

namespace atmosphere {

/// 不透明物（地形など）の太陽の影。design.md §5.5。
/// [POINT] 大気モジュールはシャドウマップを持たず、これを受け取るだけにする（Shoga では ShadowDepth パス /
///         CSM の出力に差し替える、§9）。空気遠近法の単散乱（Rayleigh・Mie）にだけ掛け、多重散乱には掛けない。
struct ShadowInput {
    /// 全体のカスケード: D32 のシャドウマップを R32_FLOAT で見る SRV。無効値なら影なし。
    /// 深度は光源に近いほど大きい（Reverse）。クリア値 0 = 遮蔽物なし。
    uint32_t mapSrv = gfx::kInvalidDescriptorIndex;
    /// カメラ相対の位置 [m] → (u, v, 深度)。カメラの位置を平行移動に含めた行列（大気の積分の点は
    /// カメラからの距離 t と方向で表すので、カメラ相対のまま変換すれば float の精度が落ちない）。
    glm::mat4 cameraRelativeToUvz{1.0f};
    /// 近景のカスケード（無効値なら全体だけ）。点が範囲の内側のときだけ使う（design.md §5.6）。
    uint32_t nearMapSrv = gfx::kInvalidDescriptorIndex;
    glm::mat4 nearCameraRelativeToUvz{1.0f};

    bool valid() const { return mapSrv != gfx::kInvalidDescriptorIndex; }
};

/// 空気遠近法の影の方式（SkyPassConstants.hlsli の AP_SHADOW_* と一致させる）。design.md §5.6。
enum class AerialPerspectiveShadowMethod : uint32_t {
    VolumePoint = 0,     // 0: ボリュームの積分で、各ステップ 1 回の比較
    VolumeFiltered = 1,  // 1: ボリュームの積分で、ボクセルの横幅の円盤をぼかして引く
    PerPixel = 2,        // 5: ボリュームは影なし。合成でピクセルごとに影の区間の単散乱を差し引く
};

struct AerialPerspectiveShadowOptions {
    AerialPerspectiveShadowMethod method = AerialPerspectiveShadowMethod::VolumePoint;
    float filterScale = 1.0f;         // 方式 1: 円盤の半径の倍率（1 = ボクセルの半幅）
    uint32_t perPixelSteps = 32;      // 方式 5: 視線上で影を調べる点の数
    uint32_t perPixelBisectSteps = 6; // 方式 5: 影の境界の二分探索の回数
};

} // namespace atmosphere
