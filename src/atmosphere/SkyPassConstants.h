#pragma once

// shaders/atmosphere/SkyPassConstants.hlsli の cbuffer と同じ並び（大気モジュール内部用）。

#include "atmosphere/AtmosphereParams.h"

#include <glm/mat4x4.hpp>

#include <cstddef>
#include <cstdint>

namespace atmosphere {

/// SkyPassConstants.hlsli の SKY_FLAG_*
inline constexpr uint32_t kSkyFlagMultipleScattering = 1u;
inline constexpr uint32_t kSkyFlagSunDisk = 2u;
inline constexpr uint32_t kSkyFlagWriteApTransmittance = 4u;  // 検証用: 地形のピクセルに透過率を書く（V3h）

struct SkyPassConstants {
    AtmosphereGpu atmosphere;          // c0-c9
    float invViewProjRot[4][4];        // c10-c13（行優先）
    float cameraPosKm[3];              // c14.xyz
    uint32_t outputUav;                // c14.w
    float sunDirection[3];             // c15.xyz
    uint32_t transmittanceLutSrv;      // c15.w
    float sunIlluminanceLux[3];        // c16.xyz
    float exposureScale;               // c16.w
    uint32_t width;                    // c17.x
    uint32_t height;                   // c17.y
    uint32_t stepCount;                // c17.z
    uint32_t sunTransmittanceMode;     // c17.w
    uint32_t nestedSteps;              // c18.x
    uint32_t outputMode;               // c18.y
    uint32_t directionUav;             // c18.z
    uint32_t multiScatteringLutSrv;    // c18.w
    float transmittanceLutSize[2];     // c19.xy
    float multiScatteringLutSize[2];   // c19.zw
    uint32_t skyViewLutSrv;            // c20.x
    uint32_t flags;                    // c20.y
    float skyViewLutSize[2];           // c20.zw
    uint32_t frameIndex;               // c21.x
    uint32_t maxDepth;                 // c21.y
    uint32_t samplesPerDispatch;       // c21.z
    uint32_t accumulationUavA;         // c21.w
    uint32_t accumulationUavB;         // c22.x
    uint32_t skyViewMieLutUav;         // c22.y
    uint32_t skyViewMieLutSrv;         // c22.z
    uint32_t apTransmittanceMode;      // c22.w（AerialPerspectiveTransmittanceMode）
    uint32_t aerialPerspectiveTransmittanceUav;  // c23.x
    uint32_t aerialPerspectiveTransmittanceSrv;  // c23.y
    uint32_t pathTraceMediumSrv;       // c23.z パストレーサーの表の媒質（design.md §5.8）。無効値なら解析式の媒質
    uint32_t skyIrradianceUav;         // c23.w
    uint32_t aerialPerspectiveUav;     // c24.x
    uint32_t aerialPerspectiveMieUav;  // c24.y
    uint32_t aerialPerspectiveSrv;     // c24.z
    uint32_t aerialPerspectiveMieSrv;  // c24.w
    uint32_t depthSrv;                 // c25.x
    float apKmPerSlice;                // c25.y
    float apSliceCount;                // c25.z
    uint32_t raymarchMeanTransmittance;  // c25.w
    float shadowFarToUvz[4][4];        // c26-c29（行優先）
    uint32_t shadowFarSrv;             // c30.x
    uint32_t shadowNearSrv;            // c30.y
    uint32_t apShadowMethod;           // c30.z
    uint32_t apPerPixelSteps;          // c30.w
    float shadowNearToUvz[4][4];       // c31-c34
    float apShadowFilterScale;         // c35.x
    uint32_t apPerPixelBisectSteps;    // c35.y
    uint32_t aerialPerspectiveMsUav;   // c35.z
    uint32_t aerialPerspectiveMsSrv;   // c35.w
    uint32_t skyViewMie1LutUav;        // c36.x Mie の成分 1（2 成分のプリセット、design.md §5.9）
    uint32_t skyViewMie1LutSrv;        // c36.y
    uint32_t aerialPerspectiveMie1Uav; // c36.z
    uint32_t aerialPerspectiveMie1Srv; // c36.w
};
// [POINT] HLSL の packoffset と 1 対 1 で照合する（cN = N×16 バイト目）
static_assert(offsetof(SkyPassConstants, invViewProjRot) == 10 * 16);
static_assert(offsetof(SkyPassConstants, cameraPosKm) == 14 * 16);
static_assert(offsetof(SkyPassConstants, outputUav) == 14 * 16 + 12);
static_assert(offsetof(SkyPassConstants, sunDirection) == 15 * 16);
static_assert(offsetof(SkyPassConstants, sunIlluminanceLux) == 16 * 16);
static_assert(offsetof(SkyPassConstants, width) == 17 * 16);
static_assert(offsetof(SkyPassConstants, nestedSteps) == 18 * 16);
static_assert(offsetof(SkyPassConstants, multiScatteringLutSrv) == 18 * 16 + 12);
static_assert(offsetof(SkyPassConstants, transmittanceLutSize) == 19 * 16);
static_assert(offsetof(SkyPassConstants, multiScatteringLutSize) == 19 * 16 + 8);
static_assert(offsetof(SkyPassConstants, skyViewLutSrv) == 20 * 16);
static_assert(offsetof(SkyPassConstants, skyViewLutSize) == 20 * 16 + 8);
static_assert(offsetof(SkyPassConstants, frameIndex) == 21 * 16);
static_assert(offsetof(SkyPassConstants, accumulationUavB) == 22 * 16);
static_assert(offsetof(SkyPassConstants, skyViewMieLutUav) == 22 * 16 + 4);
static_assert(offsetof(SkyPassConstants, skyViewMieLutSrv) == 22 * 16 + 8);
static_assert(offsetof(SkyPassConstants, apTransmittanceMode) == 22 * 16 + 12);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveTransmittanceUav) == 23 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveTransmittanceSrv) == 23 * 16 + 4);
static_assert(offsetof(SkyPassConstants, pathTraceMediumSrv) == 23 * 16 + 8);
static_assert(offsetof(SkyPassConstants, skyIrradianceUav) == 23 * 16 + 12);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveUav) == 24 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMieSrv) == 24 * 16 + 12);
static_assert(offsetof(SkyPassConstants, depthSrv) == 25 * 16);
static_assert(offsetof(SkyPassConstants, raymarchMeanTransmittance) == 25 * 16 + 12);
static_assert(offsetof(SkyPassConstants, shadowFarToUvz) == 26 * 16);
static_assert(offsetof(SkyPassConstants, shadowFarSrv) == 30 * 16);
static_assert(offsetof(SkyPassConstants, apPerPixelSteps) == 30 * 16 + 12);
static_assert(offsetof(SkyPassConstants, shadowNearToUvz) == 31 * 16);
static_assert(offsetof(SkyPassConstants, apShadowFilterScale) == 35 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMsSrv) == 35 * 16 + 12);
static_assert(offsetof(SkyPassConstants, skyViewMie1LutUav) == 36 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMie1Srv) == 36 * 16 + 12);
static_assert(sizeof(SkyPassConstants) == 37 * 16);

} // namespace atmosphere
