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
    AtmosphereGpu atmosphere;          // c0-c8
    float invViewProjRot[4][4];        // c9-c12（行優先）
    float cameraPosKm[3];              // c13.xyz
    uint32_t outputUav;                // c13.w
    float sunDirection[3];             // c14.xyz
    uint32_t transmittanceLutSrv;      // c14.w
    float sunIlluminanceLux[3];        // c15.xyz
    float exposureScale;               // c15.w
    uint32_t width;                    // c16.x
    uint32_t height;                   // c16.y
    uint32_t stepCount;                // c16.z
    uint32_t sunTransmittanceMode;     // c16.w
    uint32_t nestedSteps;              // c17.x
    uint32_t outputMode;               // c17.y
    uint32_t directionUav;             // c17.z
    uint32_t multiScatteringLutSrv;    // c17.w
    float transmittanceLutSize[2];     // c18.xy
    float multiScatteringLutSize[2];   // c18.zw
    uint32_t skyViewLutSrv;            // c19.x
    uint32_t flags;                    // c19.y
    float skyViewLutSize[2];           // c19.zw
    uint32_t frameIndex;               // c20.x
    uint32_t maxDepth;                 // c20.y
    uint32_t samplesPerDispatch;       // c20.z
    uint32_t accumulationUavA;         // c20.w
    uint32_t accumulationUavB;         // c21.x
    uint32_t skyViewMieLutUav;         // c21.y
    uint32_t skyViewMieLutSrv;         // c21.z
    uint32_t apTransmittanceMode;      // c21.w（AerialPerspectiveTransmittanceMode）
    uint32_t aerialPerspectiveTransmittanceUav;  // c22.x
    uint32_t aerialPerspectiveTransmittanceSrv;  // c22.y
    uint32_t pathTraceMediumSrv;       // c22.z パストレーサーの表の媒質（design.md §5.8）。無効値なら解析式の媒質
    uint32_t skyIrradianceUav;         // c22.w
    uint32_t aerialPerspectiveUav;     // c23.x
    uint32_t aerialPerspectiveMieUav;  // c23.y
    uint32_t aerialPerspectiveSrv;     // c23.z
    uint32_t aerialPerspectiveMieSrv;  // c23.w
    uint32_t depthSrv;                 // c24.x
    float apKmPerSlice;                // c24.y
    float apSliceCount;                // c24.z
    uint32_t raymarchMeanTransmittance;  // c24.w
    float shadowFarToUvz[4][4];        // c25-c28（行優先）
    uint32_t shadowFarSrv;             // c29.x
    uint32_t shadowNearSrv;            // c29.y
    uint32_t apShadowMethod;           // c29.z
    uint32_t apPerPixelSteps;          // c29.w
    float shadowNearToUvz[4][4];       // c30-c33
    float apShadowFilterScale;         // c34.x
    uint32_t apPerPixelBisectSteps;    // c34.y
    uint32_t aerialPerspectiveMsUav;   // c34.z
    uint32_t aerialPerspectiveMsSrv;   // c34.w
};
// [POINT] HLSL の packoffset と 1 対 1 で照合する（cN = N×16 バイト目）
static_assert(offsetof(SkyPassConstants, invViewProjRot) == 9 * 16);
static_assert(offsetof(SkyPassConstants, cameraPosKm) == 13 * 16);
static_assert(offsetof(SkyPassConstants, outputUav) == 13 * 16 + 12);
static_assert(offsetof(SkyPassConstants, sunDirection) == 14 * 16);
static_assert(offsetof(SkyPassConstants, sunIlluminanceLux) == 15 * 16);
static_assert(offsetof(SkyPassConstants, width) == 16 * 16);
static_assert(offsetof(SkyPassConstants, nestedSteps) == 17 * 16);
static_assert(offsetof(SkyPassConstants, multiScatteringLutSrv) == 17 * 16 + 12);
static_assert(offsetof(SkyPassConstants, transmittanceLutSize) == 18 * 16);
static_assert(offsetof(SkyPassConstants, multiScatteringLutSize) == 18 * 16 + 8);
static_assert(offsetof(SkyPassConstants, skyViewLutSrv) == 19 * 16);
static_assert(offsetof(SkyPassConstants, skyViewLutSize) == 19 * 16 + 8);
static_assert(offsetof(SkyPassConstants, frameIndex) == 20 * 16);
static_assert(offsetof(SkyPassConstants, accumulationUavB) == 21 * 16);
static_assert(offsetof(SkyPassConstants, skyViewMieLutUav) == 21 * 16 + 4);
static_assert(offsetof(SkyPassConstants, skyViewMieLutSrv) == 21 * 16 + 8);
static_assert(offsetof(SkyPassConstants, apTransmittanceMode) == 21 * 16 + 12);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveTransmittanceUav) == 22 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveTransmittanceSrv) == 22 * 16 + 4);
static_assert(offsetof(SkyPassConstants, pathTraceMediumSrv) == 22 * 16 + 8);
static_assert(offsetof(SkyPassConstants, skyIrradianceUav) == 22 * 16 + 12);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveUav) == 23 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMieSrv) == 23 * 16 + 12);
static_assert(offsetof(SkyPassConstants, depthSrv) == 24 * 16);
static_assert(offsetof(SkyPassConstants, raymarchMeanTransmittance) == 24 * 16 + 12);
static_assert(offsetof(SkyPassConstants, shadowFarToUvz) == 25 * 16);
static_assert(offsetof(SkyPassConstants, shadowFarSrv) == 29 * 16);
static_assert(offsetof(SkyPassConstants, apPerPixelSteps) == 29 * 16 + 12);
static_assert(offsetof(SkyPassConstants, shadowNearToUvz) == 30 * 16);
static_assert(offsetof(SkyPassConstants, apShadowFilterScale) == 34 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMsSrv) == 34 * 16 + 12);
static_assert(sizeof(SkyPassConstants) == 35 * 16);

} // namespace atmosphere
