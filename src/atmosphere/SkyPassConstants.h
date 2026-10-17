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
    AtmosphereGpu atmosphere;          // c0-c5
    float invViewProjRot[4][4];        // c6-c9（行優先）
    float cameraPosKm[3];              // c10.xyz
    uint32_t outputUav;                // c10.w
    float sunDirection[3];             // c11.xyz
    uint32_t transmittanceLutSrv;      // c11.w
    float sunIlluminanceLux[3];        // c12.xyz
    float exposureScale;               // c12.w
    uint32_t width;                    // c13.x
    uint32_t height;                   // c13.y
    uint32_t stepCount;                // c13.z
    uint32_t sunTransmittanceMode;     // c13.w
    uint32_t nestedSteps;              // c14.x
    uint32_t outputMode;               // c14.y
    uint32_t directionUav;             // c14.z
    uint32_t multiScatteringLutSrv;    // c14.w
    float transmittanceLutSize[2];     // c15.xy
    float multiScatteringLutSize[2];   // c15.zw
    uint32_t skyViewLutSrv;            // c16.x
    uint32_t flags;                    // c16.y
    float skyViewLutSize[2];           // c16.zw
    uint32_t frameIndex;               // c17.x
    uint32_t maxDepth;                 // c17.y
    uint32_t samplesPerDispatch;       // c17.z
    uint32_t accumulationUavA;         // c17.w
    uint32_t accumulationUavB;         // c18.x
    uint32_t skyViewMieLutUav;         // c18.y
    uint32_t skyViewMieLutSrv;         // c18.z
    uint32_t apTransmittanceMode;      // c18.w（AerialPerspectiveTransmittanceMode）
    uint32_t aerialPerspectiveTransmittanceUav;  // c19.x
    uint32_t aerialPerspectiveTransmittanceSrv;  // c19.y
    uint32_t pad19;                    // c19.z
    uint32_t skyIrradianceUav;         // c19.w
    uint32_t aerialPerspectiveUav;     // c20.x
    uint32_t aerialPerspectiveMieUav;  // c20.y
    uint32_t aerialPerspectiveSrv;     // c20.z
    uint32_t aerialPerspectiveMieSrv;  // c20.w
    uint32_t depthSrv;                 // c21.x
    float apKmPerSlice;                // c21.y
    float apSliceCount;                // c21.z
    uint32_t raymarchMeanTransmittance;  // c21.w
    float shadowFarToUvz[4][4];        // c22-c25（行優先）
    uint32_t shadowFarSrv;             // c26.x
    uint32_t shadowNearSrv;            // c26.y
    uint32_t apShadowMethod;           // c26.z
    uint32_t apPerPixelSteps;          // c26.w
    float shadowNearToUvz[4][4];       // c27-c30
    float apShadowFilterScale;         // c31.x
    uint32_t apPerPixelBisectSteps;    // c31.y
    uint32_t aerialPerspectiveMsUav;   // c31.z
    uint32_t aerialPerspectiveMsSrv;   // c31.w
};
// [POINT] HLSL の packoffset と 1 対 1 で照合する（cN = N×16 バイト目）
static_assert(offsetof(SkyPassConstants, invViewProjRot) == 6 * 16);
static_assert(offsetof(SkyPassConstants, cameraPosKm) == 10 * 16);
static_assert(offsetof(SkyPassConstants, outputUav) == 10 * 16 + 12);
static_assert(offsetof(SkyPassConstants, sunDirection) == 11 * 16);
static_assert(offsetof(SkyPassConstants, sunIlluminanceLux) == 12 * 16);
static_assert(offsetof(SkyPassConstants, width) == 13 * 16);
static_assert(offsetof(SkyPassConstants, nestedSteps) == 14 * 16);
static_assert(offsetof(SkyPassConstants, multiScatteringLutSrv) == 14 * 16 + 12);
static_assert(offsetof(SkyPassConstants, transmittanceLutSize) == 15 * 16);
static_assert(offsetof(SkyPassConstants, multiScatteringLutSize) == 15 * 16 + 8);
static_assert(offsetof(SkyPassConstants, skyViewLutSrv) == 16 * 16);
static_assert(offsetof(SkyPassConstants, skyViewLutSize) == 16 * 16 + 8);
static_assert(offsetof(SkyPassConstants, frameIndex) == 17 * 16);
static_assert(offsetof(SkyPassConstants, accumulationUavB) == 18 * 16);
static_assert(offsetof(SkyPassConstants, skyViewMieLutUav) == 18 * 16 + 4);
static_assert(offsetof(SkyPassConstants, skyViewMieLutSrv) == 18 * 16 + 8);
static_assert(offsetof(SkyPassConstants, apTransmittanceMode) == 18 * 16 + 12);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveTransmittanceUav) == 19 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveTransmittanceSrv) == 19 * 16 + 4);
static_assert(offsetof(SkyPassConstants, skyIrradianceUav) == 19 * 16 + 12);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveUav) == 20 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMieSrv) == 20 * 16 + 12);
static_assert(offsetof(SkyPassConstants, depthSrv) == 21 * 16);
static_assert(offsetof(SkyPassConstants, raymarchMeanTransmittance) == 21 * 16 + 12);
static_assert(offsetof(SkyPassConstants, shadowFarToUvz) == 22 * 16);
static_assert(offsetof(SkyPassConstants, shadowFarSrv) == 26 * 16);
static_assert(offsetof(SkyPassConstants, apPerPixelSteps) == 26 * 16 + 12);
static_assert(offsetof(SkyPassConstants, shadowNearToUvz) == 27 * 16);
static_assert(offsetof(SkyPassConstants, apShadowFilterScale) == 31 * 16);
static_assert(offsetof(SkyPassConstants, aerialPerspectiveMsSrv) == 31 * 16 + 12);
static_assert(sizeof(SkyPassConstants) == 32 * 16);

} // namespace atmosphere
