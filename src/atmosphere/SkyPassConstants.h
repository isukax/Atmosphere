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
    uint32_t pad18;
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

} // namespace atmosphere
