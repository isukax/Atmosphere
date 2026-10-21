#pragma once
#include "atmosphere/AtmosphereGpu.h"
#include <cstddef>
namespace compact::atmosphere {
struct MultiScatteringConstants {
    AtmosphereGpu atmosphere;
    uint32_t outputUav;
    uint32_t width;
    uint32_t height;
    uint32_t stepCount;
    float transmittanceLutSize[2];
    uint32_t transmittanceLutSrv;
    uint32_t transferUav;
    uint32_t transferSrv;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    float phaseMoment0[4];
    float phaseMoment1[4];
    float phaseMoment2[4];
};
static_assert(offsetof(MultiScatteringConstants, outputUav) == 96);
static_assert(offsetof(MultiScatteringConstants, width) == 100);
static_assert(offsetof(MultiScatteringConstants, height) == 104);
static_assert(offsetof(MultiScatteringConstants, stepCount) == 108);
static_assert(offsetof(MultiScatteringConstants, transmittanceLutSize) == 112);
static_assert(offsetof(MultiScatteringConstants, transmittanceLutSrv) == 120);
static_assert(offsetof(MultiScatteringConstants, transferUav) == 124);
static_assert(offsetof(MultiScatteringConstants, transferSrv) == 128);
static_assert(offsetof(MultiScatteringConstants, pad0) == 132);
static_assert(offsetof(MultiScatteringConstants, pad1) == 136);
static_assert(offsetof(MultiScatteringConstants, pad2) == 140);
static_assert(offsetof(MultiScatteringConstants, phaseMoment0) == 144);
static_assert(offsetof(MultiScatteringConstants, phaseMoment1) == 160);
static_assert(offsetof(MultiScatteringConstants, phaseMoment2) == 176);
static_assert(sizeof(MultiScatteringConstants) == 192);
}
