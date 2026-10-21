#pragma once
#include "atmosphere/AtmosphereGpu.h"
#include <cstddef>
namespace compact::atmosphere {
struct TransmittanceConstants {
    AtmosphereGpu atmosphere;
    uint32_t outputUav;
    uint32_t width;
    uint32_t height;
    uint32_t stepCount;
};
static_assert(offsetof(TransmittanceConstants, outputUav) == 96);
static_assert(offsetof(TransmittanceConstants, width) == 100);
static_assert(offsetof(TransmittanceConstants, height) == 104);
static_assert(offsetof(TransmittanceConstants, stepCount) == 108);
static_assert(sizeof(TransmittanceConstants) == 112);
}
