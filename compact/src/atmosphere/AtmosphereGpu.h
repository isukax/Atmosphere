#pragma once
#include <cstdint>
namespace compact::atmosphere {
struct AtmosphereGpu {
    float bottomRadiusKm;
    float topRadiusKm;
    float rayleighScaleHeightKm;
    float mieScaleHeightKm;
    float rayleighScatteringPerKm[3];
    float ozoneCenterKm;
    float mieScatteringPerKm[3];
    float ozoneHalfWidthKm;
    float mieExtinctionPerKm[3];
    float sunAngularRadiusRad;
    float ozoneAbsorptionPerKm[3];
    uint32_t miePhaseTableSrv;
    float groundAlbedo[3];
    float pad;
};
static_assert(sizeof(AtmosphereGpu)==6*16);
}
