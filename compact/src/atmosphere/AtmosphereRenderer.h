#pragma once
#include "atmosphere/AtmosphereGpu.h"
#include "atmosphere/SkyPassConstants.h"
#include "gfx/Buffer.h"
#include "gfx/Pipeline.h"
#include "gfx/Texture.h"
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace compact::atmosphere {
float defaultSolarIlluminanceLux();

struct AtmosphereSettings {
    float aod550 = 0.1f;
    glm::vec3 sunDirectionWs{0.70710678f, 0.70710678f, 0.0f}; // Toward the sun, normalized by setSettings.
    float sunIlluminanceLux = defaultSolarIlluminanceLux();   // At the atmosphere top, normal to the sun.
    glm::vec3 groundAlbedo{0.3f}; // Planet model; independent of the sample terrain material.
    float sunAngularRadiusRad = 0.004675f;
};

struct AtmosphereQuality {
    uint32_t transmittanceWidth = 256, transmittanceHeight = 64;
    uint32_t multiScatteringSize = 32;
    uint32_t skyViewWidth = 192, skyViewHeight = 108;
    uint32_t aerialPerspectiveSize = 32;
    uint32_t transmittanceSteps = 500, multiScatteringSteps = 40, skyViewSteps = 30;
    uint32_t apMaxSteps = 64, perPixelSteps = 32, bisectionSteps = 6;
    bool operator==(const AtmosphereQuality &) const = default;
};

struct ShadowInput {
    uint32_t mapSrv = gfx::kInvalidDescriptorIndex;
    glm::mat4 cameraRelativeToUvz{1.0f}; // Camera-relative meters -> UV / Reverse-Z depth.
    uint32_t nearMapSrv = gfx::kInvalidDescriptorIndex;
    glm::mat4 nearCameraRelativeToUvz{1.0f};
    bool valid() const {
        return mapSrv != gfx::kInvalidDescriptorIndex;
    }
};

struct AtmosphereFrameInput {
    glm::vec3 cameraPosM{0.0f, 2.0f, 0.0f};
    glm::mat4 invViewProjRot{1.0f}; // Inverse of projection * rotation-only view (meters).
    float exposureScale = 1.0f;
    uint32_t depthSrv = gfx::kInvalidDescriptorIndex; // 0 depth means sky. Invalid means sky-only.
    ShadowInput shadow;
};

inline glm::vec3 worldToPlanetKm(const glm::vec3 &positionM) {
    return positionM * 0.001f + glm::vec3(0.0f, 6360.0f, 0.0f);
}

// Fixed Hillaire2020 / C+A+B / SH L2+ground / mean AP / per-pixel shadow.
// Owns resources; the caller owns the output, depth, shadow maps and command list.
class AtmosphereRenderer {
  public:
    AtmosphereRenderer(gfx::Device &, const gfx::PipelineLibrary &,
                       DXGI_FORMAT lutFormat = DXGI_FORMAT_R16G16B16A16_FLOAT);
    void setSettings(const AtmosphereSettings &);
    // Call before beginFrame. Size changes wait for GPU idle before recycling descriptors.
    void setQuality(const AtmosphereQuality &);
    const AtmosphereSettings &settings() const {
        return m_settings;
    }
    const AtmosphereQuality &quality() const {
        return m_quality;
    }
    struct Updates {
        bool transmittance, multiScattering;
    };
    Updates updateLuts(ID3D12GraphicsCommandList *, gfx::UploadRing &, const AtmosphereFrameInput &);
    void updateAerialPerspective(ID3D12GraphicsCommandList *, gfx::UploadRing &, const AtmosphereFrameInput &);
    void updateSkyIrradiance(ID3D12GraphicsCommandList *, gfx::UploadRing &, const AtmosphereFrameInput &);
    void composite(ID3D12GraphicsCommandList *, gfx::UploadRing &, const AtmosphereFrameInput &, uint32_t outputUav,
                   uint32_t width, uint32_t height);
    AtmosphereGpu gpuParameters() const;
    glm::vec3 sunColor() const;

    gfx::Texture &transmittanceLut() {
        return m_transmittance;
    }
    gfx::Texture &multiScatteringShLut() {
        return m_multiScattering;
    }
    gfx::Texture &transferTable() {
        return m_transfer;
    }
    gfx::Texture &skyViewLut() {
        return m_sky;
    }
    gfx::Texture &skyViewMieLut() {
        return m_skyMie;
    }
    gfx::Texture &aerialPerspective() {
        return m_ap;
    }
    gfx::Texture &aerialPerspectiveMie() {
        return m_apMie;
    }
    gfx::Texture &aerialPerspectiveMultiple() {
        return m_apMs;
    }
    gfx::Buffer &skyIrradiance() {
        return m_irradiance;
    } // 9 float4, cosine-convolved, per unit solar lux.

  private:
    void createResources();
    SkyPassConstants constants(const AtmosphereFrameInput &, uint32_t, uint32_t, uint32_t) const;
    void dispatch(ID3D12GraphicsCommandList *, gfx::UploadRing &, ID3D12PipelineState *, const SkyPassConstants &,
                  uint32_t, uint32_t, uint32_t);
    gfx::Device &m_device;
    const gfx::PipelineLibrary &m_pipelines;
    DXGI_FORMAT m_format;
    AtmosphereSettings m_settings;
    AtmosphereQuality m_quality;
    bool m_transmittanceDirty = true, m_multiScatteringDirty = true;
    ComPtr<ID3D12PipelineState> m_transmittancePso, m_transferPso, m_multiScatteringPso, m_skyPso, m_apPso,
        m_irradiancePso, m_compositePso;
    gfx::Texture m_transmittance, m_transfer, m_multiScattering, m_sky, m_skyMie, m_ap, m_apMie, m_apMs;
    gfx::Buffer m_phase, m_irradiance;
};
} // namespace compact::atmosphere
