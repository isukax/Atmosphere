#pragma once
#include "app/Window.h"
#include "atmosphere/AtmosphereRenderer.h"
#include "display/DisplayMonitor.h"
#include "display/DisplayPass.h"
#include "display/Swapchain.h"
#include "render/SceneRenderer.h"
#include "render/ShadowRenderer.h"
#include "scene/Camera.h"
#include "ui/ImGuiLayer.h"
#include <chrono>
#include <memory>
#include <optional>

namespace compact::app {
class App {
  public:
    App(bool debugLayer, bool hidden = false);
    ~App();
    int run(uint32_t frameLimit = 0, bool smoke = false);

  private:
    enum class OutputModeSetting { Auto, SDR, HDR10 };
    void createSizeDependentResources();
    void refreshDisplay();
    void applyOutputConfiguration(bool);
    void renderFrame();
    void drawUi();
    void updateCamera();
    void logEvent(const std::string &s) {
        gfx::log(s);
    }
    float exposureScale() const;
    Window m_window;
    gfx::Device m_device;
    gfx::PipelineLibrary m_pipelines;
    atmosphere::AtmosphereRenderer m_atmosphere;
    render::SceneRenderer m_scene;
    render::ShadowRenderer m_shadow;
    std::unique_ptr<display::Swapchain> m_swapchain;
    std::unique_ptr<ui::ImGuiLayer> m_imgui;
    display::DisplayPass m_displayPass;
    gfx::Texture m_sceneColor, m_depth, m_uiTarget;
    scene::Camera m_camera;
    render::ShadowSettings m_shadowSettings;
    atmosphere::AtmosphereQuality m_pendingQuality;
    OutputModeSetting m_outputSetting = OutputModeSetting::Auto;
    bool m_peakAuto = true, m_vsync = true, m_applyQuality = false;
    float m_manualPeakNits = 1000, m_uiWhiteNits = 200, m_exposureEv100 = 15, m_cameraSpeedMps = 100;
    float m_sunZenithDeg = 45, m_sunAzimuthDeg = 0;
    display::DisplayState m_display;
    std::optional<display::OutputMode> m_requestedMode;
    float m_peakNits = 0;
    display::GT7Mode m_gt7Mode = display::GT7Mode::SDR;
    display::GT7Constants m_gt7{};
    std::chrono::steady_clock::time_point m_lastDisplayPoll{};
};
} // namespace compact::app
