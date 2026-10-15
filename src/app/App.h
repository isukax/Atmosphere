#pragma once

#include "app/Window.h"
#include "atmosphere/AtmosphereRenderer.h"
#include "display/DisplayMonitor.h"
#include "display/DisplayPass.h"
#include "display/Swapchain.h"
#include "gfx/Pipeline.h"
#include "gfx/Texture.h"
#include "scene/Camera.h"
#include "scene/TestPatternPass.h"
#include "ui/ImGuiLayer.h"

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace app {

/// 対話モードのアプリ本体（Phase 1: テストパターン + 表示パイプライン）。
class App {
public:
    explicit App(bool debugLayer);
    ~App();
    int run();

private:
    enum class OutputModeSetting { Auto, SDR, HDR10 };
    enum class SceneMode { TestPattern, Sky };

    void createSizeDependentResources();
    void refreshDisplay();
    void applyOutputConfiguration(bool forceRetry);
    void renderFrame();
    void drawUi();
    void logEvent(const std::string& message);
    float exposureScale() const;
    void updateCamera();
    void applyCoefficientSet(int index);
    void drawAtmosphereUi();

    Window m_window;
    gfx::Device m_device;
    gfx::PipelineLibrary m_pipelines;
    std::unique_ptr<display::Swapchain> m_swapchain;
    std::unique_ptr<ui::ImGuiLayer> m_imgui;
    display::DisplayPass m_displayPass;
    scene::TestPatternPass m_testPattern;
    std::unique_ptr<atmosphere::AtmosphereRenderer> m_atmosphereRenderer;

    gfx::Texture m_sceneColor;  // RGBA16F、fb 値（露出済み）、linear 作業色空間
    gfx::Texture m_uiTarget;    // R8G8B8A8_UNORM、sRGB 値、アルファ乗算済み

    // 設定（UI で変更）
    OutputModeSetting m_outputSetting = OutputModeSetting::Auto;
    bool m_peakAuto = true;
    float m_manualPeakNits = 1000.0f;
    float m_uiWhiteNits = 200.0f;
    float m_exposureEv100 = 15.0f;
    display::WorkingColorSpace m_workingSpace = display::WorkingColorSpace::Rec709;
    scene::TestPatternSettings m_testPatternSettings;
    SceneMode m_sceneMode = SceneMode::Sky;

    // 大気（Phase 2: Transmittance LUT + 単散乱レイマーチ）
    std::vector<atmosphere::CoefficientSet> m_coefficientSets;
    int m_coefficientIndex = 0;
    atmosphere::AtmosphereParams m_atmosphere;
    atmosphere::RaymarchSettings m_raymarch;
    // [POINT] Transmittance LUT の積分区間数。論文の PC 設定は 40 だが、Phase 2 の検証（V1/V2）で
    //         地平線付近の誤差が最大 1.8%（空の輝度で 1.4%）と分かった。LUT は大気パラメータの変更時しか
    //         作り直さないので、既定を 500（誤差 0.09%）にする。
    uint32_t m_transmittanceSteps = 500;
    float m_solarIlluminanceFromSpectrumLux = 0.0f;
    scene::Camera m_camera;
    float m_cameraSpeedMps = 100.0f;
    bool m_vsync = true;

    // 状態
    display::DisplayState m_display;
    std::optional<display::OutputMode> m_requestedMode;
    float m_peakNits = 0.0f;
    display::GT7Mode m_gt7Mode = display::GT7Mode::SDR;
    display::GT7Constants m_gt7{};
    std::chrono::steady_clock::time_point m_lastDisplayPoll{};
    std::deque<std::string> m_events;

    // ピクセルピッカー: フレームスロットごとに 1 texel を読み戻す
    struct Picker {
        ComPtr<ID3D12Resource> buffer;
        int x = -1;
        int y = -1;
    };
    std::array<Picker, gfx::Device::kFramesInFlight> m_pickers;
    std::array<float, 4> m_pickedFb{};
    int m_pickedX = -1;
    int m_pickedY = -1;
};

} // namespace app
