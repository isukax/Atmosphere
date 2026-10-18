#pragma once

#include "app/Window.h"
#include "atmosphere/AtmosphereRenderer.h"
#include "atmosphere/MiePhaseTables.h"
#include "atmosphere/PathTracer.h"
#include "display/DisplayMonitor.h"
#include "display/DisplayPass.h"
#include "display/Swapchain.h"
#include "gfx/GpuTimer.h"
#include "gfx/Pipeline.h"
#include "gfx/Texture.h"
#include "render/SceneRenderer.h"
#include "render/ShadowRenderer.h"
#include "scene/Camera.h"
#include "scene/TestPatternPass.h"
#include "ui/ImGuiLayer.h"

#include <imgui.h>

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace app {

/// 対話モードのアプリ本体（テストパターン / 空: LUT・レイマーチ・パストレース + 表示パイプライン）。
class App {
public:
    explicit App(bool debugLayer);
    ~App();
    int run();

private:
    enum class OutputModeSetting { Auto, SDR, HDR10 };
    enum class SceneMode { TestPattern, SkyLut, SkyRaymarch, SkyPathTrace };

    void createSizeDependentResources();
    void refreshDisplay();
    void applyOutputConfiguration(bool forceRetry);
    void renderFrame();
    void drawUi();
    void logEvent(const std::string& message);
    float exposureScale() const;
    void updateCamera();
    void applyCoefficientSet(int index);
    void applyAerosolPreset(const std::string& name);
    void drawAtmosphereUi();
    void drawPixelPickerUi(const ImVec2& pos, float width);
    void drawGpuTimeUi(const ImVec2& pos, float width);
    void drawEventsUi(const ImVec2& pos, float width);
    void drawTextureViewUi(const ImVec2& pos);
    bool isSkyMode() const { return m_sceneMode != SceneMode::TestPattern; }
    void resetPathTraceIfChanged();

    Window m_window;
    gfx::Device m_device;
    gfx::PipelineLibrary m_pipelines;
    std::unique_ptr<display::Swapchain> m_swapchain;
    std::unique_ptr<ui::ImGuiLayer> m_imgui;
    display::DisplayPass m_displayPass;
    scene::TestPatternPass m_testPattern;
    std::unique_ptr<atmosphere::AtmosphereRenderer> m_atmosphereRenderer;
    std::unique_ptr<atmosphere::PathTracer> m_pathTracer;
    std::unique_ptr<render::SceneRenderer> m_sceneRenderer;
    std::unique_ptr<render::ShadowRenderer> m_shadowRenderer;
    std::unique_ptr<gfx::GpuTimer> m_gpuTimer;
    std::vector<gfx::GpuTimer::Result> m_gpuTimes;

    gfx::Texture m_sceneColor;  // RGBA16F、fb 値（露出済み）、linear 作業色空間
    gfx::Texture m_depth;       // D32（Reverse-Z）、地形・球の深度。空気遠近法の合成で距離に戻す
    gfx::Texture m_uiTarget;    // R8G8B8A8_UNORM、sRGB 値、アルファ乗算済み

    // 設定（UI で変更）
    OutputModeSetting m_outputSetting = OutputModeSetting::Auto;
    bool m_peakAuto = true;
    float m_manualPeakNits = 1000.0f;
    float m_uiWhiteNits = 200.0f;
    float m_exposureEv100 = 15.0f;
    display::WorkingColorSpace m_workingSpace = display::WorkingColorSpace::Rec709;
    scene::TestPatternSettings m_testPatternSettings;
    SceneMode m_sceneMode = SceneMode::SkyLut;

    // 大気
    std::vector<atmosphere::CoefficientSet> m_coefficientSets;
    std::unique_ptr<atmosphere::MiePhaseTables> m_miePhaseTables;  // プリセットの位相関数の表（GPU、design.md §5.9）
    int m_coefficientIndex = 0;
    atmosphere::AtmosphereParams m_atmosphere;
    atmosphere::AtmosphereQuality m_quality;  // LUT のステップ数・多重散乱の有無（既定値の理由は定義側）
    atmosphere::RaymarchSettings m_raymarch;
    bool m_sunDisk = true;
    bool m_showScene = true;      // 地形・テスト球を描く（パストレースのモードでは描かない）
    bool m_skyIrradiance = true;  // 地形に空からの拡散照度を加える
    render::ShadowSettings m_shadow;  // 太陽のシャドウマップ（design.md §5.5）
    atmosphere::AerialPerspectiveShadowOptions m_apShadow;  // 空気遠近法の影の方式（design.md §5.6）
    // 空気遠近法の透過率の求め方（design.md §5.7）
    atmosphere::AerialPerspectiveTransmittanceMode m_apTransmittance = atmosphere::SkyView{}.apTransmittance;
    uint32_t m_pathTraceMaxDepth = 32;
    uint32_t m_pathTraceSamplesPerFrame = 1;
    std::vector<uint8_t> m_pathTraceKey;  // 累積をリセットすべき入力（カメラ・大気・設定）のバイト列
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
