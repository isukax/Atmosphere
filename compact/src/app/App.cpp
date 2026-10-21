#include "app/App.h"
#include <algorithm>
#include <cmath>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>
#include <numbers>
namespace compact::app {
namespace {
constexpr auto kDisplayPollInterval = std::chrono::seconds(2);
}
float App::exposureScale() const {
    // [POINT] 露出の規約（design.md D8、GraphicsTest/HDR と同一）:
    //         k = 1 / (1.2 · 2^EV100) [1/(cd/m²)]。1.2·2^EV100 [cd/m²] がセンサー飽和輝度で、
    //         これが fb 1.0（= 表示 100 nit）に写る。EV100 = 15（晴天日中）なら約 39,300 cd/m²。
    return 1.0f / (1.2f * std::exp2(m_exposureEv100));
}

void App::refreshDisplay() {
    const bool factoryRecreated = m_device.refreshFactoryIfStale();
    const display::DisplayState state = display::queryDisplayState(m_device.factory(), m_window.hwnd());
    m_lastDisplayPoll = std::chrono::steady_clock::now();
    if (state == m_display && !factoryRecreated) {
        return;
    }
    if (state != m_display) {
        logEvent(std::format("Display: {} HDR={} peak={:.0f} nit fullframe={:.0f} nit min={:.4f} nit ({})",
                             state.deviceName, state.hdrEnabled ? "on" : "off", state.maxLuminanceNits,
                             state.maxFullFrameLuminanceNits, state.minLuminanceNits,
                             display::toString(state.colorSpace)));
    } else {
        logEvent("DXGI factory was stale and has been recreated");
    }
    m_display = state;
    applyOutputConfiguration(true);
}

void App::applyOutputConfiguration(bool forceRetry) {
    // 1) 出力モード（スワップチェーンの形式と色空間）
    display::OutputMode desired = display::OutputMode::SDR;
    switch (m_outputSetting) {
    case OutputModeSetting::Auto:
        desired = m_display.hdrEnabled ? display::OutputMode::HDR10 : display::OutputMode::SDR;
        break;
    case OutputModeSetting::SDR:
        desired = display::OutputMode::SDR;
        break;
    case OutputModeSetting::HDR10:
        desired = display::OutputMode::HDR10;
        break;
    }
    // [POINT] HDR10 を要求して失敗した（OS の HDR が OFF）場合に毎フレーム作り直さないよう、
    //         「最後に要求したモード」と比べる。ディスプレイ状態が変わったときは forceRetry で再挑戦する。
    if (forceRetry || m_requestedMode != desired) {
        m_requestedMode = desired;
        if (desired != m_swapchain->mode() || forceRetry) {
            m_device.waitIdle();
            const bool ok = m_swapchain->setMode(desired);
            logEvent(std::format("Output mode: requested {} -> {}{}", display::toString(desired),
                                 display::toString(m_swapchain->mode()), ok ? "" : " (color space rejected)"));
        }
    }

    // 2) GT7 のピーク輝度
    // [POINT] Auto では OS が報告する MaxLuminance（小領域のピーク）を使う。報告値は不正確な
    //         ことがあるので、手動値（ゲーム機の HDR キャリブレーションに相当）で上書きできる。
    //         SDR 出力ではピークは使わず、GT7 の SDR モード（250 nit = 白）になる。
    float peak =
        m_peakAuto ? (m_display.maxLuminanceNits > 0.0f ? m_display.maxLuminanceNits : 1000.0f) : m_manualPeakNits;
    peak = std::clamp(peak, display::kGTMinPeakNits, display::kGTMaxPeakNits);
    const display::GT7Mode gt7Mode =
        m_swapchain->mode() == display::OutputMode::SDR ? display::GT7Mode::SDR : display::GT7Mode::HDR;
    if (gt7Mode != m_gt7Mode || peak != m_peakNits || forceRetry) {
        m_gt7Mode = gt7Mode;
        m_peakNits = peak;
        m_gt7 = display::makeGT7Constants(gt7Mode, peak);
        const float average =
            m_display.maxFullFrameLuminanceNits > 0.0f ? std::min(m_display.maxFullFrameLuminanceNits, peak) : peak;
        m_swapchain->setHdrMetadata(peak, average);
        logEvent(gt7Mode == display::GT7Mode::SDR
                     ? std::string("GT7: SDR (paper white 250 nit)")
                     : std::format("GT7: HDR peak {:.0f} nit ({})", peak, m_peakAuto ? "auto" : "manual"));
    }
}

void App::updateCamera() {
    const ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        constexpr float kRadPerPixel = 0.003f;
        m_camera.yawRad += io.MouseDelta.x * kRadPerPixel;
        m_camera.pitchRad = std::clamp(m_camera.pitchRad - io.MouseDelta.y * kRadPerPixel, -1.55f, 1.55f);
    }
    if (!io.WantCaptureKeyboard) {
        glm::vec3 move(0.0f);
        move += m_camera.forward() * static_cast<float>(ImGui::IsKeyDown(ImGuiKey_W) - ImGui::IsKeyDown(ImGuiKey_S));
        move += m_camera.right() * static_cast<float>(ImGui::IsKeyDown(ImGuiKey_D) - ImGui::IsKeyDown(ImGuiKey_A));
        move.y += static_cast<float>(ImGui::IsKeyDown(ImGuiKey_E) - ImGui::IsKeyDown(ImGuiKey_Q));
        const float speed = m_cameraSpeedMps * (ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 20.0f : 1.0f);
        m_camera.positionM += move * (speed * io.DeltaTime);
    }
    // [POINT] Phase 2 ではカメラを大気の内側（地表 1 m 〜 大気上端の 1 m 下）に制限する。
    //         宇宙視点は範囲外（design.md §11）。
    m_camera.positionM.y = std::clamp(m_camera.positionM.y, 1.0f, 99999.0f);
}

App::App(bool debugLayer, bool hidden)
    : m_window(L"Atmosphere compact", 1600, 900, hidden), m_device(debugLayer),
      m_pipelines(m_device, COMPACT_SHADER_DIR), m_atmosphere(m_device, m_pipelines), m_scene(m_device, m_pipelines),
      m_shadow(m_device), m_displayPass(m_pipelines) {
    m_display = display::queryDisplayState(m_device.factory(), m_window.hwnd());
    m_swapchain = std::make_unique<display::Swapchain>(m_device, m_window.hwnd(), m_window.width(), m_window.height(),
                                                       m_display.hdrEnabled ? display::OutputMode::HDR10
                                                                            : display::OutputMode::SDR);
    m_imgui = std::make_unique<ui::ImGuiLayer>(m_device, m_window.hwnd());
    createSizeDependentResources();
    applyOutputConfiguration(true);
}
App::~App() {
    m_device.waitIdle();
}

void App::createSizeDependentResources() {
    const auto w = std::max(m_window.width(), 1u), h = std::max(m_window.height(), 1u);
    m_sceneColor = gfx::Texture(m_device, {w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, true, true, true, L"Scene"});
    m_depth = gfx::Texture(m_device, {w, h, DXGI_FORMAT_D32_FLOAT, true, false, false, L"Depth", 1, true});
    m_uiTarget = gfx::Texture(m_device, {w, h, ui::ImGuiLayer::kUiFormat, true, false, true, L"UI"});
}

int App::run(uint32_t frameLimit, bool smoke) {
    uint32_t frame = 0;
    while (m_window.pumpMessages() && (!frameLimit || frame < frameLimit)) {
        if (m_window.minimized()) {
            Sleep(16);
            continue;
        }
        if (m_window.consumeResized()) {
            m_device.waitIdle();
            m_swapchain->resize(m_window.width(), m_window.height());
            createSizeDependentResources();
        }
        if (m_window.consumeDisplayEvent() || !m_device.factory()->IsCurrent() ||
            std::chrono::steady_clock::now() - m_lastDisplayPoll > kDisplayPollInterval)
            refreshDisplay();
        if (smoke) {
            m_vsync = false;
            m_scene.showTerrain = (frame & 1u) != 0;
            m_scene.showSpheres = (frame & 2u) != 0;
            m_outputSetting = frame % 2 ? OutputModeSetting::HDR10 : OutputModeSetting::SDR;
            applyOutputConfiguration(false);
            auto q = m_atmosphere.quality();
            const uint32_t scale = frame % 3;
            q.transmittanceWidth = 128u << scale;
            q.transmittanceHeight = 32u << scale;
            q.multiScatteringSize = 16u << scale;
            q.skyViewWidth = 96u << scale;
            q.skyViewHeight = 54u << scale;
            q.aerialPerspectiveSize = 16u << scale;
            m_atmosphere.setQuality(q);
            m_pendingQuality = q;
            if (frame == 2 || frame == 5)
                SetWindowPos(m_window.hwnd(), nullptr, 0, 0, frame == 2 ? 960 : 1280, frame == 2 ? 640 : 800,
                             SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
        renderFrame();
        ++frame;
    }
    m_device.waitIdle();
    ComPtr<ID3D12InfoQueue> messages;
    if (SUCCEEDED(m_device.d3d()->QueryInterface(IID_PPV_ARGS(&messages)))) {
        for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
            SIZE_T size = 0;
            messages->GetMessage(i, nullptr, &size);
            std::vector<uint8_t> storage(size);
            auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
            messages->GetMessage(i, message, &size);
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
                throw std::runtime_error(message->pDescription);
        }
    }
    return 0;
}

void App::renderFrame() {
    m_imgui->newFrame();
    updateCamera();
    drawUi();
    if (m_applyQuality) {
        m_atmosphere.setQuality(m_pendingQuality);
        m_applyQuality = false;
    }
    auto *cmd = m_device.beginFrame();
    auto &upload = m_device.upload();
    const auto w = m_sceneColor.desc().width, h = m_sceneColor.desc().height;
    const float aspect = float(w) / float(h);
    const auto settings = m_atmosphere.settings();
    atmosphere::AtmosphereFrameInput frame;
    frame.cameraPosM = m_camera.positionM;
    frame.invViewProjRot = glm::inverse(m_camera.projection(aspect) * m_camera.viewRotation());
    frame.exposureScale = exposureScale();
    m_atmosphere.updateLuts(cmd, upload, frame);
    m_shadow.updateFar(cmd, upload, m_scene, settings.sunDirectionWs);
    m_shadow.updateNear(cmd, upload, m_scene, m_camera, aspect, m_shadowSettings);
    frame.shadow = m_shadow.atmosphereShadow(m_camera.positionM, m_shadowSettings);
    m_atmosphere.updateAerialPerspective(cmd, upload, frame);
    m_atmosphere.updateSkyIrradiance(cmd, upload, frame);
    m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_depth.transition(cmd, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    const float clear[4] = {};
    cmd->ClearRenderTargetView(m_sceneColor.rtv(), clear, 0, nullptr);
    cmd->ClearDepthStencilView(m_depth.dsv(), D3D12_CLEAR_FLAG_DEPTH, 0, 0, 0, nullptr);
    render::SceneView view;
    view.viewProjRot = m_camera.projection(aspect) * m_camera.viewRotation();
    view.cameraPosM = m_camera.positionM;
    view.exposureScale = frame.exposureScale;
    view.shadow = m_shadow.sceneShadow(m_camera.positionM, m_shadowSettings);
    m_scene.record(cmd, upload, m_sceneColor.rtv(), m_sceneColor.desc().format, m_depth.dsv(), w, h, view, settings,
                   m_atmosphere);
    frame.depthSrv = m_depth.srvIndex();
    m_depth.transition(cmd, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    m_atmosphere.composite(cmd, upload, frame, m_sceneColor.uavIndex(), w, h);
    m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    m_uiTarget.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmd->ClearRenderTargetView(m_uiTarget.rtv(), clear, 0, nullptr);
    m_imgui->render(cmd, m_uiTarget.rtv());
    m_uiTarget.transition(cmd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    display::DisplayPassInput display;
    display.sceneColorSrv = m_sceneColor.srvIndex();
    display.uiSrv = m_uiTarget.srvIndex();
    display.gt7 = m_gt7;
    display.uiWhiteNits = m_uiWhiteNits;
    display.frameIndex = uint32_t(m_device.frameNumber());
    display.encode = m_swapchain->mode() == display::OutputMode::SDR ? display::OutputEncode::SdrSrgb
                                                                     : display::OutputEncode::Hdr10Pq;
    m_swapchain->transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_displayPass.record(cmd, upload, m_swapchain->rtv(), m_swapchain->format(), w, h, display);
    m_swapchain->transition(cmd, D3D12_RESOURCE_STATE_PRESENT);
    m_device.endFrame();
    m_swapchain->present(m_vsync);
}

void App::drawUi() {
    ImGui::SetNextWindowSize(ImVec2(420, 820), ImGuiCond_FirstUseEver);
    ImGui::Begin("Atmosphere compact");
    ImGui::TextUnformatted("Hillaire2020 / C+A+B / SH L2 + ground");
    ImGui::TextUnformatted("AP: RGB mean / Per Pixel / Sun disk");
    auto settings = m_atmosphere.settings();
    bool changed = false;
    if (ImGui::CollapsingHeader("Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
        changed |= ImGui::SliderFloat("AOD 550", &settings.aod550, 0.001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SameLine();
        if (ImGui::Button("Reset AOD")) {
            settings.aod550 = 0.1f;
            changed = true;
        }
        changed |= ImGui::SliderFloat("Sun zenith [deg]", &m_sunZenithDeg, 0, 120);
        changed |= ImGui::SliderFloat("Sun azimuth [deg]", &m_sunAzimuthDeg, -180, 180);
        changed |= ImGui::DragFloat("Sun [lx]", &settings.sunIlluminanceLux, 100, 0, 1e6f, "%.0f");
        if (ImGui::Button("Reset illuminance")) {
            settings.sunIlluminanceLux = atmosphere::defaultSolarIlluminanceLux();
            changed = true;
        }
        float radius = glm::degrees(settings.sunAngularRadiusRad);
        if (ImGui::SliderFloat("Sun radius [deg]", &radius, 0.05f, 5.0f, "%.3f")) {
            settings.sunAngularRadiusRad = glm::radians(radius);
            changed = true;
        }
        changed |= ImGui::SliderFloat3("Planet ground albedo", &settings.groundAlbedo.x, 0, 1);
        ImGui::SliderFloat("EV100", &m_exposureEv100, -6, 20);
    }
    if (changed) {
        const float theta = glm::radians(m_sunZenithDeg), phi = glm::radians(m_sunAzimuthDeg);
        settings.sunDirectionWs = {std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
        m_atmosphere.setSettings(settings);
    }
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Terrain", &m_scene.showTerrain);
        ImGui::Checkbox("Spheres", &m_scene.showSpheres);
        ImGui::SliderFloat("Terrain material albedo", &m_scene.terrainAlbedo, 0, 1);
    }
    if (ImGui::CollapsingHeader("Camera")) {
        ImGui::TextUnformatted("RMB: look / WASD QE: move / Shift: x20");
        ImGui::SliderFloat("Altitude [m]", &m_camera.positionM.y, 1, 99000, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Speed [m/s]", &m_cameraSpeedMps, 1, 10000, "%.0f", ImGuiSliderFlags_Logarithmic);
        float fov = glm::degrees(m_camera.verticalFovRad);
        if (ImGui::SliderFloat("FOV [deg]", &fov, 10, 120))
            m_camera.verticalFovRad = glm::radians(fov);
    }
    if (ImGui::CollapsingHeader("Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto &q = m_pendingQuality;
        auto size = [](const char *label, uint32_t &w, uint32_t &h, uint32_t bw, uint32_t bh,
                       const char *resolutions) {
            int index = w == bw / 2 ? 0 : w == bw ? 1 : 2;
            if (ImGui::Combo(label, &index, resolutions)) {
                w = (bw / 2) << index;
                h = (bh / 2) << index;
            }
        };
        size("Transmittance size", q.transmittanceWidth, q.transmittanceHeight, 256, 64,
             "128 x 32\0" "256 x 64\0" "512 x 128\0");
        uint32_t msHeight = q.multiScatteringSize;
        size("MS tile size", q.multiScatteringSize, msHeight, 32, 32, "16 x 16\0" "32 x 32\0" "64 x 64\0");
        size("Sky-View size", q.skyViewWidth, q.skyViewHeight, 192, 108,
             "96 x 54\0" "192 x 108\0" "384 x 216\0");
        uint32_t apHeight = q.aerialPerspectiveSize;
        size("AP size (cubic)", q.aerialPerspectiveSize, apHeight, 32, 32,
             "16 x 16 x 16\0" "32 x 32 x 32\0" "64 x 64 x 64\0");
        auto steps = [](const char *label, uint32_t &v, int lo, int hi) {
            int x = int(v);
            if (ImGui::SliderInt(label, &x, lo, hi))
                v = uint32_t(x);
        };
        steps("Transmittance steps", q.transmittanceSteps, 4, 1000);
        steps("MS steps", q.multiScatteringSteps, 4, 256);
        steps("Sky-View steps", q.skyViewSteps, 4, 256);
        steps("AP far slice steps", q.apMaxSteps, 4, 256);
        steps("Per-pixel steps", q.perPixelSteps, 4, 128);
        steps("Bisection steps", q.bisectionSteps, 0, 12);
        if (ImGui::Button("Apply quality"))
            m_applyQuality = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset quality")) {
            q = {};
            m_applyQuality = true;
        }
        if (q != m_atmosphere.quality())
            ImGui::TextUnformatted("Pending changes");
    }
    if (ImGui::CollapsingHeader("Display")) {
        int mode = int(m_outputSetting);
        bool outputChanged = ImGui::Combo("Output", &mode, "Auto\0SDR\0HDR10\0");
        m_outputSetting = OutputModeSetting(mode);
        outputChanged |= ImGui::Checkbox("Peak: use OS value", &m_peakAuto);
        ImGui::BeginDisabled(m_peakAuto);
        outputChanged |= ImGui::SliderFloat("Peak [nit]", &m_manualPeakNits, display::kGTMinPeakNits,
                                            display::kGTMaxPeakNits, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        if (outputChanged)
            applyOutputConfiguration(false);
        ImGui::SliderFloat("UI white [nit]", &m_uiWhiteNits, 80, 500);
        ImGui::Checkbox("VSync", &m_vsync);
        ImGui::Text("%s / %.0f nit", display::toString(m_swapchain->mode()), m_peakNits);
    }
    ImGui::End();
}
} // namespace compact::app
