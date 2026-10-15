#include "app/App.h"

#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/matrix.hpp>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace app {
namespace {

constexpr size_t kMaxEvents = 30;
// [POINT] OS の HDR キャリブレーション等でピーク輝度だけが変わった場合に IsCurrent() や
//         ウィンドウメッセージで検出できるかは確認できていないため、一定間隔でも再評価する。
constexpr auto kDisplayPollInterval = std::chrono::seconds(2);

const char* toString(display::GT7Mode mode)
{
    return mode == display::GT7Mode::SDR ? "SDR" : "HDR";
}

} // namespace

App::App(bool debugLayer)
    : m_window(L"Atmosphere", 1600, 900),
      m_device(debugLayer),
      m_pipelines(m_device, ATMOS_SHADER_DIR),
      m_displayPass(m_pipelines),
      m_testPattern(m_pipelines)
{
    m_display = display::queryDisplayState(m_device.factory(), m_window.hwnd());
    const display::OutputMode initialMode = m_display.hdrEnabled ? display::OutputMode::HDR10 : display::OutputMode::SDR;
    m_swapchain = std::make_unique<display::Swapchain>(m_device, m_window.hwnd(), m_window.width(), m_window.height(),
                                                       initialMode);
    m_imgui = std::make_unique<ui::ImGuiLayer>(m_device, m_window.hwnd());
    createSizeDependentResources();
    for (Picker& picker : m_pickers) {
        picker.buffer = gfx::createBuffer(m_device.d3d(), 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    logEvent(std::format("Display: {} HDR={} peak={:.0f} nit fullframe={:.0f} nit min={:.4f} nit ({})",
                         m_display.deviceName, m_display.hdrEnabled ? "on" : "off", m_display.maxLuminanceNits,
                         m_display.maxFullFrameLuminanceNits, m_display.minLuminanceNits,
                         display::toString(m_display.colorSpace)));
    applyOutputConfiguration(true);

    const std::filesystem::path coefficientPath = std::filesystem::path(ATMOS_DATA_DIR) / "atmosphere_coefficients.json";
    m_coefficientSets = atmosphere::loadCoefficientSets(coefficientPath);
    m_solarIlluminanceFromSpectrumLux = atmosphere::loadSolarIlluminanceFromSpectrumLux(coefficientPath);
    applyCoefficientSet(0);  // 先頭 = Hillaire2020
    m_atmosphereRenderer = std::make_unique<atmosphere::AtmosphereRenderer>(m_device, m_pipelines);
}

void App::applyCoefficientSet(int index)
{
    m_coefficientIndex = std::clamp(index, 0, static_cast<int>(m_coefficientSets.size()) - 1);
    m_atmosphere.applyCoefficientSet(m_coefficientSets[static_cast<size_t>(m_coefficientIndex)]);
    // [POINT] 係数セットは特定の作業色空間の RGB で定義されている（§2.6）。セットを選ぶと
    //         作業色空間もそれに合わせる（Rec.709 の係数を Rec.2020 として解釈しないため）。
    m_workingSpace = m_atmosphere.workingSpace;
    logEvent(std::format("Coefficient set: {} ({})", m_atmosphere.coefficientSet,
                         display::toString(m_workingSpace)));
}

void App::updateCamera()
{
    const ImGuiIO& io = ImGui::GetIO();
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
    const float topM = (m_atmosphere.topRadiusKm - m_atmosphere.bottomRadiusKm) * 1000.0f - m_atmosphere.originAltitudeM;
    m_camera.positionM.y = std::clamp(m_camera.positionM.y, 1.0f - m_atmosphere.originAltitudeM, topM - 1.0f);
}

App::~App()
{
    m_device.waitIdle();
}

float App::exposureScale() const
{
    // [POINT] 露出の規約（design.md D8、GraphicsTest/HDR と同一）:
    //         k = 1 / (1.2 · 2^EV100) [1/(cd/m²)]。1.2·2^EV100 [cd/m²] がセンサー飽和輝度で、
    //         これが fb 1.0（= 表示 100 nit）に写る。EV100 = 15（晴天日中）なら約 39,300 cd/m²。
    return 1.0f / (1.2f * std::exp2(m_exposureEv100));
}

void App::logEvent(const std::string& message)
{
    gfx::log(message);
    m_events.push_back(message);
    while (m_events.size() > kMaxEvents) {
        m_events.pop_front();
    }
}

void App::createSizeDependentResources()
{
    const uint32_t w = std::max(m_window.width(), 1u);
    const uint32_t h = std::max(m_window.height(), 1u);
    // [POINT] シーンカラーは RGBA16F。物理輝度ではなく露出済みの fb 値を入れるので FP16 で足りる
    //         （物理輝度のままだと太陽ディスク ~1.9×10⁹ cd/m² が FP16 の上限 65504 を超える）。
    m_sceneColor = gfx::Texture(m_device, {w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, true, true, false, L"SceneColor"});
    m_uiTarget = gfx::Texture(m_device, {w, h, ui::ImGuiLayer::kUiFormat, true, false, true, L"UiTarget"});
}

void App::refreshDisplay()
{
    const bool factoryRecreated = m_device.refreshFactoryIfStale();
    const display::DisplayState state = display::queryDisplayState(m_device.factory(), m_window.hwnd());
    m_lastDisplayPoll = std::chrono::steady_clock::now();
    if (state == m_display && !factoryRecreated) {
        return;
    }
    if (state != m_display) {
        logEvent(std::format("Display: {} HDR={} peak={:.0f} nit fullframe={:.0f} nit min={:.4f} nit ({})",
                             state.deviceName, state.hdrEnabled ? "on" : "off", state.maxLuminanceNits,
                             state.maxFullFrameLuminanceNits, state.minLuminanceNits, display::toString(state.colorSpace)));
    } else {
        logEvent("DXGI factory was stale and has been recreated");
    }
    m_display = state;
    applyOutputConfiguration(true);
}

void App::applyOutputConfiguration(bool forceRetry)
{
    // 1) 出力モード（スワップチェーンの形式と色空間）
    display::OutputMode desired = display::OutputMode::SDR;
    switch (m_outputSetting) {
    case OutputModeSetting::Auto:
        desired = m_display.hdrEnabled ? display::OutputMode::HDR10 : display::OutputMode::SDR;
        break;
    case OutputModeSetting::SDR: desired = display::OutputMode::SDR; break;
    case OutputModeSetting::HDR10: desired = display::OutputMode::HDR10; break;
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
    float peak = m_peakAuto ? (m_display.maxLuminanceNits > 0.0f ? m_display.maxLuminanceNits : 1000.0f)
                            : m_manualPeakNits;
    peak = std::clamp(peak, display::kGTMinPeakNits, display::kGTMaxPeakNits);
    const display::GT7Mode gt7Mode =
        m_swapchain->mode() == display::OutputMode::SDR ? display::GT7Mode::SDR : display::GT7Mode::HDR;
    if (gt7Mode != m_gt7Mode || peak != m_peakNits || forceRetry) {
        m_gt7Mode = gt7Mode;
        m_peakNits = peak;
        m_gt7 = display::makeGT7Constants(gt7Mode, peak);
        const float average = m_display.maxFullFrameLuminanceNits > 0.0f
                                  ? std::min(m_display.maxFullFrameLuminanceNits, peak)
                                  : peak;
        m_swapchain->setHdrMetadata(peak, average);
        logEvent(gt7Mode == display::GT7Mode::SDR
                     ? std::string("GT7: SDR (paper white 250 nit)")
                     : std::format("GT7: HDR peak {:.0f} nit ({})", peak, m_peakAuto ? "auto" : "manual"));
    }
}

int App::run()
{
    while (m_window.pumpMessages()) {
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
            std::chrono::steady_clock::now() - m_lastDisplayPoll > kDisplayPollInterval) {
            refreshDisplay();
        }
        renderFrame();
    }
    return 0;
}

void App::renderFrame()
{
    ID3D12GraphicsCommandList* cmd = m_device.beginFrame();
    const uint32_t slot = m_device.frameSlot();
    const uint32_t w = m_sceneColor.desc().width;
    const uint32_t h = m_sceneColor.desc().height;

    // このスロットで前回コピーしたピッカーの値（GPU 完了済み）を読む
    if (m_pickers[slot].x >= 0) {
        uint16_t* mapped = nullptr;
        const D3D12_RANGE range{0, 8};
        if (SUCCEEDED(m_pickers[slot].buffer->Map(0, &range, reinterpret_cast<void**>(&mapped)))) {
            for (int c = 0; c < 4; ++c) {
                m_pickedFb[c] = glm::unpackHalf1x16(mapped[c]);
            }
            const D3D12_RANGE none{0, 0};
            m_pickers[slot].buffer->Unmap(0, &none);
            m_pickedX = m_pickers[slot].x;
            m_pickedY = m_pickers[slot].y;
        }
    }

    m_imgui->newFrame();
    updateCamera();
    drawUi();

    // 1) シーン
    if (m_sceneMode == SceneMode::Sky) {
        // Transmittance LUT（媒質パラメータが変わったときだけ作り直す）→ 空の単散乱レイマーチ
        m_atmosphereRenderer->updateTransmittanceLut(cmd, m_device.upload(), m_atmosphere, m_transmittanceSteps);
        m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        atmosphere::RaymarchView view;
        const float aspect = static_cast<float>(w) / static_cast<float>(h);
        view.invViewProjRot = glm::inverse(m_camera.projection(aspect) * m_camera.viewRotation());
        view.cameraPosKm = atmosphere::worldToPlanetKm(m_atmosphere, m_camera.positionM);
        view.exposureScale = exposureScale();
        m_atmosphereRenderer->renderSkyRaymarch(cmd, m_device.upload(), m_atmosphere, view, m_raymarch,
                                                m_sceneColor.uavIndex(), w, h);
    } else {
        m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        m_testPattern.record(cmd, m_device.upload(), m_sceneColor.uavIndex(), w, h, exposureScale(), m_workingSpace,
                             m_testPatternSettings);
    }

    // 2) ピクセルピッカー: マウス位置の 1 texel を読み戻しバッファへ
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    m_pickers[slot].x = -1;
    if (mouse.x >= 0 && mouse.y >= 0 && mouse.x < w && mouse.y < h) {
        m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{m_sceneColor.resource(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        D3D12_TEXTURE_COPY_LOCATION dst{m_pickers[slot].buffer.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, 1, 1, 1, 256};
        const D3D12_BOX box{static_cast<UINT>(mouse.x), static_cast<UINT>(mouse.y), 0,
                            static_cast<UINT>(mouse.x) + 1, static_cast<UINT>(mouse.y) + 1, 1};
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        m_pickers[slot].x = static_cast<int>(mouse.x);
        m_pickers[slot].y = static_cast<int>(mouse.y);
    }
    m_sceneColor.transition(cmd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // 3) UI（専用 RT へ）
    m_uiTarget.transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
    const float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cmd->ClearRenderTargetView(m_uiTarget.rtv(), clear, 0, nullptr);
    m_imgui->render(cmd, m_uiTarget.rtv());
    m_uiTarget.transition(cmd, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // 4) 表示パス → バックバッファ
    display::DisplayPassInput input;
    input.sceneColorSrv = m_sceneColor.srvIndex();
    input.uiSrv = m_uiTarget.srvIndex();
    input.workingSpace = m_workingSpace;
    input.encode = m_swapchain->mode() == display::OutputMode::SDR ? display::OutputEncode::SdrSrgb
                                                                   : display::OutputEncode::Hdr10Pq;
    input.toneMap = display::ToneMapOperator::GT7;
    input.gt7 = m_gt7;
    input.exposureRescale = 1.0f;
    input.uiWhiteNits = m_uiWhiteNits;
    input.frameIndex = static_cast<uint32_t>(m_device.frameNumber());
    m_swapchain->transition(cmd, D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_displayPass.record(cmd, m_device.upload(), m_swapchain->rtv(), m_swapchain->format(), m_swapchain->width(),
                         m_swapchain->height(), input);
    m_swapchain->transition(cmd, D3D12_RESOURCE_STATE_PRESENT);

    m_device.endFrame();
    m_swapchain->present(m_vsync);
}

void App::drawUi()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Atmosphere - Phase 2");
    int sceneMode = static_cast<int>(m_sceneMode);
    if (ImGui::Combo("Scene", &sceneMode, "Test pattern\0Sky (single scattering raymarch)\0")) {
        m_sceneMode = static_cast<SceneMode>(sceneMode);
        if (m_sceneMode == SceneMode::Sky) {
            m_workingSpace = m_atmosphere.workingSpace;
        }
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Monitor: %s", m_display.deviceName.c_str());
        ImGui::Text("OS HDR: %s  (%s, %u bpc)", m_display.hdrEnabled ? "ON" : "OFF",
                    display::toString(m_display.colorSpace), m_display.bitsPerColor);
        ImGui::Text("Reported peak %.0f nit / full-frame %.0f nit / min %.4f nit", m_display.maxLuminanceNits,
                    m_display.maxFullFrameLuminanceNits, m_display.minLuminanceNits);
        ImGui::Separator();
        int setting = static_cast<int>(m_outputSetting);
        bool changed = ImGui::Combo("Output", &setting, "Auto\0SDR\0HDR10\0");
        m_outputSetting = static_cast<OutputModeSetting>(setting);
        changed |= ImGui::Checkbox("Peak: use OS value", &m_peakAuto);
        ImGui::BeginDisabled(m_peakAuto);
        changed |= ImGui::SliderFloat("Peak [nit]", &m_manualPeakNits, display::kGTMinPeakNits,
                                      display::kGTMaxPeakNits, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        if (changed) {
            applyOutputConfiguration(false);
        }
        ImGui::SliderFloat("UI white [nit]", &m_uiWhiteNits, 80.0f, 500.0f, "%.0f");
        ImGui::Checkbox("VSync", &m_vsync);
        ImGui::Text("Active: %s, GT7 %s, target %.0f nit", display::toString(m_swapchain->mode()),
                    toString(m_gt7Mode), m_gt7.targetFb * display::kGTReferenceLuminanceNits);
    }

    if (ImGui::CollapsingHeader("Exposure / Color", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("EV100", &m_exposureEv100, -6.0f, 20.0f, "%.2f");
        ImGui::Text("k = 1/(1.2*2^EV100) = %.3e /(cd/m2)", exposureScale());
        ImGui::Text("fb 1.0 (= 100 nit on display) <- %.4g cd/m2 in scene", 1.0f / exposureScale());
        // 空の表示中は作業色空間を係数セットに従わせる（Atmosphere セクションで選ぶ）
        ImGui::BeginDisabled(m_sceneMode == SceneMode::Sky);
        int space = static_cast<int>(m_workingSpace);
        if (ImGui::Combo("Working space", &space, "Rec.709\0Rec.2020\0")) {
            m_workingSpace = static_cast<display::WorkingColorSpace>(space);
        }
        ImGui::EndDisabled();
        if (m_sceneMode == SceneMode::TestPattern) {
            ImGui::DragFloatRange2("Pattern log10(cd/m2)", &m_testPatternSettings.log10MinNits,
                                   &m_testPatternSettings.log10MaxNits, 0.05f, -4.0f, 10.0f, "%.1f");
        }
    }

    if (m_sceneMode == SceneMode::Sky) {
        drawAtmosphereUi();
    }

    if (ImGui::CollapsingHeader("Pixel picker", ImGuiTreeNodeFlags_DefaultOpen) && m_pickedX >= 0) {
        // [POINT] ピッカーで単位の流れを確認できるようにする:
        //         シーン輝度 [cd/m²] → 露出 → fb → GT7 → 表示輝度 [nit]（HDR）/ SDR 白に対する割合（SDR）
        const glm::vec3 fb(m_pickedFb[0], m_pickedFb[1], m_pickedFb[2]);
        const float fbY = glm::dot(fb, display::luminanceWeights(m_workingSpace));
        ImGui::Text("Pixel (%d, %d)", m_pickedX, m_pickedY);
        ImGui::Text("Scene fb RGB  : %.4g, %.4g, %.4g", fb.x, fb.y, fb.z);
        ImGui::Text("Scene L       : %.4g cd/m2  (fb Y %.4g / k)", fbY / exposureScale(), fbY);
        const glm::vec3 rec2020 = display::mul(display::workingToRec2020(m_workingSpace), fb);
        const glm::vec3 mapped = display::applyGT7(m_gt7, rec2020);
        const float mappedY = glm::dot(mapped, display::luminanceWeights(display::WorkingColorSpace::Rec2020));
        if (m_gt7Mode == display::GT7Mode::HDR) {
            ImGui::Text("Display       : %.1f nit (Rec.2020 RGB %.1f, %.1f, %.1f nit)", mappedY * 100.0f,
                        mapped.x * 100.0f, mapped.y * 100.0f, mapped.z * 100.0f);
        } else {
            ImGui::Text("Display       : %.1f%% of SDR white (~%.1f nit at 250 nit white)", mappedY * 100.0f,
                        mappedY * display::kGTSdrPaperWhiteNits);
        }
    }

    if (ImGui::CollapsingHeader("Events")) {
        for (const std::string& e : m_events) {
            ImGui::TextUnformatted(e.c_str());
        }
    }
    ImGui::End();
}

void App::drawAtmosphereUi()
{
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled("RMB drag: look, WASD/QE: move, Shift: x20");
        float altitudeM = m_camera.positionM.y + m_atmosphere.originAltitudeM;
        if (ImGui::SliderFloat("Altitude [m]", &altitudeM, 1.0f, 99000.0f, "%.0f", ImGuiSliderFlags_Logarithmic)) {
            m_camera.positionM.y = altitudeM - m_atmosphere.originAltitudeM;
        }
        ImGui::SliderFloat("Speed [m/s]", &m_cameraSpeedMps, 1.0f, 10000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
        float fovDeg = m_camera.verticalFovRad * 180.0f / std::numbers::pi_v<float>;
        if (ImGui::SliderFloat("Vertical FOV [deg]", &fovDeg, 10.0f, 120.0f, "%.0f")) {
            m_camera.verticalFovRad = fovDeg * std::numbers::pi_v<float> / 180.0f;
        }
        ImGui::Text("Yaw %.1f deg, pitch %.1f deg", m_camera.yawRad * 180.0f / std::numbers::pi_v<float>,
                    m_camera.pitchRad * 180.0f / std::numbers::pi_v<float>);
    }

    if (ImGui::CollapsingHeader("Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::string names;
        for (const atmosphere::CoefficientSet& set : m_coefficientSets) {
            names += set.name + " (" + display::toString(set.workingSpace) + ")";
            names.push_back('\0');
        }
        int index = m_coefficientIndex;
        if (ImGui::Combo("Coefficient set", &index, names.c_str())) {
            applyCoefficientSet(index);
        }
        const glm::vec3& rs = m_atmosphere.rayleighScatteringPerKm;
        const glm::vec3& oz = m_atmosphere.ozoneAbsorptionPerKm;
        ImGui::Text("Rayleigh sigma_s [1e-3/km]: %.3f %.3f %.3f", rs.x * 1e3f, rs.y * 1e3f, rs.z * 1e3f);
        ImGui::Text("Ozone sigma_a [1e-3/km]   : %.3f %.3f %.3f", oz.x * 1e3f, oz.y * 1e3f, oz.z * 1e3f);
        ImGui::Text("Mie sigma_s / sigma_a [1e-3/km]: %.3f / %.3f", m_atmosphere.mieScatteringPerKm.x * 1e3f,
                    m_atmosphere.mieAbsorptionPerKm.x * 1e3f);
        ImGui::Text("Sun color (Y=1): %.3f %.3f %.3f", m_atmosphere.sunColor.x, m_atmosphere.sunColor.y,
                    m_atmosphere.sunColor.z);
        ImGui::SliderFloat("Mie g", &m_atmosphere.miePhaseG, 0.0f, 0.99f, "%.2f");

        ImGui::SeparatorText("Sun");
        ImGui::SliderFloat("Zenith [deg]", &m_atmosphere.sunZenithDeg, 0.0f, 120.0f, "%.2f");
        ImGui::SliderFloat("Azimuth [deg]", &m_atmosphere.sunAzimuthDeg, -180.0f, 180.0f, "%.1f");
        ImGui::DragFloat("Illuminance [lx]", &m_atmosphere.sunIlluminanceLux, 100.0f, 0.0f, 1.0e6f, "%.0f");
        ImGui::Text("(from ASTM G-173 spectrum: %.0f lx)", m_solarIlluminanceFromSpectrumLux);

        ImGui::SeparatorText("Quality");
        int lutSteps = static_cast<int>(m_transmittanceSteps);
        if (ImGui::SliderInt("Transmittance LUT steps", &lutSteps, 4, 1000, "%d", ImGuiSliderFlags_Logarithmic)) {
            m_transmittanceSteps = static_cast<uint32_t>(lutSteps);
        }
        int primary = static_cast<int>(m_raymarch.primarySteps);
        if (ImGui::SliderInt("View ray steps", &primary, 4, 512, "%d", ImGuiSliderFlags_Logarithmic)) {
            m_raymarch.primarySteps = static_cast<uint32_t>(primary);
        }
        int sunMode = static_cast<int>(m_raymarch.sunMode);
        if (ImGui::Combo("Sun transmittance", &sunMode, "Transmittance LUT\0Nested raymarch (reference)\0")) {
            m_raymarch.sunMode = static_cast<atmosphere::SunTransmittanceMode>(sunMode);
        }
        if (m_raymarch.sunMode == atmosphere::SunTransmittanceMode::Nested) {
            int nested = static_cast<int>(m_raymarch.nestedSteps);
            if (ImGui::SliderInt("Nested steps", &nested, 4, 256, "%d", ImGuiSliderFlags_Logarithmic)) {
                m_raymarch.nestedSteps = static_cast<uint32_t>(nested);
            }
        }

        ImGui::SeparatorText("Transmittance LUT (256x64)");
        // 横軸: 大気上端までの距離（左 = 真上、右 = 地平線）、縦軸: 高度（上 = 地表、下 = 大気上端）
        const gfx::Texture& lut = m_atmosphereRenderer->transmittanceLut();
        ImGui::Image(static_cast<ImTextureID>(m_device.bindless().gpu(lut.srvIndex()).ptr), ImVec2(384.0f, 96.0f));
        ImGui::TextDisabled("x: view zenith (left: up, right: horizon), y: altitude (top: ground)");
    }
}

} // namespace app
