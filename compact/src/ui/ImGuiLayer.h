#pragma once

#include "gfx/Device.h"

namespace compact::ui {

/// ImGui の初期化と描画。UI は専用の RT（R8G8B8A8_UNORM、sRGB 値）に描き、
/// 表示パスでシーンと合成する（HDR10 では UI の白の輝度を指定して合成するため）。
class ImGuiLayer {
public:
    static constexpr DXGI_FORMAT kUiFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

    ImGuiLayer(gfx::Device& device, HWND window);
    ~ImGuiLayer();
    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    void newFrame();
    /// 描画データを `rtv` に記録する（RT は RENDER_TARGET 状態、事前にクリア済みであること）。
    void render(ID3D12GraphicsCommandList* cmd, D3D12_CPU_DESCRIPTOR_HANDLE rtv);

private:
    gfx::Device& m_device;
};

} // namespace ui
