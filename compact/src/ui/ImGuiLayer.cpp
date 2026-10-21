#include "ui/ImGuiLayer.h"

#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_win32.h>

namespace compact::ui {

ImGuiLayer::ImGuiLayer(gfx::Device& device, HWND window)
    : m_device(device)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    // Keep the sample layout in memory; never share the verification app's imgui.ini.
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(window);

    // [POINT] ImGui のテクスチャ用ディスクリプタも Bindless ヒープから確保する。
    //         D3D12 では同時にバインドできるシェーダー可視ヒープは 1 つだけなので、
    //         ImGui 専用のヒープを別に作ると、アプリのパスと交互に切り替える必要が出る。
    ImGui_ImplDX12_InitInfo info{};
    info.Device = device.d3d();
    info.CommandQueue = device.queue();
    info.NumFramesInFlight = gfx::Device::kFramesInFlight;
    info.RTVFormat = kUiFormat;
    info.DSVFormat = DXGI_FORMAT_UNKNOWN;
    info.UserData = &device;
    info.SrvDescriptorHeap = device.bindless().heap();
    info.SrvDescriptorAllocFn = [](ImGui_ImplDX12_InitInfo* initInfo, D3D12_CPU_DESCRIPTOR_HANDLE* cpu,
                                   D3D12_GPU_DESCRIPTOR_HANDLE* gpu) {
        auto* dev = static_cast<gfx::Device*>(initInfo->UserData);
        const uint32_t index = dev->bindless().allocate();
        *cpu = dev->bindless().cpu(index);
        *gpu = dev->bindless().gpu(index);
    };
    info.SrvDescriptorFreeFn = [](ImGui_ImplDX12_InitInfo* initInfo, D3D12_CPU_DESCRIPTOR_HANDLE cpu,
                                  D3D12_GPU_DESCRIPTOR_HANDLE) {
        auto* dev = static_cast<gfx::Device*>(initInfo->UserData);
        dev->bindless().free(dev->bindless().indexOf(cpu));
    };
    ImGui_ImplDX12_Init(&info);
}

ImGuiLayer::~ImGuiLayer()
{
    m_device.waitIdle();
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::newFrame()
{
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(ID3D12GraphicsCommandList* cmd, D3D12_CPU_DESCRIPTOR_HANDLE rtv)
{
    ImGui::Render();
    cmd->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    ID3D12DescriptorHeap* heaps[] = {m_device.bindless().heap()};
    cmd->SetDescriptorHeaps(1, heaps);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmd);
}

} // namespace ui
