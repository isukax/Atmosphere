#include "app/Window.h"

#include <imgui.h>
#include <imgui_impl_win32.h>

#include <stdexcept>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

namespace compact::app {

Window::Window(const wchar_t* title, uint32_t width, uint32_t height, bool hidden)
{
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = &Window::wndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"AtmosphereWindow";
    RegisterClassExW(&wc);

    RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    m_hwnd = CreateWindowExW(0, wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                             rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, wc.hInstance, this);
    if (!m_hwnd) {
        throw std::runtime_error("CreateWindowExW failed");
    }
    RECT client{};
    GetClientRect(m_hwnd, &client);
    m_width = static_cast<uint32_t>(client.right - client.left);
    m_height = static_cast<uint32_t>(client.bottom - client.top);
    if (!hidden) ShowWindow(m_hwnd, SW_SHOW);
}

Window::~Window()
{
    if (m_hwnd) {
        DestroyWindow(m_hwnd);
    }
}

bool Window::pumpMessages()
{
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return !m_closed;
}

bool Window::consumeResized()
{
    const bool resized = m_resized;
    m_resized = false;
    return resized;
}

bool Window::consumeDisplayEvent()
{
    const bool event = m_displayEvent;
    m_displayEvent = false;
    return event;
}

LRESULT CALLBACK Window::wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_NCCREATE) {
        // CreateWindowExW が戻る前に届くメッセージでも DefWindowProcW に正しい HWND を渡せるよう、
        // ここで HWND を保存する（m_hwnd が null のままだと WM_NCCREATE が失敗し、作成自体が失敗する）。
        auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
        auto* self = static_cast<Window*>(create->lpCreateParams);
        self->m_hwnd = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    }
    if (ImGui::GetCurrentContext() && ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam)) {
        return 1;
    }
    auto* self = reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    return self ? self->handle(msg, wParam, lParam) : DefWindowProcW(hwnd, msg, wParam, lParam);
}

LRESULT Window::handle(UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_SIZE:
        m_minimized = wParam == SIZE_MINIMIZED;
        if (!m_minimized) {
            const uint32_t w = LOWORD(lParam);
            const uint32_t h = HIWORD(lParam);
            if (w != m_width || h != m_height) {
                m_width = w;
                m_height = h;
                m_resized = true;
            }
        }
        m_displayEvent = true;
        return 0;
    // [POINT] ディスプレイの再評価のきっかけ: ウィンドウの移動（別モニターへ移った可能性）、
    //         DPI 変更（モニター間移動で起きる）、ディスプレイ構成の変更（解像度・接続・HDR 切替）。
    case WM_MOVE:
    case WM_DPICHANGED:
    case WM_DISPLAYCHANGE:
        m_displayEvent = true;
        break;
    case WM_CLOSE:
        m_closed = true;
        return 0;
    default:
        break;
    }
    return DefWindowProcW(m_hwnd, msg, wParam, lParam);
}

} // namespace app
