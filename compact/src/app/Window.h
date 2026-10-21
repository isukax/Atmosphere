#pragma once

#include <windows.h>

#include <cstdint>

namespace compact::app {

/// Win32 ウィンドウ。メッセージから「サイズ変更」「移動」「ディスプレイ構成変更」を
/// フラグとして記録し、フレームループ側で消費する。
class Window {
public:
    Window(const wchar_t* title, uint32_t width, uint32_t height, bool hidden=false);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    /// メッセージを処理する。ウィンドウが閉じられたら false。
    bool pumpMessages();

    HWND hwnd() const { return m_hwnd; }
    uint32_t width() const { return m_width; }
    uint32_t height() const { return m_height; }
    bool minimized() const { return m_minimized; }

    /// 前回呼び出し以降にサイズが変わったか（変わっていればフラグを下ろして true）。
    bool consumeResized();
    /// 前回呼び出し以降に、ディスプレイの再評価が必要なイベント（移動・DPI・構成変更）があったか。
    bool consumeDisplayEvent();

private:
    static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    LRESULT handle(UINT msg, WPARAM wParam, LPARAM lParam);

    HWND m_hwnd = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_minimized = false;
    bool m_resized = false;
    bool m_displayEvent = true;  // 起動直後に 1 回評価する
    bool m_closed = false;
};

} // namespace app
