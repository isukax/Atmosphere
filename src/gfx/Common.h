#pragma once

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>

template <class T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

namespace gfx {

/// HRESULT が失敗なら例外にする。`what` は失敗した呼び出しの説明。
inline void check(HRESULT hr, const char* what)
{
    if (FAILED(hr)) {
        throw std::runtime_error(std::format("{} failed (HRESULT 0x{:08X})", what, static_cast<uint32_t>(hr)));
    }
}

/// 標準出力とデバッガ出力の両方にログを書く。
void log(const std::string& message);

std::string narrow(const std::wstring& wide);

} // namespace gfx
