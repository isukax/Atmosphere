#pragma once

#include "gfx/Common.h"

#include <vector>

namespace compact::gfx {

inline constexpr uint32_t kInvalidDescriptorIndex = 0xFFFFFFFFu;

/// シェーダー可視の CBV/SRV/UAV ヒープ（Bindless 用）。
///
/// [POINT] シェーダーは SM 6.6 の `ResourceDescriptorHeap[index]` でこのヒープを直接引く。
///         C++ 側はリソースを作るたびにここからインデックスを 1 つ確保し、
///         そのインデックスを定数バッファでシェーダーに渡すだけでよい（ディスクリプタテーブル不要）。
///         ImGui も同じヒープから確保する（ヒープは同時に 1 つしかバインドできないため）。
class BindlessHeap {
public:
    void init(ID3D12Device* device, uint32_t capacity);

    uint32_t allocate();
    void free(uint32_t index);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu(uint32_t index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(uint32_t index) const;
    uint32_t indexOf(D3D12_CPU_DESCRIPTOR_HANDLE handle) const;
    ID3D12DescriptorHeap* heap() const { return m_heap.Get(); }

private:
    ComPtr<ID3D12DescriptorHeap> m_heap;
    uint32_t m_stride = 0;
    uint32_t m_capacity = 0;
    std::vector<uint32_t> m_free;
};

/// シェーダー非可視の RTV / DSV ヒープ。フリーリストで確保・解放する。
class CpuDescriptorPool {
public:
    void init(ID3D12Device* device, uint32_t capacity, D3D12_DESCRIPTOR_HEAP_TYPE type);
    D3D12_CPU_DESCRIPTOR_HANDLE allocate();
    void free(D3D12_CPU_DESCRIPTOR_HANDLE handle);

private:
    ComPtr<ID3D12DescriptorHeap> m_heap;
    uint32_t m_stride = 0;
    std::vector<uint32_t> m_free;
};

} // namespace gfx
