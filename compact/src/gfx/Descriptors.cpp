#include "gfx/Descriptors.h"

namespace compact::gfx {

void BindlessHeap::init(ID3D12Device* device, uint32_t capacity)
{
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = capacity;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_heap)), "CreateDescriptorHeap(bindless)");
    m_heap->SetName(L"BindlessHeap");
    m_stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_capacity = capacity;
    m_free.clear();
    m_free.reserve(capacity);
    for (uint32_t i = capacity; i > 0; --i) {
        m_free.push_back(i - 1);  // 小さいインデックスから使う
    }
}

uint32_t BindlessHeap::allocate()
{
    if (m_free.empty()) {
        throw std::runtime_error("BindlessHeap exhausted");
    }
    const uint32_t index = m_free.back();
    m_free.pop_back();
    return index;
}

void BindlessHeap::free(uint32_t index)
{
    if (index != kInvalidDescriptorIndex) {
        m_free.push_back(index);
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE BindlessHeap::cpu(uint32_t index) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * m_stride;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE BindlessHeap::gpu(uint32_t index) const
{
    D3D12_GPU_DESCRIPTOR_HANDLE handle = m_heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(index) * m_stride;
    return handle;
}

uint32_t BindlessHeap::indexOf(D3D12_CPU_DESCRIPTOR_HANDLE handle) const
{
    return static_cast<uint32_t>((handle.ptr - m_heap->GetCPUDescriptorHandleForHeapStart().ptr) / m_stride);
}

void CpuDescriptorPool::init(ID3D12Device* device, uint32_t capacity, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = type;
    desc.NumDescriptors = capacity;
    check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_heap)), "CreateDescriptorHeap(cpu)");
    m_stride = device->GetDescriptorHandleIncrementSize(type);
    m_free.clear();
    for (uint32_t i = capacity; i > 0; --i) {
        m_free.push_back(i - 1);
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE CpuDescriptorPool::allocate()
{
    if (m_free.empty()) {
        throw std::runtime_error("CpuDescriptorPool exhausted");
    }
    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(m_free.back()) * m_stride;
    m_free.pop_back();
    return handle;
}

void CpuDescriptorPool::free(D3D12_CPU_DESCRIPTOR_HANDLE handle)
{
    if (handle.ptr == 0) {
        return;
    }
    m_free.push_back(static_cast<uint32_t>((handle.ptr - m_heap->GetCPUDescriptorHandleForHeapStart().ptr) / m_stride));
}

} // namespace gfx
