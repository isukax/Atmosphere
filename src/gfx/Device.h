#pragma once

#include "gfx/Common.h"
#include "gfx/Descriptors.h"

#include <array>
#include <cstring>
#include <functional>

namespace gfx {

/// フレームごとの定数バッファ用アップロードリング。フレームスロットごとに線形確保し、
/// そのスロットの GPU 完了を待ったあとで先頭に戻す。
class UploadRing {
public:
    struct Allocation {
        void* cpu = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS gpu = 0;
    };

    void init(ID3D12Device* device, uint64_t bytesPerSlot, uint32_t slotCount);
    void beginSlot(uint32_t slot);
    Allocation allocate(uint64_t bytes);

    template <class T>
    D3D12_GPU_VIRTUAL_ADDRESS push(const T& value)
    {
        const Allocation a = allocate(sizeof(T));
        std::memcpy(a.cpu, &value, sizeof(T));
        return a.gpu;
    }

private:
    ComPtr<ID3D12Resource> m_buffer;
    uint8_t* m_mapped = nullptr;
    uint64_t m_bytesPerSlot = 0;
    uint64_t m_slotBegin = 0;
    uint64_t m_offset = 0;
};

/// D3D12 デバイス・直接キュー・フレームループ用コマンドリストをまとめた薄い層。
/// （Shoga では RHI が担う部分。大気モジュールからは直接依存しない。）
class Device {
public:
    /// [POINT] 同時に処理中にできるフレーム数。CPU はスロット i のフェンスを待ってから
    ///         そのスロットのコマンドアロケータ・アップロード領域を再利用する。
    static constexpr uint32_t kFramesInFlight = 2;

    explicit Device(bool debugLayer);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    ID3D12Device* d3d() const { return m_device.Get(); }
    ID3D12CommandQueue* queue() const { return m_queue.Get(); }
    IDXGIFactory6* factory() const { return m_factory.Get(); }
    const std::string& adapterName() const { return m_adapterName; }

    /// [POINT] DXGI ファクトリは OS のディスプレイ構成（HDR の ON/OFF、接続変更など）が
    ///         変わると古くなり、`IsCurrent()` が false を返す。古いファクトリから列挙した
    ///         出力情報は更新されないので、作り直してから再列挙する。作り直したら true。
    bool refreshFactoryIfStale();

    /// フレームスロットの GPU 完了を待ち、コマンドリストを記録可能な状態にして返す。
    ID3D12GraphicsCommandList* beginFrame();
    /// コマンドリストを閉じて実行し、スロットのフェンス値を記録する（Present は呼び出し側）。
    void endFrame();
    uint32_t frameSlot() const { return m_frameSlot; }
    uint64_t frameNumber() const { return m_frameNumber; }

    /// キューが空になるまで待つ（リソースの作り直し前に使う）。
    void waitIdle();
    /// 単発のコマンドを記録・実行し、完了まで待つ（アップロード・リードバック・バッチ検証用）。
    void executeImmediate(const std::function<void(ID3D12GraphicsCommandList*)>& record);

    BindlessHeap& bindless() { return m_bindless; }
    CpuDescriptorPool& rtvPool() { return m_rtvPool; }
    CpuDescriptorPool& dsvPool() { return m_dsvPool; }
    UploadRing& upload() { return m_upload; }
    ID3D12GraphicsCommandList* cmd() const { return m_cmd.Get(); }

private:
    void createFactory();
    uint64_t signal();
    void waitFor(uint64_t value);

    ComPtr<IDXGIFactory6> m_factory;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;
    std::string m_adapterName;

    std::array<ComPtr<ID3D12CommandAllocator>, kFramesInFlight> m_allocators;
    std::array<uint64_t, kFramesInFlight> m_slotFence{};
    ComPtr<ID3D12GraphicsCommandList> m_cmd;
    ComPtr<ID3D12CommandAllocator> m_immediateAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_immediateCmd;

    ComPtr<ID3D12Fence> m_fence;
    uint64_t m_fenceValue = 0;
    HANDLE m_fenceEvent = nullptr;

    uint32_t m_frameSlot = 0;
    uint64_t m_frameNumber = 0;

    BindlessHeap m_bindless;
    CpuDescriptorPool m_rtvPool;
    CpuDescriptorPool m_dsvPool;
    UploadRing m_upload;
};

} // namespace gfx
