#pragma once

#include "gfx/Device.h"

#include <string>
#include <vector>

namespace gfx {

/// GPU のタイムスタンプでパスごとの時間を測る（V8）。
///
/// [POINT] タイムスタンプは GPU がコマンドを実行した時刻なので、CPU 側の計測と違い
///         パスの実際の GPU 時間が分かる。結果は、そのフレームのコマンドの完了を待ってから読む
///         （フレームスロットごとにクエリ範囲を分け、次にそのスロットを使うときに読む）。
class GpuTimer {
public:
    static constexpr uint32_t kMaxScopes = 48;

    GpuTimer(Device& device, uint32_t slotCount);

    /// スロットの計測を始める（前回の結果は read() で先に読んでおくこと）。
    void beginFrame(uint32_t slot);
    /// 区間の開始。戻り値を end() に渡す。上限を超えたら計測しない（kMaxScopes を返す）。
    uint32_t begin(ID3D12GraphicsCommandList* cmd, const char* name);
    void end(ID3D12GraphicsCommandList* cmd, uint32_t scope);
    /// クエリの結果を読み戻しバッファへ解決する（コマンドリストの最後で呼ぶ）。
    void resolve(ID3D12GraphicsCommandList* cmd);

    struct Result {
        std::string name;
        double milliseconds;
    };
    /// GPU がスロットのコマンドを完了した後で呼ぶ。
    std::vector<Result> read(uint32_t slot) const;

private:
    ComPtr<ID3D12QueryHeap> m_heap;
    ComPtr<ID3D12Resource> m_readback;
    double m_ticksPerMs = 1.0;
    uint32_t m_slot = 0;
    std::vector<std::vector<std::string>> m_names;  // スロットごとの区間名
};

} // namespace gfx
