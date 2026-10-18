#pragma once

#include "atmosphere/AtmosphereParams.h"
#include "gfx/Buffer.h"

#include <vector>

namespace atmosphere {

/// エアロゾルのプリセットの Mie の位相関数の表（design.md §5.9）を GPU に上げて持つ。
/// [POINT] 表はプリセットごとに 1 本の StructuredBuffer<float4>（2,049 点、RGB）。起動時に全プリセット分を一度だけ
///         上げ、作り直さない（描画中のフレームが参照しているバッファを差し替えないため）。
class MiePhaseTables {
public:
    /// sets のすべてのプリセットの表を上げ、各成分の phaseTableSrv を設定する。完了まで待つ。
    /// sets はこのオブジェクトより長く使ってよいが、SRV はこのオブジェクトが生きている間だけ有効。
    MiePhaseTables(gfx::Device& device, std::vector<CoefficientSet>& sets);

private:
    std::vector<gfx::Buffer> m_buffers;
};

} // namespace atmosphere
