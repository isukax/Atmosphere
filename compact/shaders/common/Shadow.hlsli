// 太陽のシャドウマップを引く（design.md §5.5）。
//
// シャドウマップは正射影の D32。深度は光源（太陽）に近いほど大きい（Reverse、クリア値 0 = 遮蔽物なし）。
// 受け手の位置は C++ 側で作った行列で (u, v, 深度) に変換してから渡す（render/ShadowRenderer）。
// [POINT] mapSrv はスレッドごとに違ってよい（空気遠近法は点ごとに近景 / 全体のカスケードを選ぶ）。
//         ウェーブ内で値が揃わないディスクリプタのインデックスは NonUniformResourceIndex で包む。包まないと
//         GPU が 1 つのスレッドのインデックスでウェーブ全体を読み（未定義動作）、別のカスケードを引いた
//         （design.md §14 Phase 4.5 の方式の比較）。
#ifndef SHADOW_HLSLI
#define SHADOW_HLSLI

#include "common/Bindless.hlsli"

// 比較 1 回（線形補間で 2×2 の PCF）。戻り値は日なたの割合 [0, 1]。
// [POINT] シャドウマップの外は日なた。受け手の深度は 0 以上に丸める: シャドウマップの奥行きの範囲より
//         光源から遠い点（範囲の外の空気など）でも、遮蔽物のない所（深度 0）では日なたになるように。
float SampleShadowMap(uint mapSrv, float3 uvz)
{
    if (any(uvz.xy < 0.0) || any(uvz.xy > 1.0)) {
        return 1.0;
    }
    Texture2D<float> map = ResourceDescriptorHeap[NonUniformResourceIndex(mapSrv)];
    return map.SampleCmpLevelZero(gShadowCompare, uvz.xy, saturate(uvz.z));
}

// 3×3 回の比較（1 テクセルずつずらす。各回が 2×2 の PCF なので 4×4 テクセルの範囲をなめらかに混ぜる）
float SampleShadowMapPcf3x3(uint mapSrv, float3 uvz)
{
    if (any(uvz.xy < 0.0) || any(uvz.xy > 1.0)) {
        return 1.0;
    }
    Texture2D<float> map = ResourceDescriptorHeap[NonUniformResourceIndex(mapSrv)];
    float width;
    float height;
    map.GetDimensions(width, height);
    const float2 texel = 1.0 / float2(width, height);
    const float depth = saturate(uvz.z);
    float sum = 0.0;
    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            sum += map.SampleCmpLevelZero(gShadowCompare, uvz.xy + float2(x, y) * texel, depth);
        }
    }
    return sum / 9.0;
}

#endif // SHADOW_HLSLI
