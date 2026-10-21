// 地形・テスト球の描画（ランバート）。design.md §6。
//
// 単位:
//   位置 [m]（ワールド、+Y 上）。カメラ相対で変換する（遠景でも float の精度を保つため）
//   輝度 L [cd/m²] = ρ/π × (太陽の照度 + 空からの照度) → 露出して fb でシーンカラーに書く
//   空気遠近法は後の合成パス（SkyFromLut / SkyRaymarch）で掛ける
//   太陽の直達光には、シャドウマップ（2 段のカスケード）の影を掛ける（design.md §5.5）
// シャドウマップを描くときも同じ頂点シェーダーを使う（gViewProjRot = 原点相対 → クリップ、gCameraPosM = 原点）。
#include "common/Bindless.hlsli"
#include "common/Shadow.hlsli"
#include "common/SphericalHarmonics.hlsli"
#include "atmosphere/AtmosphereCommon.hlsli"

cbuffer MeshConstants : register(b0)
{
    row_major float4x4 gViewProjRot     : packoffset(c0);   // 回転のみのビュー × 射影（カメラ相対の座標を変換する）
    float3 gCameraPosM                  : packoffset(c4);
    uint   gVertexBufferSrv             : packoffset(c4.w);
    float3 gObjectOffsetM               : packoffset(c5);   // 物体の中心 [m]（地形は 0）
    float  gObjectScale                 : packoffset(c5.w); // 半径 [m]（地形は 1）
    float3 gAlbedo                      : packoffset(c6);
    uint   gSkyIrradianceSrv            : packoffset(c6.w); // 空からの照度の SH 係数（単位照度あたり）
    float3 gSunDirection                : packoffset(c7);
    uint   gTransmittanceLutSrv         : packoffset(c7.w);
    float3 gSunIlluminanceLux           : packoffset(c8);   // E_sun × 太陽の色 [lx]
    float  gExposureScale               : packoffset(c8.w); // 1/(1.2·2^EV100) [1/(cd/m²)]
    float3 gPlanetOffsetKm              : packoffset(c9);   // ワールド原点の惑星座標 (0, R + h0, 0) [km]
    float  gPad9                        : packoffset(c9.w);
    float2 gTransmittanceLutSize        : packoffset(c10.x);
    uint   gUseSkyIrradiance            : packoffset(c10.z);
    AtmosphereParameters gAtmosphere    : packoffset(c11);  // c11-c16
    row_major float4x4 gShadowNearToUvz : packoffset(c17);  // カメラ相対 [m] → 近景のカスケードの (u, v, 深度)
    row_major float4x4 gShadowFarToUvz  : packoffset(c21);  // カメラ相対 [m] → 全体のカスケードの (u, v, 深度)
    uint   gShadowNearSrv               : packoffset(c25.x); // 無効値なら近景なし
    uint   gShadowFarSrv                : packoffset(c25.y); // 無効値なら影なし
    float  gShadowBlendBand             : packoffset(c25.z); // 近景の端で全体と混ぜる帯の幅（近景の半幅に対する割合）
    uint   gOutputMode                  : packoffset(c25.w); // MESH_OUTPUT_*
    float  gShadowNearNormalOffsetM     : packoffset(c26.x); // 法線オフセット [m]（近景）
    float  gShadowFarNormalOffsetM      : packoffset(c26.y); // 法線オフセット [m]（全体）
};


struct Vertex
{
    float3 position;
    float3 normal;
};

struct VSOut
{
    float4 pos : SV_Position;
    float3 relativePosM : POSITION0;  // カメラ相対の位置 [m]
    float3 normal : NORMAL0;
};

static const float kMaxSceneColorFb = 65000.0;

VSOut VSMain(uint vertexId : SV_VertexID)
{
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[gVertexBufferSrv];
    const Vertex v = vertices[vertexId];
    // [POINT] カメラ相対の位置（物体の位置 − カメラの位置）を変換する。ワールド座標のまま変換すると、
    //         数十 km 先で float の精度（数 mm〜cm）が足りなくなり頂点が揺れる。
    const float3 relative = gObjectOffsetM + v.position * gObjectScale - gCameraPosM;
    VSOut o;
    o.pos = mul(gViewProjRot, float4(relative, 1.0));
    o.relativePosM = relative;
    o.normal = v.normal;  // 一様な拡大と平行移動だけなので、法線はそのまま
    return o;
}

// 太陽の見える割合 [0, 1]（地形・球の影）。nearWeight: 近景のカスケードを使った割合（検証用）
float SunVisibility(float3 relativePosM, float3 n, out float nearWeight)
{
    nearWeight = 0.0;
    if (gShadowFarSrv == kInvalidDescriptorIndex) {
        return 1.0;
    }
    // [POINT] 法線オフセット: 面の点を法線方向に 1 テクセル程度ずらしてから引く。シャドウマップの 1 テクセルの中で
    //         面が傾いていると、自分自身の深度で影になる（シャドウアクネ）。深度のバイアス（シャドウマップ側の
    //         傾斜スケールのバイアス）と併用する。太陽が低いほど面と光線のなす角が浅く、アクネが出やすい。
    const float3 uvzFar = mul(gShadowFarToUvz, float4(relativePosM + n * gShadowFarNormalOffsetM, 1.0)).xyz;
    if (gShadowNearSrv != kInvalidDescriptorIndex) {
        const float3 uvzNear = mul(gShadowNearToUvz, float4(relativePosM + n * gShadowNearNormalOffsetM, 1.0)).xyz;
        // 近景の中心からの距離（0 = 中心、1 = 端）。端の帯で全体のカスケードと混ぜて、境目の段差を隠す
        const float edge = max(abs(uvzNear.x - 0.5), abs(uvzNear.y - 0.5)) * 2.0;
        nearWeight = saturate((1.0 - edge) / max(gShadowBlendBand, 1e-3));
        if (nearWeight >= 1.0) {
            return SampleShadowMapPcf3x3(gShadowNearSrv, uvzNear);
        }
        if (nearWeight > 0.0) {
            return lerp(SampleShadowMapPcf3x3(gShadowFarSrv, uvzFar), SampleShadowMapPcf3x3(gShadowNearSrv, uvzNear),
                        nearWeight);
        }
    }
    return SampleShadowMapPcf3x3(gShadowFarSrv, uvzFar);
}

float4 PSMain(VSOut i) : SV_Target
{
    const float3 n = normalize(i.normal);
    const float3 worldPosM = i.relativePosM + gCameraPosM;
    // [POINT] シェーディング点の惑星座標 [km]（m → km）。太陽方向の透過率をこの点の高度で引くので、
    //         山の上ほど太陽光が強く白くなる（カメラ位置で 1 回だけ引くと、この差が出ない）。
    const float3 pKm = worldPosM * 0.001 + gPlanetOffsetKm;
    const float r = length(pKm);
    const float muSun = dot(pKm, gSunDirection) / r;
    float3 sunTransmittance = 0.0;
    if (!RayIntersectsGround(gAtmosphere, r, muSun)) {
        Texture2D<float4> lut = ResourceDescriptorHeap[gTransmittanceLutSrv];
        sunTransmittance = SampleTransmittanceLut(gAtmosphere, lut, gLinearClamp, r, muSun, gTransmittanceLutSize);
    }

    // 照度（単位照度あたり）: 太陽の直達光 T·cos·(影) + 空の拡散光 E_sky(n)/E_sun
    const float nDotL = saturate(dot(n, gSunDirection));
    float nearWeight = 0.0;
    const float visibility = nDotL > 0.0 ? SunVisibility(i.relativePosM, n, nearWeight) : 0.0;

    float3 irradiancePerLux = sunTransmittance * (nDotL * visibility);
    if (gUseSkyIrradiance != 0) {
        StructuredBuffer<float4> sh = ResourceDescriptorHeap[gSkyIrradianceSrv];
        irradiancePerLux += EvaluateShIrradiance(sh, n);
    }
    // [POINT] ランバート: 輝度 [cd/m²] = ρ/π × 照度 [lx]。照度 = (単位照度あたり) × E_sun × 太陽の色
    const float3 luminance = gAlbedo / kPi * irradiancePerLux * gSunIlluminanceLux;
    return float4(min(luminance * gExposureScale, kMaxSceneColorFb), 1.0);
}
