// Sky-View LUT から空を描く（Hillaire 2020 の実用パス）+ 太陽ディスク。
// Mie の単散乱だけは位相関数を LUT の外でピクセルごとに掛ける（本実装の変更点。SkyViewLut.hlsl 参照）。
#include "atmosphere/SkyPassConstants.hlsli"

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight) {
        return;
    }
    const float3 viewDir = OutputViewDirection(id.xy);
    const float r = length(gCameraPosKm);
    const float3 up = gCameraPosKm / r;
    const float viewZenithCos = dot(viewDir, up);

    // [POINT] 視線と太陽の「方位差の余弦」を、カメラの水平面上で求める（参照実装と同じ）。
    //         視線が真上・真下だと方位が定まらないが、その方向では値が方位に依存しないので 1 とする。
    float lightViewCos = 1.0;
    const float3 side = cross(up, viewDir);
    if (dot(side, side) > 1e-12) {
        const float3 sideN = normalize(side);
        const float3 forward = normalize(cross(sideN, up));  // 視線の水平成分の向き
        const float2 lightOnPlane = float2(dot(gSunDirection, forward), dot(gSunDirection, sideN));
        const float len = length(lightOnPlane);
        lightViewCos = len > 1e-6 ? lightOnPlane.x / len : 1.0;
    }

    float t0;
    float t1;
    const bool intersectGround = RaySphere(gCameraPosKm, viewDir, gAtmosphere.bottomRadiusKm, t0, t1) && t0 > 0.0;
    const float lutR = max(r, gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm);
    const float2 uv = SkyViewViewParamsToUv(gAtmosphere, intersectGround, viewZenithCos, lightViewCos, lutR,
                                            gSkyViewLutSize);
    Texture2D<float4> skyView = ResourceDescriptorHeap[gSkyViewLutSrv];
    Texture2D<float4> skyViewMie = ResourceDescriptorHeap[gSkyViewMieLutSrv];
    // [POINT] Mie の単散乱は、LUT の「位相関数を除いた積分」[無次元] に、このピクセルの正確な散乱角での
    //         位相関数 p_M(θ) [sr⁻¹] を掛けて足す。LUT のテクセルより細い光輪（g → 1）も形が崩れない
    //         （Sky-View LUT に位相関数を焼き込むと、g = 0.99 でピークが縦に伸び 6〜97% に暗くなった。design.md §14）。
    const float phaseMie = CornetteShanksPhase(gAtmosphere.miePhaseG, dot(viewDir, gSunDirection));
    float3 luminance = skyView.SampleLevel(gLinearClamp, uv, 0.0).rgb +
                       phaseMie * skyViewMie.SampleLevel(gLinearClamp, uv, 0.0).rgb;

    if ((gFlags & SKY_FLAG_SUN_DISK) != 0) {
        luminance += SunDiskPerLux(gCameraPosKm, viewDir);
    }
    WriteSkyOutput(id.xy, luminance, viewDir);
}
