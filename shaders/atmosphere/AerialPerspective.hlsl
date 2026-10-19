// 空気遠近法のボリューム（Hillaire 2020 §5.4、参照実装 RenderCameraVolumePS に合わせる）。
//
// カメラの視錐台に沿った 32×32×32 のボクセル。各ボクセル = 「カメラからその点まで」の
//   gAerialPerspectiveUav    : RGB = Rayleigh の単散乱の内散乱 [sr⁻¹]、A = 透過率の RGB 平均
//   gAerialPerspectiveMieUav : RGB = Mie の単散乱から位相関数を除いた内散乱 [無次元]、
//                              A = 視線上の点で太陽が見える割合の平均（地形の影の検証用。合成では使わない）
//   gAerialPerspectiveMie1Uav: RGB = Mie の成分 1 の同じ量（2 成分のプリセットだけ。design.md §5.9）。A は 1
//   gAerialPerspectiveMsUav  : RGB = 多重散乱の内散乱 [sr⁻¹]（影の対象外）
//   gAerialPerspectiveTransmittanceUav : RGB = 透過率（design.md §5.7 の方式 2。A は 1）
// 地形の色には、合成パス（SkyFromLut）で color × T + 内散乱 として掛ける。
// [POINT] Mie の位相関数は Sky-View LUT と同じ理由で分けて持つ（design.md §12 Q9）。画面方向の解像度は
//         画角 60° で 1 ボクセル ~1.9° しかなく、g が大きい Mie の光輪（g = 0.99 で幅 0.88°）を表せない。
// [POINT] 単散乱と多重散乱を分けて持つのは、合成でピクセルごとに影の区間の単散乱を差し引くため
//         （design.md §5.6 の方式 5）。カメラからの累積なので、区間 [a, b] の単散乱 = F(b) − F(a)。
#include "atmosphere/SkyPassConstants.hlsli"

// ボクセル (x, y)（小数可）の中心の視線方向（画面の UV。合成パスでは同じ UV で引く）
float3 VoxelViewDirection(float2 voxel)
{
    const float2 uv = (voxel + 0.5) / gApSliceCount;
    const float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    const float4 h = mul(gInvViewProjRot, float4(ndc, 1.0, 1.0));
    return normalize(h.xyz / h.w);
}

[numthreads(4, 4, 4)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    const float sliceCount = gApSliceCount;
    const float3 voxelDir = VoxelViewDirection(float2(id.xy));
    float3 viewDir = voxelDir;

    // [POINT] 深度方向は 2 次配置: スライス座標 s = ((z + 0.5)/32)² × 32、距離 = s × 4 km（最大 128 km）。
    //         カメラの近くを細かくする（参照実装と同じ、design.md §12 Q10）。
    float slice = (float(id.z) + 0.5) / sliceCount;
    slice = slice * slice * sliceCount;
    float distanceKm = slice * gApKmPerSlice;

    // ボクセルの点が地表の下なら、地表の少し上に移す（参照実装と同じ。地表付近の大きなボクセルの破綻を防ぐ）
    const float3 pointKm = gCameraPosKm + distanceKm * viewDir;
    if (length(pointKm) <= gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm) {
        const float3 moved = normalize(pointKm) * (gAtmosphere.bottomRadiusKm + kSurfaceOffsetKm + 0.001);
        viewDir = normalize(moved - gCameraPosKm);
        distanceKm = length(moved - gCameraPosKm);
    }

    IntegrationSettings settings;
    // 区間数はスライスが遠いほど多く（参照実装と同じ (z + 1) × 2）、等間隔
    settings.stepCount = max(1u, (id.z + 1) * 2);
    settings.sunTransmittanceMode = SUN_TRANSMITTANCE_LUT;
    settings.nestedSteps = 0;
    settings.multipleScattering = (gFlags & SKY_FLAG_MULTIPLE_SCATTERING) != 0;
    settings.isotropicPhase = false;
    settings.groundRadiance = false;
    settings.uniformSteps = true;
    settings.separateMiePhase = true;
    settings.maxDistanceKm = distanceKm;
    // [POINT] 地形の影を単散乱に掛ける（design.md §5.5、§5.6）。ボクセルはカメラから積分し直すので、
    //         光線の原点はカメラ = カメラ相対の位置で影を引ける。
    if (gApShadowMethod == AP_SHADOW_PER_PIXEL) {
        DisableOpaqueShadow(settings);  // 方式 5: ボリュームは影なし（合成でピクセルごとに差し引く）
    } else {
        SetOpaqueShadowFromConstants(settings);
        if (gApShadowMethod == AP_SHADOW_VOLUME_FILTERED) {
            // [POINT] 方式 1: 影をボクセルの横幅に合わせてぼかす（低域通過フィルター）。ボクセルの列は、カメラから
            //         距離 t で横幅 2·t·tan(半角) に広がる。半角 = 隣のボクセルの中心方向との角度の半分（x, y の平均）。
            //         格子より細かい影は、ボクセルの中の平均として入る（ブロックの縁の段差がなくなる）。
            const float angleX = acos(clamp(dot(voxelDir, VoxelViewDirection(float2(id.xy) + float2(1.0, 0.0))), -1.0, 1.0));
            const float angleY = acos(clamp(dot(voxelDir, VoxelViewDirection(float2(id.xy) + float2(0.0, 1.0))), -1.0, 1.0));
            settings.opaqueShadowFilterRadiusPerM = tan(0.25 * (angleX + angleY)) * gApShadowFilterScale;
        }
    }
    const ScatteringResult result = IntegrateScatteredLuminance(gAtmosphere, LoadAtmosphereLuts(), settings,
                                                                gCameraPosKm, viewDir, gSunDirection);

    RWTexture3D<float4> output = ResourceDescriptorHeap[gAerialPerspectiveUav];
    RWTexture3D<float4> mieOutput = ResourceDescriptorHeap[gAerialPerspectiveMieUav];
    RWTexture3D<float4> msOutput = ResourceDescriptorHeap[gAerialPerspectiveMsUav];
    // separateMiePhase なので luminance = Rayleigh の単散乱 + 多重散乱
    output[id] = float4(result.luminance - result.multipleScattering, dot(result.transmittance, 1.0 / 3.0));
    mieOutput[id] = float4(result.mieSingleNoPhase, result.opaqueShadowMean);
    msOutput[id] = float4(result.multipleScattering, 1.0);
#if MIE_GENERAL
    if (gAtmosphere.mieComponentCount > 1) {
        RWTexture3D<float4> mie1Output = ResourceDescriptorHeap[gAerialPerspectiveMie1Uav];
        mie1Output[id] = float4(result.mie1SingleNoPhase, 1.0);
    }
#endif
    RWTexture3D<float4> transmittanceOutput = ResourceDescriptorHeap[gAerialPerspectiveTransmittanceUav];
    transmittanceOutput[id] = float4(result.transmittance, 1.0);
}
