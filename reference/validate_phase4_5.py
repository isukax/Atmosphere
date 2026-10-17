"""Phase 4.5 の検証（docs/design.md §8.2 V3e / V3f）。

    reference/.venv/Scripts/python.exe reference/validate_phase4_5.py

  V3e: 地表の影。GPU の太陽の見える割合（シャドウマップ、3×3 の PCF、0.5 で二値化）vs Python のレイキャスト
       （地形 = メッシュと同じ三角形、球 = 解析的）。N·L > 0.05 の地形・球のピクセル。
         一致率（全体）、影の境界（受け手を光源空間で ±1 テクセルずらすと Python の判定が変わる点）を除いた一致率、
         不一致の内訳（GPU だけ影 = アクネ / GPU だけ日なた = 光漏れ）と、カメラからの距離ごとの分布を記録する。
       記録のみ（§12 Q15）。
  V3f: 空気遠近法の影（地形・球の面の影はすべて有り）。
         ① 影ありのボリューム vs 影ありの深度までのレイマーチ（透過率は RGB 平均）の相対 RMSE（V3c と同じ扱い）
         ② 影で減った量: GPU（レイマーチの影なし − 影あり）vs Python（影に入る単散乱、64 点）の合計の比と相対 RMSE
         ③ 空気遠近法のボクセルごとの、視線上の点で太陽が見える割合: GPU（Mie ボリュームの A）vs Python
            （ボクセルの光線は地形より奥では山の中を通るので、地面を中身の詰まった地盤として扱えているかを見る）
       記録のみ。確認画像 docs/images/phase4_5_ap_shadow.png。
  V3g: 空気遠近法の影の方式（design.md §5.6 の 0: ボリューム（点）/ 1: ボリューム（ぼかし）/ 5: ピクセルごと）vs
       深度までのレイマーチ（影あり、透過率は RGB 平均でボリュームと揃える）。地形・球のピクセルで、全体と、
       空気の影が効く画素（影で 2% 以上暗くなる画素）の相対 RMSE、最大誤差を記録する。
       確認画像 docs/images/phase4_5_ap_methods.png（行 = ケース、列 = 方式 0 / 1 / 5 / レイマーチ）と、
       docs/images/phase4_5_ap_methods_error.png（地表 1 m のケースの拡大と、基準との相対誤差）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atmosphere_ref import Atmosphere, sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from pngio import tonemap_srgb, write_png  # noqa: E402
from terrain_ref import (HeightGrid, aerial_single_scattering, camera_basis, in_shadow, reconstruct,  # noqa: E402
                         surface_normals)
from validate_phase1 import DEFAULT_EXE, ROOT, run_app  # noqa: E402

SURFACE_CASES = ("high_sz45", "low_sz80", "sunset_sz88", "near_sz60")
AP_CASES = ("day_front", "low_front", "sunset_front")
DISTANCE_BINS_M = [(0.0, 100.0), (100.0, 1000.0), (1000.0, 5000.0), (5000.0, 20000.0), (20000.0, 1e12)]


def rgb(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float64)


def rel_rmse(test: np.ndarray, ref: np.ndarray) -> list[float]:
    return (np.sqrt(np.mean((test - ref) ** 2, axis=0)) / np.maximum(np.mean(np.abs(ref), axis=0), 1e-30)).tolist()


def scene_meta(meta: dict):
    cam = np.asarray(meta["camera_position_m"], dtype=np.float64)
    return cam, meta["camera_yaw_deg"], meta["camera_pitch_deg"], meta["vertical_fov_deg"], meta["spheres"]


# ---------------------------------------------------------------------- V3e
def check_surface(out_dir: Path, stride: int) -> list[dict]:
    grid = HeightGrid()
    results = []
    for case in SURFACE_CASES:
        vis_img, meta = read_exr(out_dir / f"{case}_visibility.exr")
        depth_img, depth_meta = read_exr(out_dir / f"{case}_depth.exr")
        cam, yaw, pitch, fov, spheres = scene_meta(meta)
        a = meta["atmosphere"]
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        valid = (depth_img[..., 0] > 0.0) & (vis_img[..., 1] > 0.05)  # 地形・球のピクセルで、太陽の側を向く面
        ys, xs = np.nonzero(valid[::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        _, distance_m, hit = reconstruct(depth_img, depth_meta["near_m"], cam, yaw, pitch, fov, ys, xs)
        normal = surface_normals(hit, grid, spheres)
        # [POINT] レイの始点は面から少し離す（法線方向 5 cm + 太陽方向 0.5 m）。深度から戻した点の誤差で
        #         自分の面に当たらないように
        start = hit + normal * 0.05 + sun * 0.5
        python_shadow = in_shadow(start, sun, grid, spheres)
        gpu_shadow = vis_img[ys, xs, 0] < 0.5
        near_weight = vis_img[ys, xs, 2]
        # 影の境界: 受け手を光源空間の x, y に ±1 テクセル動かして、Python の判定が変わる点
        near_c, far_c = meta["shadow_near"], meta["shadow_far"]
        boundary = np.zeros(len(start), dtype=bool)
        for cascade, use in ((near_c, near_weight > 0.5), (far_c, near_weight <= 0.5)):
            if not np.any(use):
                continue
            texel = np.asarray(cascade["texel_size_m"], dtype=np.float64)
            axes = (np.asarray(cascade["axis_x"]) * texel[0], np.asarray(cascade["axis_y"]) * texel[1])
            idx = np.nonzero(use)[0]
            for axis in axes:
                for sign in (-1.0, 1.0):
                    moved = in_shadow(start[idx] + sign * axis, sun, grid, spheres)
                    boundary[idx] |= moved != python_shadow[idx]
        agree = gpu_shadow == python_shadow
        interior = ~boundary
        bins = []
        for lo, hi in DISTANCE_BINS_M:
            sel = (distance_m >= lo) & (distance_m < hi)
            if np.any(sel):
                bins.append({"distance_m": [lo, hi], "pixels": int(sel.sum()),
                             "agreement": float(agree[sel].mean()),
                             "agreement_excluding_boundary": float(agree[sel & interior].mean()) if np.any(sel & interior) else None,
                             "gpu_only_shadow": int((gpu_shadow & ~python_shadow & sel & interior).sum()),
                             "gpu_only_lit": int((~gpu_shadow & python_shadow & sel & interior).sum())})
        results.append({
            "check": f"V3e surface shadow ({case})",
            "sun": {"zenith_deg": a["sun_zenith_deg"], "azimuth_deg": a["sun_azimuth_deg"]},
            "pixels": int(len(start)),
            "python_shadow_fraction": float(python_shadow.mean()),
            "gpu_shadow_fraction": float(gpu_shadow.mean()),
            "near_cascade_pixels": int((near_weight > 0.5).sum()),
            "boundary_pixels": int(boundary.sum()),
            "agreement": float(agree.mean()),
            "agreement_excluding_boundary": float(agree[interior].mean()),
            "gpu_only_shadow_excluding_boundary": int((gpu_shadow & ~python_shadow & interior).sum()),
            "gpu_only_lit_excluding_boundary": int((~gpu_shadow & python_shadow & interior).sum()),
            "by_distance": bins,
            "texel_size_m": {"near": near_c["texel_size_m"], "far": far_c["texel_size_m"]},
            "tolerance": None,
            "pass": True,
            "info": "recorded only (design.md V3e, Q15)",
        })
    return results


# ---------------------------------------------------------------------- V3f
def check_aerial_perspective(out_dir: Path, stride: int, image_path: Path) -> list[dict]:
    grid = HeightGrid()
    results = []
    rows = []
    for case in AP_CASES:
        img = {p: read_exr(out_dir / f"{case}_{p}.exr") for p in
               ("geometry", "volume", "volume_noshadow", "raymarch_rgb", "raymarch_mean", "raymarch_rgb_noshadow")}
        geometry, meta = img["geometry"]
        depth_img, depth_meta = read_exr(out_dir / f"{case}_depth.exr")
        mask = geometry[..., 3] > 0.5
        # ① ボリューム vs レイマーチ（両方とも影あり、透過率は RGB 平均）
        v, rm = rgb(img["volume"][0])[mask], rgb(img["raymarch_mean"][0])[mask]
        # ② 影で減った量
        cam, yaw, pitch, fov, spheres = scene_meta(meta)
        a = meta["atmosphere"]
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        sun_color = np.asarray(a.get("sun_color", [1.0, 1.0, 1.0]), dtype=np.float64)
        atmo = Atmosphere.from_meta(meta)
        ys, xs = np.nonzero(mask[::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        direction, distance_m, _ = reconstruct(depth_img, depth_meta["near_m"], cam, yaw, pitch, fov, ys, xs)
        total, shadowed, _ = aerial_single_scattering(atmo, cam, direction, distance_m, sun, grid, spheres)
        gpu_removed = rgb(img["raymarch_rgb_noshadow"][0])[ys, xs] - rgb(img["raymarch_rgb"][0])[ys, xs]
        python_removed = shadowed * sun_color
        pixel = rgb(img["raymarch_rgb_noshadow"][0])[ys, xs]
        significant = python_removed.sum(1) > 0.01 * pixel.sum(1)
        results.append({
            "check": f"V3f aerial perspective shadow ({case})",
            "sun": {"zenith_deg": a["sun_zenith_deg"], "azimuth_deg": a["sun_azimuth_deg"]},
            "geometry_pixels": int(mask.sum()),
            "volume_vs_raymarch_rel_rmse_rgb": rel_rmse(v, rm),
            "volume_vs_raymarch_max_rel_error": float(np.max(np.abs(v - rm) / np.maximum(rm, 1e-30))),
            "removed_share_of_pixel_rgb": (gpu_removed.sum(0) / pixel.sum(0)).tolist(),
            "removed_gpu_over_python_rgb": (gpu_removed.sum(0) / np.maximum(python_removed.sum(0), 1e-30)).tolist(),
            "removed_rel_rmse_rgb_where_significant": rel_rmse(gpu_removed[significant], python_removed[significant])
            if np.any(significant) else None,
            "significant_pixels": int(significant.sum()),
            "python_single_scattering_shadowed_fraction_rgb": (shadowed.sum(0) / np.maximum(total.sum(0), 1e-30)).tolist(),
            "tolerance": None,
            "pass": bool(np.all(np.isfinite(v))),
            "info": "recorded only (design.md V3f)",
        })
        # 確認画像の行: 地形のみ / ボリューム（影なし）/ ボリューム（影あり）/ レイマーチ（影あり）
        scale = 0.18 / max(np.median(rgb(img["volume_noshadow"][0]).mean(-1)), 1e-12)
        tiles = []
        for p in ("geometry", "volume_noshadow", "volume", "raymarch_rgb"):
            tiles += [tonemap_srgb(rgb(img[p][0]) * scale)[::2, ::2], np.ones((geometry.shape[0] // 2, 4, 3))]
        rows += [np.concatenate(tiles[:-1], axis=1), np.ones((4, 4 * (geometry.shape[1] // 2) + 12, 3))]
    write_png(image_path, np.concatenate(rows[:-1]))
    return results


def check_voxel_visibility(out_dir: Path, max_slice: int = 16) -> list[dict]:
    """③ 空気遠近法のボクセルごとに、視線上の点（区間の中点）で太陽が見える割合: GPU（Mie ボリュームの A）vs Python。
    ボクセルの光線・区間は AerialPerspective.hlsl と同じ（地表の下のボクセルは地表の 2 m 上に移す）。"""

    n = 32
    grid = HeightGrid()
    results = []
    for case in AP_CASES:
        vol, _ = read_exr(out_dir / f"{case}_ap_mie_volume.exr")
        gpu = vol[..., 3].reshape(n, n, n)  # [z, y, x]
        _, meta = read_exr(out_dir / f"{case}_geometry.exr")
        cam, yaw, pitch, fov, spheres = scene_meta(meta)
        a = meta["atmosphere"]
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        forward, right, up = camera_basis(yaw, pitch)
        tan_half = np.tan(0.5 * np.radians(fov))
        aspect = 640.0 / 360.0  # Phase45Tests の解像度
        radius_m = a["bottom_radius_km"] * 1000.0
        cam_planet = np.array([0.0, radius_m + cam[1], 0.0])
        zz, yy, xx = np.meshgrid(np.arange(max_slice), np.arange(n), np.arange(n), indexing="ij")
        zz, yy, xx = zz.ravel(), yy.ravel(), xx.ravel()
        u, v = (xx + 0.5) / n, (yy + 0.5) / n
        ray = (right[None] * ((u * 2 - 1) * tan_half * aspect)[:, None] + up[None] * ((1 - v * 2) * tan_half)[:, None]
               + forward[None])
        dirs = ray / np.linalg.norm(ray, axis=1, keepdims=True)
        dist = ((zz + 0.5) / n) ** 2 * n * 4000.0
        end = cam_planet + dirs * dist[:, None]
        below = np.linalg.norm(end, axis=1) <= radius_m + 1.0
        moved = end[below] / np.linalg.norm(end[below], axis=1, keepdims=True) * (radius_m + 2.0)
        dirs[below] = (moved - cam_planet) / np.linalg.norm(moved - cam_planet, axis=1, keepdims=True)
        dist[below] = np.linalg.norm(moved - cam_planet, axis=1)
        python = np.zeros(len(zz))
        for z in range(max_slice):
            sel = np.nonzero(zz == z)[0]
            steps = (z + 1) * 2
            t = dist[sel, None] * ((np.arange(steps) + 0.5) / steps)[None]
            pts = (cam + dirs[sel, None, :] * t[..., None]).reshape(-1, 3)
            python[sel] = 1.0 - in_shadow(pts, sun, grid, spheres).reshape(len(sel), steps).mean(1)
        gpu_vis = gpu[zz, yy, xx]
        diff = gpu_vis - python
        results.append({
            "check": f"V3f voxel sun visibility ({case})",
            "voxels": int(len(zz)),
            "gpu_mean_visibility": float(gpu_vis.mean()),
            "python_mean_visibility": float(python.mean()),
            "mean_abs_difference": float(np.abs(diff).mean()),
            "max_abs_difference": float(np.abs(diff).max()),
            "tolerance": None,
            "pass": True,
            "info": "recorded only (design.md V3f)",
        })
    return results


METHODS = (("method0_volume_point", "0: volume (point)"), ("method1_volume_filtered", "1: volume (filtered)"),
           ("method5_per_pixel", "5: per pixel"))
METHOD_CASES = ("ground_sz83", "low_front", "sunset_front")


def check_ap_methods(out_dir: Path) -> list[dict]:
    results = []
    rows = []
    for case in METHOD_CASES:
        geometry, meta = read_exr(out_dir / f"{case}_geometry.exr")
        mask = geometry[..., 3] > 0.5
        ref = rgb(read_exr(out_dir / f"{case}_raymarch_mean.exr")[0])
        noshadow = rgb(read_exr(out_dir / f"{case}_raymarch_mean_noshadow.exr")[0])
        # 空気の影が効く画素: 影なしより 2% 以上暗い（地形の面の影は両方に入っているので、差は空気の影だけ）
        affected = mask & ((noshadow - ref).mean(-1) > 0.02 * noshadow.mean(-1))
        scale = 0.18 / max(np.median(ref.mean(-1)), 1e-12)
        tiles = []
        for name, label in METHODS:
            img = rgb(read_exr(out_dir / f"{case}_{name}.exr")[0])
            rel = np.abs(img - ref).mean(-1) / np.maximum(ref.mean(-1), 1e-30)
            results.append({
                "check": f"V3g AP shadow {label} ({case})",
                "geometry_pixels": int(mask.sum()),
                "affected_pixels": int(affected.sum()),
                "rel_rmse_rgb": rel_rmse(img[mask], ref[mask]),
                "rel_rmse_rgb_affected": rel_rmse(img[affected], ref[affected]) if np.any(affected) else None,
                "p99_rel_error": float(np.percentile(rel[mask], 99)),
                "max_rel_error": float(rel[mask].max()),
                "tolerance": None,
                "pass": bool(np.all(np.isfinite(img[mask]))),
                "info": "recorded only (design.md V3g)",
            })
            tiles += [tonemap_srgb(img * scale), np.ones((img.shape[0], 4, 3))]
        tiles += [tonemap_srgb(ref * scale)]
        rows += [np.concatenate(tiles, axis=1), np.ones((4, 4 * 640 + 12, 3))]
    write_png(ROOT / "docs" / "images" / "phase4_5_ap_methods.png", np.concatenate(rows[:-1]))

    # 地表 1 m のケースの拡大（球の影が空気に入る所）と、基準との相対誤差（赤 = 明るすぎ、青 = 暗すぎ、0〜10%）
    case = "ground_sz83"
    y0, y1, x0, x1 = 120, 260, 120, 440
    ref = rgb(read_exr(out_dir / f"{case}_raymarch_mean.exr")[0])
    mask = read_exr(out_dir / f"{case}_geometry.exr")[0][..., 3] > 0.5
    scale = 0.18 / max(np.median(ref.mean(-1)), 1e-12)
    zoom_rows = []
    for name, _ in METHODS + (("raymarch_mean", "reference"),):
        img = rgb(read_exr(out_dir / f"{case}_{name}.exr")[0])
        signed = (img.mean(-1) - ref.mean(-1)) / np.maximum(ref.mean(-1), 1e-30) * mask
        err = np.stack([np.clip(signed / 0.1, 0, 1), np.zeros_like(signed), np.clip(-signed / 0.1, 0, 1)], -1)
        crop = tonemap_srgb(img * scale)[y0:y1, x0:x1].repeat(2, 0).repeat(2, 1)
        crop_err = err[y0:y1, x0:x1].repeat(2, 0).repeat(2, 1)
        zoom_rows += [np.concatenate([crop, np.ones((crop.shape[0], 4, 3)), crop_err], axis=1),
                      np.ones((4, crop.shape[1] * 2 + 4, 3))]
    write_png(ROOT / "docs" / "images" / "phase4_5_ap_methods_error.png", np.concatenate(zoom_rows[:-1]))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase4_5")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    parser.add_argument("--stride", type=int, default=2, help="V3e で調べるピクセルの間隔")
    parser.add_argument("--ap-stride", type=int, default=4, help="V3f で調べるピクセルの間隔")
    args = parser.parse_args()

    if not args.skip_run:
        run_app(args.exe, "shadow_surface", args.out / "shadow_surface")
        run_app(args.exe, "shadow_aerial_perspective", args.out / "shadow_aerial_perspective")
        run_app(args.exe, "shadow_ap_methods", args.out / "shadow_ap_methods")

    results = check_surface(args.out / "shadow_surface", args.stride)
    results += check_aerial_perspective(args.out / "shadow_aerial_perspective", args.ap_stride,
                                        ROOT / "docs" / "images" / "phase4_5_ap_shadow.png")
    results += check_voxel_visibility(args.out / "shadow_aerial_perspective")
    results += check_ap_methods(args.out / "shadow_ap_methods")
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        if "affected_pixels" in r:
            aff = r["rel_rmse_rgb_affected"]
            line = ("rel RMSE " + " ".join(f"{x:.2%}" for x in r["rel_rmse_rgb"])
                    + " | affected " + (" ".join(f"{x:.2%}" for x in aff) if aff else "-")
                    + f" ({r['affected_pixels']} px) | p99 {r['p99_rel_error']:.2%} max {r['max_rel_error']:.1%}")
        elif "mean_abs_difference" in r:
            line = (f"visibility GPU {r['gpu_mean_visibility']:.4f} Python {r['python_mean_visibility']:.4f}, "
                    f"mean |diff| {r['mean_abs_difference']:.4f}, max {r['max_abs_difference']:.3f}")
        elif "agreement" in r:
            line = (f"agreement {r['agreement']:.2%}, excl. boundary {r['agreement_excluding_boundary']:.2%} "
                    f"(gpu-only shadow {r['gpu_only_shadow_excluding_boundary']}, "
                    f"gpu-only lit {r['gpu_only_lit_excluding_boundary']}, boundary {r['boundary_pixels']}/{r['pixels']})")
        else:
            line = ("volume " + " ".join(f"{x:.2%}" for x in r["volume_vs_raymarch_rel_rmse_rgb"])
                    + " | removed gpu/python " + " ".join(f"{x:.3f}" for x in r["removed_gpu_over_python_rgb"])
                    + " | share " + " ".join(f"{x:.1%}" for x in r["removed_share_of_pixel_rgb"]))
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<44} {line}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
