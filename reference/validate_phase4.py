"""Phase 4 の検証（docs/design.md §8.2 V3c / V3d / V8）。

    reference/.venv/Scripts/python.exe reference/validate_phase4.py

  V3c: 地形の空気遠近法。ボリューム経由の合成 vs 深度までのレイマーチ（同じ LUT）。地形のピクセルで:
         ボリュームの誤差      = ボリューム vs レイマーチ（透過率を RGB 平均）
         RGB 平均の近似の誤差  = レイマーチ（RGB 平均）vs レイマーチ（RGB）
         合計                  = ボリューム vs レイマーチ（RGB）
         方式 1 / 2 の合計     = ボリューム（透過率は Transmittance LUT の比 / ボリュームの RGB、§5.7）vs レイマーチ（RGB）
       相対 RMSE と、奥行きごとの平均相対誤差を記録する。基準（§12 Q17）: 合計の相対 RMSE < 1%（RGB 各）。
       方式ごとに判定し、表示の PASS / FAIL は方式 1 / 2 のどちらかが満たすか。
  V3h: 空気遠近法の透過率（design.md §5.7）。地形のピクセルの透過率（RGB）だけを書き出して比べる:
         方式 0 = ボリュームの RGB 平均、方式 1 = Transmittance LUT の比、方式 2 = ボリュームの RGB、
         基準 = 深度までのレイマーチの RGB。
       カメラ C1〜C3、LUT は RGBA32F / RGBA16F。基準（§12 Q17、RGBA32F・地平線 ±0.25° の外）:
         相対 RMSE < 0.5%（RGB 各）、最大 < 2%、分岐の境目（カメラの地平線）に段差がないこと（方式ごとに 1 行）。
       合否は既定の方式 2 の RGBA32F だけ。不採用の方式 1、地平線 ±0.25° の帯、RGBA16F は記録のみ（Q17 met / not met）。
  V3d: 空からの照度。GPU の SH（128×64 方向で射影、Mie の位相関数は l ≤ 2 の Legendre 展開に置き換え）を、
       同じ空の LUT 描画（2048×1024 方向）から求めた照度と比べる。法線 ±X, ±Y, ±Z。
         実装の誤差   = GPU の SH vs 同じ画像を Python で L2 に射影・畳み込みした照度
         L2 の打ち切り = Python の L2 vs 余弦重みの直接の求積（真の照度）
       記録のみ。
  V8 : Release ビルドの perf.json（パスごとの GPU 時間の中央値）を表示する。Release がなければスキップ。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exrio import read_exr  # noqa: E402
from pngio import write_png  # noqa: E402
from validate_phase1 import DEFAULT_EXE, ROOT, run_app  # noqa: E402

RELEASE_EXE = ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe"
CASES = ("day_front", "day_front_g099", "day_back", "sunset_front")
DEPTH_BINS_M = [(0.0, 1000.0), (1000.0, 5000.0), (5000.0, 20000.0), (20000.0, 1e12)]
V3C_LUT_T_TOLERANCE = 0.01          # 方式 1 の合計の相対 RMSE（§12 Q17）
V3H_RMSE_TOLERANCE = 0.005          # 方式 1 の透過率の相対 RMSE
V3H_MAX_TOLERANCE = 0.02            # 方式 1 の透過率の最大相対誤差
V3H_HORIZON_BAND_DEG = 0.25         # カメラの地平線からの仰角がこれ以内は記録のみ
V3H_CAMERAS = (("c1", ("fp32", "fp16")), ("c2", ("fp32", "fp16")), ("c3", ("fp32",)))
# 地平線からの仰角の区分 [deg]（負 = 地平線より下）
ELEVATION_BINS_DEG = [(-90.0, -1.0), (-1.0, -0.25), (-0.25, 0.25), (0.25, 1.0), (1.0, 90.0)]


def rgb(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float64)


def rel_rmse(test: np.ndarray, ref: np.ndarray) -> list[float]:
    return (np.sqrt(np.mean((test - ref) ** 2, axis=0)) / np.mean(ref, axis=0)).tolist()


# ---------------------------------------------------------------------- V3c
def check_aerial_perspective(out_dir: Path) -> list[dict]:
    results = []
    for case in CASES:
        geometry, meta = read_exr(out_dir / f"{case}_geometry.exr")
        volume, _ = read_exr(out_dir / f"{case}_volume.exr")
        ray_rgb, _ = read_exr(out_dir / f"{case}_raymarch_rgb.exr")
        ray_mean, _ = read_exr(out_dir / f"{case}_raymarch_mean.exr")
        volume_lut_t, _ = read_exr(out_dir / f"{case}_volume_lut_t.exr")
        volume_rgb_t, _ = read_exr(out_dir / f"{case}_volume_rgb_t.exr")
        depth_img, depth_meta = read_exr(out_dir / f"{case}_depth.exr")
        mask = geometry[..., 3] > 0.5
        view_depth = depth_meta["near_m"] / np.maximum(depth_img[..., 0], 1e-30)
        v, rm, rr = rgb(volume)[mask], rgb(ray_mean)[mask], rgb(ray_rgb)[mask]
        vt = rgb(volume_lut_t)[mask]
        vr = rgb(volume_rgb_t)[mask]
        bins = []
        for lo, hi in DEPTH_BINS_M:
            sel = (view_depth[mask] >= lo) & (view_depth[mask] < hi)
            if np.any(sel):
                bins.append({"view_depth_m": [lo, hi], "pixels": int(sel.sum()),
                             "volume_mean_rel_error_rgb": (np.mean(np.abs(v[sel] - rm[sel]) / rm[sel], axis=0)).tolist(),
                             "mean_transmittance_rel_error_rgb": (np.mean((rm[sel] - rr[sel]) / rr[sel], axis=0)).tolist(),
                             "lut_t_mean_rel_error_rgb": (np.mean((vt[sel] - rr[sel]) / rr[sel], axis=0)).tolist(),
                             "rgb_t_mean_rel_error_rgb": (np.mean((vr[sel] - rr[sel]) / rr[sel], axis=0)).tolist()})
        lut_t_rmse = rel_rmse(vt, rr)
        rgb_t_rmse = rel_rmse(vr, rr)
        results.append({
            "check": f"V3c aerial perspective ({case})",
            "geometry_pixels": int(mask.sum()),
            "volume_rel_rmse_rgb": rel_rmse(v, rm),
            "volume_max_rel_error": float(np.max(np.abs(v - rm) / rm)),
            "mean_transmittance_rel_rmse_rgb": rel_rmse(rm, rr),
            "total_rel_rmse_rgb": rel_rmse(v, rr),
            "lut_t_total_rel_rmse_rgb": lut_t_rmse,
            "lut_t_total_max_rel_error": float(np.max(np.abs(vt - rr) / rr)),
            "lut_t_pass": max(lut_t_rmse) < V3C_LUT_T_TOLERANCE,
            "rgb_t_total_rel_rmse_rgb": rgb_t_rmse,
            "rgb_t_total_max_rel_error": float(np.max(np.abs(vr - rr) / rr)),
            "rgb_t_pass": max(rgb_t_rmse) < V3C_LUT_T_TOLERANCE,
            "by_view_depth": bins,
            "sun": {"zenith_deg": meta["atmosphere"]["sun_zenith_deg"],
                    "azimuth_deg": meta["atmosphere"]["sun_azimuth_deg"],
                    "mie_g": meta["atmosphere"]["mie_phase_g"]},
            "tolerance": {"lut_t_total_rel_rmse": V3C_LUT_T_TOLERANCE},
            "pass": bool(np.all(np.isfinite(v)) and np.all(np.isfinite(vt)) and np.all(np.isfinite(vr))
                         and (max(lut_t_rmse) < V3C_LUT_T_TOLERANCE or max(rgb_t_rmse) < V3C_LUT_T_TOLERANCE)),
            "info": "volume / meanT / total recorded only; criterion per RGB-transmittance method (design.md V3c, Q17)",
        })
    return results


# ---------------------------------------------------------------------- V3h
def pixel_elevation_deg(meta: dict, h: int, w: int) -> np.ndarray:
    """ピクセル中心の視線の仰角 [deg]（scene::Camera と同じ対称な透視投影、ヨーは仰角に効かない）"""

    tan_half = np.tan(np.radians(meta["vertical_fov_deg"]) * 0.5)
    pitch = np.radians(meta["camera_pitch_deg"])
    ndc_y = 1.0 - (np.arange(h) + 0.5) / h * 2.0
    ndc_x = (np.arange(w) + 0.5) / w * 2.0 - 1.0
    x = ndc_x[None, :] * tan_half * (w / h)
    y = ndc_y[:, None] * tan_half
    # カメラ空間 (x, y, −1) をピッチで回す: 前 = (0, sin p, −cos p)、上 = (0, cos p, sin p)
    world_y = np.sin(pitch) + y * np.cos(pitch)
    length = np.sqrt(x * x + y * y + 1.0)
    return np.degrees(np.arcsin(np.clip(world_y / length, -1.0, 1.0)))


def horizon_dip_deg(meta: dict) -> float:
    """カメラから見た地表の球の地平線の仰角 [deg]（負）。−acos(R / (R + h))"""

    r_bottom = meta["bottom_radius_km"]
    r = r_bottom + meta["camera_height_m"] * 1e-3
    return -float(np.degrees(np.arccos(r_bottom / r)))


def boundary_step(rel_error: np.ndarray, mask: np.ndarray, elevation_from_horizon: np.ndarray) -> dict:
    """分岐の境目の段差: 列ごとに、縦に隣り合う 2 ピクセル（両方が地形）の相対誤差の差（RGB の最大）を求め、
    境目をまたぐ組と、それ以外の組の分布を比べる。境目の組の中央値が、それ以外の組の 99 パーセンタイル以下なら段差なし。"""

    both = mask[:-1] & mask[1:]
    jump = np.max(np.abs(rel_error[1:] - rel_error[:-1]), axis=-1)
    crosses = (np.sign(elevation_from_horizon[:-1]) != np.sign(elevation_from_horizon[1:])) & both
    others = both & ~crosses
    if not np.any(crosses):
        return {"boundary_pairs": 0, "pass": True, "info": "the camera horizon does not fall on terrain"}
    boundary_median = float(np.median(jump[crosses]))
    others_p99 = float(np.percentile(jump[others], 99.0))
    return {"boundary_pairs": int(crosses.sum()), "boundary_jump_median": boundary_median,
            "boundary_jump_max": float(np.max(jump[crosses])), "other_jump_p99": others_p99,
            "pass": boundary_median <= others_p99}


def signed_error_rgb(err: np.ndarray, mask: np.ndarray, saturate: float) -> np.ndarray:
    """符号付きの相対誤差（B）を色にする: 赤 = 明るすぎ（透過率が大きい）、青 = 暗すぎ。地形でない所は灰"""

    x = np.clip(err / saturate, -1.0, 1.0)
    img = np.stack([1.0 - np.maximum(-x, 0.0), 1.0 - np.abs(x), 1.0 - np.maximum(x, 0.0)], axis=-1)
    img[~mask] = 0.35
    return img


def profile_panel(rel_error: np.ndarray, mask: np.ndarray, horizon_row: float, width: int, saturate: float) -> np.ndarray:
    """中央の列の相対誤差（R/G/B）の縦方向のプロファイル。横軸 = 誤差（±saturate）、縦 = 画面の行。黄の線 = 地平線"""

    h = rel_error.shape[0]
    img = np.ones((h, width, 3))
    img[:, width // 2] = 0.6
    if 0 <= horizon_row < h:
        img[int(horizon_row)] = (0.9, 0.75, 0.0)
    col = rel_error.shape[1] // 2
    colors = ((0.85, 0.1, 0.1), (0.1, 0.6, 0.1), (0.1, 0.2, 0.9))
    for y in range(h):
        if not mask[y, col]:
            continue
        for c in range(3):
            x = int(round((np.clip(rel_error[y, col, c] / saturate, -1.0, 1.0) * 0.5 + 0.5) * (width - 1)))
            img[y, max(x - 1, 0):x + 2] = colors[c]
    return img


def check_ap_transmittance(out_dir: Path, image_path: Path) -> list[dict]:
    results = []
    rows = []
    for camera, formats in V3H_CAMERAS:
        for fmt in formats:
            t = {}
            meta = None
            for path in ("volume_mean", "lut_ratio", "volume_rgb", "raymarch_rgb", "raymarch_mean"):
                img, m = read_exr(out_dir / f"{camera}_{fmt}_t_{path}.exr")
                t[path] = rgb(img)
                meta = m
            depth_img, depth_meta = read_exr(out_dir / f"{camera}_{fmt}_depth.exr")
            mask = depth_img[..., 0] > 0.0
            h, w = mask.shape
            view_depth = depth_meta["near_m"] / np.maximum(depth_img[..., 0], 1e-30)
            dip = horizon_dip_deg(meta)
            from_horizon = pixel_elevation_deg(meta, h, w) - dip
            ref = t["raymarch_rgb"]
            err0 = (t["volume_mean"] - ref) / ref
            err1 = (t["lut_ratio"] - ref) / ref
            err2 = (t["volume_rgb"] - ref) / ref
            outside = mask & (np.abs(from_horizon) > V3H_HORIZON_BAND_DEG)
            band = mask & ~outside

            def stats(err: np.ndarray, sel: np.ndarray) -> dict:
                if not np.any(sel):
                    return {"pixels": 0}
                e = err[sel]
                return {"pixels": int(sel.sum()), "rel_rmse_rgb": np.sqrt(np.mean(e * e, axis=0)).tolist(),
                        "max_abs_rel_error": float(np.max(np.abs(e))), "mean_rel_error_rgb": np.mean(e, axis=0).tolist()}

            # 方式 0 とレイマーチの RGB 平均の差 = ボリュームの補間の誤差だけ（確認用）
            volume_only = stats((t["volume_mean"] - t["raymarch_mean"]) / t["raymarch_mean"], mask)
            for method, name, err in ((1, "lut_ratio", err1), (2, "volume_rgb", err2)):
                by_depth = []
                for lo, hi in DEPTH_BINS_M:
                    sel = mask & (view_depth >= lo) & (view_depth < hi)
                    if np.any(sel):
                        by_depth.append({"view_depth_m": [lo, hi], "method0": stats(err0, sel), "method": stats(err, sel)})
                by_elevation = []
                for lo, hi in ELEVATION_BINS_DEG:
                    sel = mask & (from_horizon >= lo) & (from_horizon < hi)
                    if np.any(sel):
                        by_elevation.append({"from_horizon_deg": [lo, hi], "method0": stats(err0, sel),
                                             "method": stats(err, sel)})
                step = boundary_step(err, mask, from_horizon)
                m = stats(err, outside)
                finite = bool(np.all(np.isfinite(t[name][mask])))
                meets = finite and max(m["rel_rmse_rgb"]) < V3H_RMSE_TOLERANCE \
                    and m["max_abs_rel_error"] < V3H_MAX_TOLERANCE and step["pass"]
                # [POINT] 合否で判定するのは既定の方式 2 の RGBA32F だけ。方式 1 は不採用（design.md §14 Phase 4.6）なので
                #         基準を満たすかを meets_q17_criteria に記録するだけにする（終了コードを左右しない）。
                graded = fmt == "fp32" and method == 2
                ok = meets if graded else finite
                results.append({
                    "check": f"V3h AP transmittance ({camera}, {fmt}, method {method})",
                    "method": {"id": method, "name": name},
                    "camera": {k: meta[k] for k in ("camera_height_m", "camera_pitch_deg", "vertical_fov_deg")},
                    "horizon_dip_deg": dip,
                    "terrain_pixels": int(mask.sum()),
                    "method0_all": stats(err0, mask),
                    "method_all": stats(err, mask),
                    "method_outside_horizon_band": m,
                    "method_horizon_band": stats(err, band),
                    "volume_interpolation_only": volume_only,
                    "by_view_depth": by_depth,
                    "by_elevation_from_horizon": by_elevation,
                    "boundary_step": step,
                    "tolerance": ({"rel_rmse": V3H_RMSE_TOLERANCE, "max_rel_error": V3H_MAX_TOLERANCE,
                                   "horizon_band_excluded_deg": V3H_HORIZON_BAND_DEG}),
                    "meets_q17_criteria": meets,
                    "pass": ok,
                    "info": "criterion on the default method 2 with fp32, outside the horizon band (design.md V3h, Q17)"
                            if graded else "recorded only (method 1 rejected / RGBA16F LUT)",
                })
            # 確認画像: 方式 0 / 1 / 2 の符号付き相対誤差（B、±5%）と、中央の列の方式 1 / 2 のプロファイル（±1%）
            horizon_row = float(np.interp(0.0, from_horizon[::-1, w // 2], np.arange(h)[::-1]))
            sep = np.ones((h, 4, 3))
            rows.append(np.concatenate([signed_error_rgb(err0[..., 2], mask, 0.05), sep,
                                        signed_error_rgb(err1[..., 2], mask, 0.05), sep,
                                        signed_error_rgb(err2[..., 2], mask, 0.05), sep,
                                        profile_panel(err1, mask, horizon_row, 200, 0.01), sep,
                                        profile_panel(err2, mask, horizon_row, 200, 0.01)], axis=1))
            rows.append(np.ones((4, rows[-1].shape[1], 3)))
    write_png(image_path, np.concatenate(rows[:-1]))
    return results


# ---------------------------------------------------------------------- V3d
def sh_basis(n: np.ndarray) -> np.ndarray:
    """shaders/common/SphericalHarmonics.hlsli の ShBasis と同じ基底（形状 (..., 9)）"""

    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], axis=-1)


COSINE_LOBE = np.array([np.pi] + [2 * np.pi / 3] * 3 + [np.pi / 4] * 5)
NORMALS = {"+X": [1, 0, 0], "-X": [-1, 0, 0], "+Y": [0, 1, 0], "-Y": [0, -1, 0], "+Z": [0, 0, 1], "-Z": [0, 0, -1]}


def check_sky_irradiance(out_dir: Path) -> list[dict]:
    entries = json.loads((out_dir / "sky_irradiance.json").read_text(encoding="utf-8"))
    results = []
    for e in entries:
        image, _ = read_exr(out_dir / e["image"])
        h, w = image.shape[:2]
        phi = 2 * np.pi * (np.arange(w) + 0.5) / w
        el = 0.5 * np.pi - np.pi * (np.arange(h) + 0.5) / h
        cos_el = np.cos(el)[:, None]
        dirs = np.stack([cos_el * np.cos(phi)[None, :], np.repeat(np.sin(el)[:, None], w, axis=1),
                         cos_el * np.sin(phi)[None, :]], axis=-1).reshape(-1, 3)
        d_omega = ((2 * np.pi / w) * (np.pi / h) * np.repeat(np.cos(el), w)).reshape(-1, 1)
        radiance = rgb(image).reshape(-1, 3)
        # 真の照度: 余弦重みの直接の求積
        normals = np.array(list(NORMALS.values()), dtype=np.float64)
        cos_term = np.maximum(dirs @ normals.T, 0.0)                        # (N, 6)
        exact = cos_term.T @ (radiance * d_omega)                           # (6, 3)
        # Python の L2: 同じ画像を SH に射影し、余弦の畳み込み
        coeff_py = (sh_basis(dirs) * d_omega).T @ radiance * COSINE_LOBE[:, None]   # (9, 3)
        l2_py = sh_basis(normals) @ coeff_py
        gpu = sh_basis(normals) @ np.asarray(e["irradiance_sh_per_lux"], dtype=np.float64)
        implementation = (gpu - l2_py) / l2_py
        truncation = (l2_py - exact) / exact
        results.append({
            "check": f"V3d sky irradiance g={e['mie_phase_g']:.2f} sz={e['sun_zenith_deg']:g}deg",
            "normals": list(NORMALS.keys()),
            "exact_irradiance_per_lux": exact.tolist(),
            "gpu_sh_irradiance_per_lux": gpu.tolist(),
            "implementation_rel_error": implementation.tolist(),
            "l2_truncation_rel_error": truncation.tolist(),
            "total_rel_error": ((gpu - exact) / exact).tolist(),
            "max_abs_implementation_rel_error": float(np.max(np.abs(implementation))),
            "max_abs_total_rel_error": float(np.max(np.abs((gpu - exact) / exact))),
            "tolerance": None,
            "pass": True,
            "info": "recorded only (design.md V3d)",
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--release-exe", type=Path, default=RELEASE_EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase4")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    args = parser.parse_args()

    if not args.skip_run:
        run_app(args.exe, "aerial_perspective", args.out / "aerial_perspective")
        run_app(args.exe, "ap_transmittance", args.out / "ap_transmittance")
        run_app(args.exe, "sky_irradiance", args.out / "sky_irradiance")
        if args.release_exe.exists():
            run_app(args.release_exe, "perf", args.out / "perf")

    results = (check_aerial_perspective(args.out / "aerial_perspective")
               + check_ap_transmittance(args.out / "ap_transmittance",
                                        ROOT / "docs" / "images" / "phase4_6_ap_transmittance.png")
               + check_sky_irradiance(args.out / "sky_irradiance"))
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        if "volume_rel_rmse_rgb" in r:
            line = ("volume " + " ".join(f"{x:.2%}" for x in r["volume_rel_rmse_rgb"])
                    + f" (max {r['volume_max_rel_error']:.1%}) | meanT " + " ".join(f"{x:.2%}" for x in r["mean_transmittance_rel_rmse_rgb"])
                    + " | total " + " ".join(f"{x:.2%}" for x in r["total_rel_rmse_rgb"])
                    + " | LUT-T total " + " ".join(f"{x:.2%}" for x in r["lut_t_total_rel_rmse_rgb"])
                    + " | volRGB-T total " + " ".join(f"{x:.2%}" for x in r["rgb_t_total_rel_rmse_rgb"]))
        elif "method_outside_horizon_band" in r:
            m0, m1, band = r["method0_all"], r["method_outside_horizon_band"], r["method_horizon_band"]
            line = ("meanT " + " ".join(f"{x:.2%}" for x in m0["rel_rmse_rgb"])
                    + " | T " + " ".join(f"{x:.3%}" for x in m1["rel_rmse_rgb"]) + f" (max {m1['max_abs_rel_error']:.2%})"
                    + (" | band " + " ".join(f"{x:.2%}" for x in band["rel_rmse_rgb"]) + f" (max {band['max_abs_rel_error']:.1%})"
                       if band["pixels"] else " | band -")
                    + f" | step {'ok' if r['boundary_step']['pass'] else 'NG'}"
                    + f" | Q17 {'met' if r['meets_q17_criteria'] else 'not met'}")
        else:
            line = (f"implementation max {r['max_abs_implementation_rel_error']:.2%} | total max {r['max_abs_total_rel_error']:.2%}")
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<52} {line}")

    perf_path = args.out / "perf" / "perf.json"
    if perf_path.exists():
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        print(f"V8 perf ({perf.get('build')}, {perf.get('adapter')})")
        for key, passes in perf.items():
            if isinstance(passes, dict):
                print(f"  {key}")
                for name, stats in passes.items():
                    print(f"    {name:<40} {stats['median_ms']:.3f} ms (min {stats['min_ms']:.3f}, max {stats['max_ms']:.3f})")
    else:
        print("V8 perf: skipped (Release build not found)")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
