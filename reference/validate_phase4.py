"""Phase 4 の検証（docs/design.md §8.2 V3c / V3d / V8）。

    reference/.venv/Scripts/python.exe reference/validate_phase4.py

  V3c: 地形の空気遠近法。ボリューム経由の合成 vs 深度までのレイマーチ（同じ LUT）。地形のピクセルで:
         ボリュームの誤差      = ボリューム vs レイマーチ（透過率を RGB 平均）
         RGB 平均の近似の誤差  = レイマーチ（RGB 平均）vs レイマーチ（RGB）
         合計                  = ボリューム vs レイマーチ（RGB）
       相対 RMSE と、奥行きごとの平均相対誤差を記録する（合否はなし）。
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
from validate_phase1 import DEFAULT_EXE, ROOT, run_app  # noqa: E402

RELEASE_EXE = ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe"
CASES = ("day_front", "day_front_g099", "day_back", "sunset_front")
DEPTH_BINS_M = [(0.0, 1000.0), (1000.0, 5000.0), (5000.0, 20000.0), (20000.0, 1e12)]


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
        depth_img, depth_meta = read_exr(out_dir / f"{case}_depth.exr")
        mask = geometry[..., 3] > 0.5
        view_depth = depth_meta["near_m"] / np.maximum(depth_img[..., 0], 1e-30)
        v, rm, rr = rgb(volume)[mask], rgb(ray_mean)[mask], rgb(ray_rgb)[mask]
        bins = []
        for lo, hi in DEPTH_BINS_M:
            sel = (view_depth[mask] >= lo) & (view_depth[mask] < hi)
            if np.any(sel):
                bins.append({"view_depth_m": [lo, hi], "pixels": int(sel.sum()),
                             "volume_mean_rel_error_rgb": (np.mean(np.abs(v[sel] - rm[sel]) / rm[sel], axis=0)).tolist(),
                             "mean_transmittance_rel_error_rgb": (np.mean((rm[sel] - rr[sel]) / rr[sel], axis=0)).tolist()})
        results.append({
            "check": f"V3c aerial perspective ({case})",
            "geometry_pixels": int(mask.sum()),
            "volume_rel_rmse_rgb": rel_rmse(v, rm),
            "volume_max_rel_error": float(np.max(np.abs(v - rm) / rm)),
            "mean_transmittance_rel_rmse_rgb": rel_rmse(rm, rr),
            "total_rel_rmse_rgb": rel_rmse(v, rr),
            "by_view_depth": bins,
            "sun": {"zenith_deg": meta["atmosphere"]["sun_zenith_deg"],
                    "azimuth_deg": meta["atmosphere"]["sun_azimuth_deg"],
                    "mie_g": meta["atmosphere"]["mie_phase_g"]},
            "tolerance": None,
            "pass": bool(np.all(np.isfinite(v))),
            "info": "recorded only (design.md V3c)",
        })
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
        run_app(args.exe, "sky_irradiance", args.out / "sky_irradiance")
        if args.release_exe.exists():
            run_app(args.release_exe, "perf", args.out / "perf")

    results = check_aerial_perspective(args.out / "aerial_perspective") + check_sky_irradiance(args.out / "sky_irradiance")
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        if "volume_rel_rmse_rgb" in r:
            line = ("volume " + " ".join(f"{x:.2%}" for x in r["volume_rel_rmse_rgb"])
                    + f" (max {r['volume_max_rel_error']:.1%}) | meanT " + " ".join(f"{x:.2%}" for x in r["mean_transmittance_rel_rmse_rgb"])
                    + " | total " + " ".join(f"{x:.2%}" for x in r["total_rel_rmse_rgb"]))
        else:
            line = (f"implementation max {r['max_abs_implementation_rel_error']:.2%} | total max {r['max_abs_total_rel_error']:.2%}")
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<44} {line}")

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
