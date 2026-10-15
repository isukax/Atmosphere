"""Phase 2 の数値検証（docs/design.md §8.2 V1 / V2）。

    reference/.venv/Scripts/python.exe reference/validate_phase2.py

  V1: Transmittance LUT（GPU、float32、台形則 500 区間）vs 倍精度の Gauss-Legendre 求積
      全テクセルで相対誤差 < 1e-3。GPU が実際に使った (r, mu) で評価する（積分の検証）。
      別途、LUT のパラメタ化（UV → r, mu）を倍精度の式と照合する。
      40 区間（論文の PC 設定）の誤差は参考値として記録する。
  V2: 単散乱の GPU レイマーチ（太陽方向は入れ子レイマーチ = LUT なし）vs 倍精度の求積
      方向グリッド 32×16 × 6 条件（高度 0.2 / 10 km × 太陽天頂角 30 / 60 / 85°）で相対誤差 < 1e-3。
      LUT モード（Transmittance LUT 40 区間・RGBA32F）の誤差は参考値として記録する。
  リファレンス自身の精度は、求積の区間数を倍にした結果との差で確認する（< 1e-6）。
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
from validate_phase1 import DEFAULT_EXE, ROOT, run_app  # noqa: E402

V1_REL_TOLERANCE = 1e-3
V2_REL_TOLERANCE = 1e-3
REFERENCE_SELF_TOLERANCE = 1e-6


def relative_error(gpu: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(gpu.astype(np.float64) - ref) / np.abs(ref)


def check_transmittance(out_dir: Path) -> list[dict]:
    r_mu, meta = read_exr(out_dir / "transmittance_r_mu.exr")
    atm = Atmosphere.from_meta(meta)
    h, w = r_mu.shape[:2]
    r_gpu = r_mu[..., 0].astype(np.float64).ravel()
    mu_gpu = r_mu[..., 1].astype(np.float64).ravel()

    # リファレンス自身の収束確認
    ref = atm.transmittance_to_top(r_gpu, mu_gpu, panels=32, order=4)
    ref_fine = atm.transmittance_to_top(r_gpu, mu_gpu, panels=64, order=4)
    self_err = float(np.max(np.abs(ref - ref_fine) / ref_fine))

    results = []
    for steps in (500, 40):
        lut, _ = read_exr(out_dir / f"transmittance_steps{steps}.exr")
        err = relative_error(lut[..., :3].reshape(-1, 3), ref)
        worst = np.unravel_index(np.argmax(err), err.shape)
        entry = {
            "check": f"V1 transmittance LUT ({steps} steps)",
            "max_rel_error": float(err.max()),
            "mean_rel_error": float(err.mean()),
            "worst_texel": [int(worst[0] % w), int(worst[0] // w)],
            "worst_r_km_mu": [float(r_gpu[worst[0]]), float(mu_gpu[worst[0]])],
            "reference_self_error": self_err,
        }
        if steps == 500:
            entry.update(tolerance=V1_REL_TOLERANCE,
                         **{"pass": bool(err.max() < V1_REL_TOLERANCE and self_err < REFERENCE_SELF_TOLERANCE)})
        else:
            entry.update(tolerance=None, **{"pass": True, "info": "paper PC step count; informational"})
        results.append(entry)

    # パラメタ化の照合: テクセル中心の UV → (r, mu) を倍精度で計算し、GPU の値と比べる
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    r_ref, mu_ref = atm.transmittance_uv_to_r_mu(uu.ravel(), vv.ravel(), w, h)
    worst_mu = int(np.argmax(np.abs(mu_gpu - mu_ref)))
    # [POINT] 判定は「GPU の (r, mu) を倍精度で UV に戻したとき、元のテクセル中心から何テクセルずれるか」。
    #         mu の絶対差は、大気上端付近の真上方向（d ≈ 0）で float32 の桁落ちにより 5e-5 程度になるが、
    #         そこは透過率がほぼ 1 で mu に鈍感なので、LUT 空間でのずれで評価する。
    u_back, v_back = atm.transmittance_r_mu_to_uv(r_gpu, mu_gpu, w, h)
    texel_error = np.maximum(np.abs(u_back - uu.ravel()) * w, np.abs(v_back - vv.ravel()) * h)
    results.append({
        "check": "V1 LUT parameterization (uv->r,mu)",
        "max_texel_error": float(texel_error.max()),
        "max_abs_error_r_km_info": float(np.max(np.abs(r_gpu - r_ref))),
        "max_abs_error_mu_info": float(np.max(np.abs(mu_gpu - mu_ref))),
        "worst_mu_texel_info": [worst_mu % w, worst_mu // w],
        "tolerance": "round trip < 1e-2 texel",
        "pass": bool(texel_error.max() < 1e-2),
    })
    return results


def check_single_scattering(out_dir: Path) -> list[dict]:
    results = []
    for altitude in (0.2, 10.0):
        for sun_zenith in (30.0, 60.0, 85.0):
            stem = f"ss_h{altitude:g}_sz{sun_zenith:g}"
            gpu_nested, meta = read_exr(out_dir / f"{stem}_nested.exr")
            gpu_lut40, _ = read_exr(out_dir / f"{stem}_lut40.exr")
            gpu_lut500, _ = read_exr(out_dir / f"{stem}_lut500.exr")
            dirs, _ = read_exr(out_dir / f"{stem}_directions.exr")
            atm = Atmosphere.from_meta(meta)
            a = meta["atmosphere"]
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            camera = np.array([0.0, atm.bottom_radius_km + meta["camera_altitude_km"], 0.0])
            d = dirs[..., :3].reshape(-1, 3).astype(np.float64)

            ref = atm.single_scattering_per_lux(camera, d, sun, outer_panels=32, inner_panels=8)
            # リファレンス自身の収束確認（一部の方向で区間数を倍に）
            subset = slice(0, None, 37)
            ref_fine = atm.single_scattering_per_lux(camera, d[subset], sun, outer_panels=64, inner_panels=16)
            self_err = float(np.max(np.abs(ref[subset] - ref_fine) / ref_fine))

            err = relative_error(gpu_nested[..., :3].reshape(-1, 3), ref)
            err_lut40 = relative_error(gpu_lut40[..., :3].reshape(-1, 3), ref)
            err_lut500 = relative_error(gpu_lut500[..., :3].reshape(-1, 3), ref)
            worst = int(np.unravel_index(np.argmax(err), err.shape)[0])
            results.append({
                "check": f"V2 single scattering h={altitude:g}km sz={sun_zenith:g}deg",
                "max_rel_error": float(err.max()),
                "mean_rel_error": float(err.mean()),
                "worst_direction": d[worst].tolist(),
                "worst_gpu": gpu_nested[..., :3].reshape(-1, 3)[worst].tolist(),
                "worst_ref": ref[worst].tolist(),
                "lut40_max_rel_error_info": float(err_lut40.max()),
                "lut40_mean_rel_error_info": float(err_lut40.mean()),
                "lut500_max_rel_error_info": float(err_lut500.max()),
                "lut500_mean_rel_error_info": float(err_lut500.mean()),
                "reference_self_error": self_err,
                "tolerance": V2_REL_TOLERANCE,
                "pass": bool(err.max() < V2_REL_TOLERANCE and self_err < REFERENCE_SELF_TOLERANCE),
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase2")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    args = parser.parse_args()

    if not args.skip_run:
        run_app(args.exe, "transmittance", args.out / "transmittance")
        run_app(args.exe, "single_scattering", args.out / "single_scattering")

    results = check_transmittance(args.out / "transmittance") + check_single_scattering(args.out / "single_scattering")
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for r in results:
        metric = r.get("max_rel_error", r.get("max_texel_error"))
        extra = ""
        if "lut40_max_rel_error_info" in r:
            extra = f"  (lut40 {r['lut40_max_rel_error_info']:.2e}, lut500 {r['lut500_max_rel_error_info']:.2e})"
        if "reference_self_error" in r:
            extra += f"  ref-self {r['reference_self_error']:.1e}"
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<42} max={metric:.3e}{extra}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
