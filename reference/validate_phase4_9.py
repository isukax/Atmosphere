"""Phase 4.9 の検証（docs/design.md §5.9、Mie の 2 成分（D）と自由大気の背景の項。基準は §12 Q20）。

    reference/.venv/Scripts/python.exe reference/validate_phase4_9.py [--skip-run] [--only ...]

アプリの aerosol_presets（2 成分のプリセットを含む）と aerosol_scene の出力（dumps/phase4_9）を確かめる。
1 成分のプリセットは Phase 4.8 から変わらないこと（画素の回帰、compare_dumps.py）で確かめるので、ここでは扱わない。
  背景: JSON の a・H_FT・σ から求めた AOD550（成分ごと・背景の項だけ）が、プリセットの定義
        （fine 0.07 のうち背景 0.015、coarse 0.03）と相対差 < 1e-6。
  V3i / V3j / V3k / V3b: validate_phase4_8.py と同じ関数・同じ基準を、C+D・C+A+B+D の組み合わせで。
  V4b: Hillaire2020 の C+A+B+D の GPU の LUT 描画 vs パストレース（validate_v4b.py で新しい分布で取り直した
       dumps/v4b）。記録のみ（既知の限界）。
  V9: SpectralFit709 / 2020 の C+A+B+D の単散乱（GPU）vs 分光リファレンス（fine / coarse を別の成分にして、
      それぞれ Mie 理論の σ(λ)・p(θ, λ)）。記録のみ。
  scene: aerosol_scene（Hillaire2020 の C+A+B+D と、比較用に同じ量の 1 成分の C+A+B）の、地形の空気遠近法（V3c と同じ指標）・空気遠近法の影の方式
         （V3g と同じ指標）・空からの照度（V3d と同じ指標）。記録のみ（2 成分の LUT・ボリューム・合成・影の方式 5・
         SH の経路が動き、有限の値を出すことを確かめる）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import colour
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerosol_models as am  # noqa: E402
import derive_coefficients as dc  # noqa: E402
import spectral_data as sd  # noqa: E402
import validate_phase4 as vp4  # noqa: E402
import validate_phase4_5 as vp45  # noqa: E402
import validate_phase4_8 as v48  # noqa: E402
from atmosphere_ref import Atmosphere, sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from validate_phase1 import ROOT, delta_e_itp, run_app  # noqa: E402
from validate_phase3 import WORKING_SPACES, rgb  # noqa: E402

# Phase48Tests.cpp の kCombos の 2 成分の組み合わせ（C+D の灰色の Mie は色空間で変わらないので Hillaire2020 だけ）
D_COMBOS = (("Hillaire2020", "C+D"), ("Hillaire2020", "C+A+B+D"),
            ("SpectralFit709", "C+A+B+D"), ("SpectralFit2020", "C+A+B+D"))
BACKGROUND_TOLERANCE = 1e-6
SCENE_AP_CASES = ("day_front", "day_back", "sunset_front")  # aerosol_scene は g を変えたケースを描かない


# ---------------------------------------------------------------------- 背景の項
def check_background(coeff: dict) -> list[dict]:
    """プリセットの AOD550 の内訳。Hillaire2020 は G = 550 nm なので値そのもの、分光フィットのセットは G が 550 nm と
    違うので、灰色の C+D だけ値を、C+A+B+D は背景の割合（波長によらない）だけを比べる。"""

    expected_bg_fraction = am.FREE_TROP_FRACTION_OF_AOD / am.FINE_FRACTION_OF_AOD
    aod = am.AOD550_REALISTIC
    results = []
    for s in coeff["sets"]:
        for preset in ("C+D", "C+A+B+D"):
            atm = am.atmosphere_from_json(s, preset)
            fine, coarse = atm.mie
            fine_aod = am.column_km(fine.profile) * (fine.scattering + fine.absorption)
            fine_bg = am.column_km(fine.profile.free_troposphere) * (fine.scattering + fine.absorption)
            coarse_aod = am.column_km(coarse.profile) * (coarse.scattering + coarse.absorption)
            errors = {"fine_background_fraction": float(np.max(np.abs(fine_bg / fine_aod - expected_bg_fraction))
                                                        / expected_bg_fraction),
                      "coarse_background_density": float(coarse.profile.free_trop_density)}
            if s["name"] == "Hillaire2020" or preset == "C+D":
                g = am.G_CHANNEL
                errors["fine_aod550"] = abs(fine_aod[g] - am.FINE_FRACTION_OF_AOD * aod) / (am.FINE_FRACTION_OF_AOD * aod)
                errors["fine_background_aod550"] = (abs(fine_bg[g] - am.FREE_TROP_FRACTION_OF_AOD * aod)
                                                    / (am.FREE_TROP_FRACTION_OF_AOD * aod))
                errors["coarse_aod550"] = (abs(coarse_aod[g] - (1 - am.FINE_FRACTION_OF_AOD) * aod)
                                           / ((1 - am.FINE_FRACTION_OF_AOD) * aod))
            worst = max(v for k, v in errors.items() if k != "coarse_background_density")
            results.append({"check": f"background term ({s['name']}, {preset})",
                            "relative_errors": errors,
                            "fine_free_troposphere_density": fine.profile.free_trop_density,
                            "fine_free_troposphere_scale_height_km": fine.profile.free_trop_scale_height_km,
                            "fine_aod_rgb": fine_aod.tolist(), "fine_background_aod_rgb": fine_bg.tolist(),
                            "coarse_aod_rgb": coarse_aod.tolist(),
                            "tolerance": BACKGROUND_TOLERANCE,
                            "pass": bool(worst < BACKGROUND_TOLERANCE and errors["coarse_background_density"] == 0.0)})
    return results


# ---------------------------------------------------------------------- V9（2 成分）
def spectral_mie_components(coeff: dict, preset: str, wl: np.ndarray):
    """5 nm 刻みの Mie を、fine / coarse の成分ごとに (σs(λ), σt(λ), 位相関数の表 (U, L), 鉛直分布)。
    量はプリセットと同じ（AOD550 0.1 の fine 70%・coarse 30%、各成分の柱で割る）。鉛直分布は JSON の値"""

    profiles = [c.profile for c in am.atmosphere_from_json(
        next(s for s in coeff["sets"] if s["name"] == "Hillaire2020"), preset).mie]
    mie_wl = np.arange(360.0, 830.0 + dc.MIE_WAVELENGTH_STEP_NM / 2, dc.MIE_WAVELENGTH_STEP_NM)
    interp = dc.interpolation_matrix(wl, mie_wl)
    i550 = int(np.argmin(np.abs(mie_wl - 550.0)))
    out = []
    for mode, frac, profile in ((am.FINE, am.FINE_FRACTION_OF_AOD, profiles[0]),
                                (am.COARSE, 1.0 - am.FINE_FRACTION_OF_AOD, profiles[1])):
        spec = dc.spectral_mode(mode, mie_wl)
        ext550 = frac * am.AOD550_REALISTIC / am.column_km(profile)
        sigma_s = ext550 * (interp @ spec.c_sca) / spec.c_ext[i550]
        sigma_t = ext550 * (interp @ spec.c_ext) / spec.c_ext[i550]
        phase = (interp @ spec.phase).T  # 1 つのモードなので σs の重みは要らない
        out.append((sigma_s, sigma_t, phase, profile))
    return out


def check_v9_d(out: Path, coeff: dict) -> list[dict]:
    wl = np.arange(360.0, 830.0 + 1e-9, 5.0)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
        colour.SpectralShape(360.0, 830.0, 5.0)).values
    solar = sd.solar_irradiance(wl)
    dirs, _ = read_exr(out / "single" / "directions.exr")
    d = dirs[..., :3].reshape(-1, 3).astype(np.float64)[::4]  # 128 方向（validate_phase4_8.py の V9 と同じ）
    (s0, t0, p0, prof0), (s1, t1, p1, prof1) = spectral_mie_components(coeff, "C+A+B+D", wl)
    results = []
    for sun_zenith in (30.0, 60.0, 85.0):
        ref_xyz = None
        for set_name in ("SpectralFit709", "SpectralFit2020"):
            gpu, meta = read_exr(out / "single" / f"{set_name}_C_A_B_D_h0.2_sz{sun_zenith:g}.exr")
            a = meta["atmosphere"]
            if ref_xyz is None:
                atm = dataclasses.replace(
                    Atmosphere.spectral(meta, wl),
                    mie_scattering_per_km=s0, mie_extinction_per_km=t0, mie_phase_table=p0, mie_phase_g=am.BASE_G,
                    mie_boundary_layer_top_km=prof0.top_km, mie_scale_height_km=prof0.scale_height_km,
                    mie_free_trop_density=prof0.free_trop_density,
                    mie_free_trop_scale_height_km=prof0.free_trop_scale_height_km,
                    mie1_scattering_per_km=s1, mie1_extinction_per_km=t1, mie1_phase_table=p1, mie1_phase_g=am.BASE_G,
                    mie1_boundary_layer_top_km=prof1.top_km, mie1_scale_height_km=prof1.scale_height_km,
                    mie1_free_trop_density=prof1.free_trop_density,
                    mie1_free_trop_scale_height_km=prof1.free_trop_scale_height_km)
                camera = np.array([0.0, atm.bottom_radius_km + float(meta["camera_altitude_km"]), 0.0])
                sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
                per_irradiance = atm.single_scattering_per_lux(camera, d, sun, outer_panels=16, inner_panels=4, chunk=2)
                spectral_lux = 683.0 * np.sum(solar * cmfs[:, 1]) * 5.0
                scale = float(a["sun_illuminance_lux"]) / spectral_lux
                ref_xyz = 683.0 * (per_irradiance * solar) @ cmfs * 5.0 * scale
            cs = colour.RGB_COLOURSPACES[WORKING_SPACES[meta["color_space"]]]
            sun_rgb = float(a["sun_illuminance_lux"]) * np.asarray(a["sun_color"])
            gpu_rgb = rgb(gpu)[::4] * sun_rgb
            gpu_xyz = gpu_rgb @ np.asarray(cs.matrix_RGB_to_XYZ).T
            to_2020 = colour.matrix_RGB_to_RGB(cs, colour.RGB_COLOURSPACES["ITU-R BT.2020"])
            xyz_to_2020 = np.asarray(colour.RGB_COLOURSPACES["ITU-R BT.2020"].matrix_XYZ_to_RGB)
            de = delta_e_itp(gpu_rgb @ to_2020.T, ref_xyz @ xyz_to_2020.T)
            lum_err = (gpu_xyz[:, 1] - ref_xyz[:, 1]) / ref_xyz[:, 1]
            results.append({"check": f"V9 {set_name} C+A+B+D sz={sun_zenith:g}deg",
                            "mean_delta_e_itp": float(de.mean()), "max_delta_e_itp": float(de.max()),
                            "mean_luminance_rel_error": float(lum_err.mean()),
                            "max_abs_luminance_rel_error": float(np.abs(lum_err).max()),
                            "tolerance": None, "pass": True, "info": "recorded only (design.md V9)"})
    return results


# ---------------------------------------------------------------------- scene（地形ありの 2 成分の経路）
def check_scene(scene_dir: Path, label: str) -> list[dict]:
    results = []
    for r in vp4.check_aerial_perspective(scene_dir / "aerial_perspective", cases=SCENE_AP_CASES):
        # V3c の基準（RGB の透過率の方式の合計の相対 RMSE < 1%）は満たすかを記録し、合否は有限の値だけで取る
        r["check"] = r["check"].replace("V3c", f"V3c ({label})")
        r["v3c_criterion_met"] = r["pass"]
        r["pass"] = bool(np.isfinite(r["rgb_t_total_rel_rmse_rgb"]).all())
        r["info"] = "recorded (V3c metrics on the 2-component preset, design.md §5.9)"
        results.append(r)
    image_prefix = "phase4_9_" + label.replace("+", "_").lower()
    for r in vp45.check_ap_methods(scene_dir / "shadow_ap_methods", image_prefix=image_prefix):
        r["check"] = r["check"].replace("V3g", f"V3g ({label})")
        results.append(r)
    for r in vp4.check_sky_irradiance(scene_dir / "sky_irradiance"):
        r["check"] = r["check"].replace("V3d", f"V3d ({label})")
        r["pass"] = bool(np.isfinite(r["max_abs_total_rel_error"]))
        results.append(r)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe")
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase4_9")
    parser.add_argument("--pt", type=Path, default=ROOT / "dumps" / "v4b", help="validate_v4b.py のパストレースのダンプ")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    parser.add_argument("--only", nargs="*", default=None,
                        help="background V3i V3j V3k V3b V4b V9 scene のうち実行するもの")
    args = parser.parse_args()
    if not args.skip_run:
        run_app(args.exe, "aerosol_presets", args.out)
        run_app(args.exe, "aerosol_scene", args.out / "scene")
    coeff = json.loads((ROOT / "data" / "atmosphere_coefficients.json").read_text(encoding="utf-8"))
    run = (lambda k: args.only is None or k in args.only)  # noqa: E731
    results = []
    if run("background"):
        results += check_background(coeff)
    for combo in D_COMBOS:
        if run("V3i"):
            results.append(v48.check_transmittance(args.out, combo))
            results += v48.check_single(args.out, combo)
        if run("V3j"):
            results.append(v48.check_ms(args.out, combo))
        if run("V3k"):
            results += v48.check_sky(args.out, combo, coeff)
        if run("V3b"):
            results += v48.check_closeup(args.out, combo)
        print(f"  {v48.stem(combo)} done", flush=True)
    if run("V4b"):
        results += v48.check_v4b(args.out, args.pt, presets=("C_A_B_D",))
    if run("V9"):
        results += check_v9_d(args.out, coeff)
    if run("scene"):
        for label in ("C+A+B", "C+A+B+D"):  # C+A+B は同じ量の 1 成分（比較用）
            results += check_scene(args.out / "scene" / label.replace("+", "_"), label)
    tag = "" if args.only is None else "_" + "_".join(args.only)
    (args.out / f"report{tag}.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        keys = [k for k in r if k not in ("check", "pass", "tolerance", "info", "lut", "raymarch")]
        brief = ", ".join(f"{k}={r[k]:.3g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys[:3])
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<60} {brief[:160]}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
