"""Phase 4.11 の検証（docs/design.md §5.11、多重散乱の SH 近似。基準は §12 Q28）。

    reference/.venv/Scripts/python.exe reference/validate_phase4_11.py [--skip-run] [--only CHECK ...]

アプリの multiscattering_sh（dumps/phase4_11）を Python と比べる。
  chi:     位相関数の Legendre のモーメント（サイドカー = C++ の PhaseMoments）vs Python（atmosphere_ref の
           legendre_moments。別の積分）。相対差 < 1e-5。
  V3j-SH:  SH の Multi-scattering LUT（入れ子 256 区間・視線 256 区間・等間隔）vs 倍精度（atmosphere_ref の
           multiple_scattering_sh、同じ 1,024 方向・同じ F の求積（サイドカーの layout.transfer）・同じブロックの解法）。係数ごとに |Δc_b| / |c00| < 1%
           （|c00| はテクセルの値。夜側で 0 に近い所は全体の最大の 1e-6 を下限にする）。
  MSsh:    既定の設定（40 区間・2 次配置・Transmittance LUT）と検証用の差、閉じ方の行列式の最小、F_00 の行内の
           ばらつき（F は μs によらないはずなので、方向の格子の求積の誤差の目安）。記録。F を行ごとの表（天頂角の
           1 次元の求積）にしてからは 0（太陽中心の格子で積んでいた最初の版は 11〜36%、dumps/phase4_11_grid）。
  V3k-SH:  全天（128×64）で、GPU の SH のレイマーチ vs 同じ SH の LUT を差し込んだ Python の描画（aerosol_experiment の
           integrate + ms_sh.GpuShMultiScatteringTable）。ΔE_ITP の平均 < 0.5（§5.9 の V3k と同じ）。これで描画側（局所の枠・
           基底の符号・χ・アトラスのタイル）を LUT と切り分けて確かめる。LUT 描画 vs レイマーチ、SH vs 等方の LUT 描画は記録。
  V4b-2:   地表アルベド 0 の GPU の LUT 描画（等方 / SH）vs パストレース（dumps/v4b、validate_v4b.py と同じ ΔE_sys）。
           判定は C+A+B+D の地平線より上（昼・低い太陽・夕方・高度 3 km で ΔE_sys < 3、太陽側・反対側とも）。
           ただし低い太陽の反対側は既知の限界（Python の試作でも 3.4）として記録する（§12 Q28）。
           全方向の値・多重散乱成分の比・10 km・地球の影・C と C+A+B は記録。
  Views:   地面を照らした画面（C+A+B+D、アルベド 0.3、今の媒質で撮り直したパストレース dumps/v4b_views_4_11）vs GPU の
           SH の LUT を差し込んだ Python の描画（compare_v4b_views.py --variants current sh gpu_sh --gpu-ms ... --results
           <out>/views の report.json を読む）。判定（§12 Q28）: 昼・高度 3 km は空・地面とも ΔE_sys < 3。低い太陽は < 4
           かつ等方の半分以下。夕方は空 < 4・地面 < 6、どちらも等方の半分以下。10 km は記録。Python の SH との差も記録。
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerosol_experiment as ae  # noqa: E402
import aerosol_models as am  # noqa: E402
import ms_sh  # noqa: E402
import validate_v4b as v4b  # noqa: E402
from atmosphere_ref import Atmosphere, sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from validate_phase1 import ROOT, delta_e_itp, run_app  # noqa: E402
from validate_phase3 import rel_rmse, rgb  # noqa: E402

DEFAULT_OUT = ROOT / "dumps" / "phase4_11"
RELEASE_EXE = ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe"
CHI_TOLERANCE = 1e-5
SH_TOLERANCE = 1e-2
SKY_DELTA_E_TOLERANCE = 3.0
V3K_DELTA_E_TOLERANCE = 0.5   # §12 Q20（GPU のレイマーチ vs Python の近似の描画）
SKY_SCENARIOS = ("day", "low_sun", "sunset", "day_10km", "altitude_3km")
SKY_PRESETS = ("C+A+B+D", "C+A+B")
TILE = 32
COEFFS = 6
# Phase411Tests.cpp の kShCombos と同じ（ファイル名の stem, 判定するか）
SH_COMBOS = (("Hillaire2020_base", True), ("Hillaire2020_base_g0.6", True), ("Hillaire2020_C", True),
             ("Hillaire2020_C_A_B", True), ("Hillaire2020_C_A_B_D", True), ("Hillaire2020_C_A_B_D_albedo0", True),
             ("Hillaire2020_C_A_B_D_aod0.3", False), ("Hillaire2020_C_A_B_D_aod1_albedo1", False))
# V3j-SH で比べるテクセル: 高度の行（0 = 地表、1 = 3.2 km、4 = 12.9 km、18 = 58 km）× μs の列
PICK_ROWS = (0, 1, 4, 18)
PICK_COLS = (2, 9, 14, 16, 22, 29)
V4B_SCENARIOS = ("day", "low_sun", "sunset", "earth_shadow", "day_10km", "altitude_3km")
V4B_GATED = ("day", "low_sun", "sunset", "altitude_3km")   # 低い高度・低い太陽・夕方を重視（§12 Q28）
V4B_PRESETS = ("C", "C_A_B", "C_A_B_D")
V4B_GATED_PRESET = "C_A_B_D"
# 既知の限界として記録だけにする（シナリオ, 半球）（§12 Q28）
V4B_KNOWN_LIMITS = (("low_sun", "sky_anti_sun"),)
# 地面を照らした画面の判定（§12 Q28）: シナリオ → {sky / ground: 上限}。half = 等方の近似の半分以下も求める
VIEW_LIMITS = {"day": ({"sky": 3.0, "ground": 3.0}, False), "altitude_3km": ({"sky": 3.0, "ground": 3.0}, False),
               "low_sun": ({"sky": 4.0, "ground": 4.0}, True), "sunset": ({"sky": 4.0, "ground": 6.0}, True)}


def read_atlas(path: Path) -> np.ndarray:
    """192×32 のアトラス → (行, 列, 係数, RGB)"""

    img, _ = read_exr(path)
    return img[..., :3].astype(np.float64).reshape(TILE, COEFFS, TILE, 3).transpose(0, 2, 1, 3)


def atmosphere_and_moments(meta: dict) -> tuple[Atmosphere, np.ndarray]:
    atmo = Atmosphere.from_meta(meta)
    count = 2 if atmo.has_mie1 else 1
    moments = np.stack([np.broadcast_to(atmo.legendre_moments(k), (3, 3)) for k in range(count)])
    return atmo, moments


# ---------------------------------------------------------------------- χ
def check_chi(out: Path) -> list[dict]:
    results = []
    for stem, _ in SH_COMBOS:
        _, meta = read_exr(out / "ms_sh" / f"{stem}_sh_default.exr")
        atmo, moments = atmosphere_and_moments(meta)
        gpu = np.asarray(meta["atmosphere"]["mie_phase_legendre_moments"], dtype=np.float64)  # (成分, l, RGB)
        rel = np.abs(gpu - moments) / np.maximum(np.abs(moments), 1e-6)
        results.append({"check": f"chi ({stem})", "max_rel_diff": float(rel.max()),
                        "chi_python": moments.tolist(), "tolerance": CHI_TOLERANCE,
                        "pass": bool(rel.max() < CHI_TOLERANCE)})
    return results


# ---------------------------------------------------------------------- V3j-SH
def transfer_method(meta: dict) -> tuple[str, tuple[int, int]]:
    """GPU の F の求積（サイドカーの layout.transfer。ない古いダンプは太陽中心の格子）"""

    t = meta.get("layout", {}).get("transfer", {})
    return t.get("method", "grid"), tuple(t.get("rays", (64, 64)))


def _reference_texel(args):
    meta, r_km, mu_sun, panels = args
    atmo, moments = atmosphere_and_moments(meta)
    albedo = np.asarray(meta["atmosphere"]["ground_albedo"], dtype=np.float64)
    method, rays = transfer_method(meta)
    return atmo.multiple_scattering_sh(r_km, mu_sun, albedo, moments, outer_panels=panels[0], inner_panels=panels[1],
                                       transfer=method, zenith_rays=rays)


def check_v3j_sh(out: Path, workers: int) -> list[dict]:
    r_mu, _ = read_exr(out / "ms_sh" / "r_musun.exr")
    jobs = []
    for stem, _ in SH_COMBOS:
        _, meta = read_exr(out / "ms_sh" / f"{stem}_sh_reference.exr")
        for y in PICK_ROWS:
            for x in PICK_COLS:
                jobs.append((stem, y, x, (meta, float(r_mu[y, x, 0]), float(r_mu[y, x, 1]), (16, 8))))
    # Python 自身の収束: 区間数を倍にした結果との差（地表・低い太陽のテクセル、C+A+B+D）
    _, meta_t = read_exr(out / "ms_sh" / "Hillaire2020_C_A_B_D_sh_reference.exr")
    conv_args = [(meta_t, float(r_mu[0, 16, 0]), float(r_mu[0, 16, 1]), p) for p in ((16, 8), (32, 16))]
    with ProcessPoolExecutor(workers) as pool:
        values = list(pool.map(_reference_texel, [j[3] for j in jobs] + conv_args, chunksize=1))
    conv = np.abs(values[-1] - values[-2]).max() / np.abs(values[-1][0]).max()
    values = values[:-2]
    results = [{"check": "V3j-SH Python self-convergence (16/8 vs 32/16 panels, C+A+B+D ground)",
                "max_rel_diff_over_c00": float(conv), "tolerance": None, "pass": True, "info": "recorded"}]
    by_stem: dict[str, list] = {}
    for (stem, y, x, _), ref in zip(jobs, values):
        by_stem.setdefault(stem, []).append((y, x, ref))
    for stem, gated in SH_COMBOS:
        gpu_ref = read_atlas(out / "ms_sh" / f"{stem}_sh_reference.exr")
        gpu_def = read_atlas(out / "ms_sh" / f"{stem}_sh_default.exr")
        refs = by_stem[stem]
        c00_max = max(np.abs(r[0]).max() for _, _, r in refs)
        worst = {"reference": 0.0, "default": 0.0}
        worst_at = {}
        for y, x, ref in refs:
            denom = np.maximum(np.abs(ref[0]), 1e-6 * c00_max)                 # (C,)
            for name, gpu in (("reference", gpu_ref), ("default", gpu_def)):
                err = float((np.abs(gpu[y, x] - ref) / denom).max())
                if err > worst[name]:
                    worst[name], worst_at[name] = err, [y, x]
        results.append({"check": f"V3j-SH SH LUT vs double ({stem})",
                        "max_rel_error_over_c00": worst["reference"], "at_row_col": worst_at.get("reference"),
                        "default_settings_max_rel_error_info": worst["default"],
                        "default_at_row_col": worst_at.get("default"),
                        "tolerance": SH_TOLERANCE if gated else None,
                        "pass": bool(worst["reference"] < SH_TOLERANCE) if gated else True,
                        "info": None if gated else "recorded only"})
    return results


def check_ms_debug(out: Path) -> list[dict]:
    results = []
    for stem, _ in SH_COMBOS:
        dbg, _ = read_exr(out / "ms_sh" / f"{stem}_sh_debug.exr")
        det, f00 = dbg[..., 2], dbg[..., 3]
        lit = f00 > 1e-6
        spread = float(np.max([(f00[y, lit[y]].max() - f00[y, lit[y]].min()) / f00[y, lit[y]].mean()
                               for y in range(TILE) if lit[y].any()]))
        results.append({"check": f"MSsh closure ({stem})", "min_det": float(det.min()),
                        "f00_max_relative_spread_in_row": spread, "f00_ground_row_mean": float(f00[0].mean()),
                        "tolerance": None, "pass": bool(np.isfinite(det).all() and det.min() > 0.0),
                        "info": "recorded (pass = finite and non-singular)"})
    return results


# ---------------------------------------------------------------------- V3k-SH
def check_v3k_sh(out: Path) -> list[dict]:
    coeff = json.loads((ROOT / "data" / "atmosphere_coefficients.json").read_text(encoding="utf-8"))
    hillaire = next(s for s in coeff["sets"] if s["name"] == "Hillaire2020")
    dirs_img, _ = read_exr(out / "sky_sh" / "directions.exr")
    dirs = dirs_img[..., :3].reshape(-1, 3).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sky = dirs[:, 1] > 0.0
    results = []
    for preset in SKY_PRESETS:
        stem = "Hillaire2020_" + preset.replace("+", "_")
        atmo = am.atmosphere_from_json(hillaire, preset)
        table = ae.TransmittanceTable(atmo)
        # GPU の既定の SH の LUT（地表アルベド 0.3。sky_sh の描画と同じ設定）を、GPU と同じ双線形で引く
        psi = ms_sh.GpuShMultiScatteringTable(out / "ms_sh" / f"{stem}_sh_default.exr", ms_sh.medium_moments(atmo))
        for scenario in SKY_SCENARIOS:
            ray, meta = read_exr(out / "sky_sh" / f"{stem}_{scenario}_sh_raymarch.exr")
            lut, _ = read_exr(out / "sky_sh" / f"{stem}_{scenario}_sh_lut.exr")
            iso, _ = read_exr(out / "sky_sh" / f"{stem}_{scenario}_iso_lut.exr")
            a = meta["atmosphere"]
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            origin = np.array([0.0, a["bottom_radius_km"] + meta["camera_altitude_km"], 0.0])
            approx, _ = ae.integrate(atmo, table, psi, origin, dirs, sun, 256, None)
            g_ray, g_lut, g_iso = rgb(ray), rgb(lut), rgb(iso)
            expo = ae.exposure_for(approx[sky])
            to_nits = lambda x: ae.to_display(x, expo)[0]  # noqa: E731
            de = delta_e_itp(to_nits(g_ray), to_nits(approx))
            de_lut = delta_e_itp(to_nits(g_lut), to_nits(g_ray))
            de_iso = delta_e_itp(to_nits(g_iso), to_nits(g_lut))
            results.append({
                "check": f"V3k-SH sky ({stem}, {scenario})",
                "raymarch_vs_python_delta_e_mean": float(de.mean()),
                "raymarch_vs_python_delta_e_p95": float(np.percentile(de, 95)),
                "raymarch_vs_python_rel_rmse_rgb": rel_rmse(g_ray, approx).tolist(),
                "lut_vs_raymarch_delta_e_mean_info": float(de_lut.mean()),
                "lut_vs_raymarch_rel_rmse_rgb_info": rel_rmse(g_lut, g_ray).tolist(),
                "sh_vs_iso_lut_delta_e_mean_sky_info": float(de_iso[sky].mean()),
                "tolerance": {"raymarch_vs_python_delta_e_mean": V3K_DELTA_E_TOLERANCE},
                "pass": bool(de.mean() < V3K_DELTA_E_TOLERANCE)})
    return results


# ---------------------------------------------------------------------- V4b-2（GPU の LUT 描画）
def check_v4b(out: Path, pt_dir: Path) -> list[dict]:
    results = []
    for preset in V4B_PRESETS:
        for scenario in V4B_SCENARIOS:
            pt_stem = pt_dir / f"{preset}_{scenario}_pt"
            pt_img, _ = read_exr(pt_stem.with_name(pt_stem.name + ".exr"))
            pt_single_img, _ = read_exr(pt_stem.with_name(pt_stem.name + "_single.exr"))
            dirs_img, _ = read_exr(pt_dir / f"{scenario}_directions.exr")
            dirs = dirs_img[..., :3].reshape(-1, 3).astype(np.float64)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            pt, pt_single = rgb(pt_img), rgb(pt_single_img)
            expo = v4b.scenario_exposure(pt, dirs)
            acc_a, _ = read_exr(pt_stem.with_name(pt_stem.name + "_accA.exr"))
            acc_b, _ = read_exr(pt_stem.with_name(pt_stem.name + "_accB.exr"))
            half_a = rgb(acc_a) / acc_a[..., 3].reshape(-1, 1)
            half_b = rgb(acc_b) / acc_b[..., 3].reshape(-1, 1)
            de_noise = delta_e_itp(v4b.display_nits(half_a, expo), v4b.display_nits(half_b, expo))
            entry = {"check": f"V4b-2 GPU LUT vs PT ({preset}, {scenario})"}
            ok = True
            for model in ("iso", "sh"):
                gpu_img, meta = read_exr(out / "v4b_sh" / f"{preset}_{scenario}_lut_{model}.exr")
                a = meta["atmosphere"]
                sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
                gpu = rgb(gpu_img)
                de = delta_e_itp(v4b.display_nits(gpu, expo), v4b.display_nits(pt, expo))
                masks = v4b.hemisphere_masks(dirs, sun)
                sys_rms = {k: float(np.sqrt(max(np.mean(de[m] ** 2) - np.mean(de_noise[m] ** 2) / 4.0, 0.0)))
                           for k, m in masks.items()}
                # 多重散乱成分の比（GPU の単散乱はパストレースの散乱 1 回と同じとみなす。V4b-1 で確認済み）
                ms_ratio = (gpu - pt_single).sum(axis=0) / (pt - pt_single).sum(axis=0)
                entry[model] = {"delta_e_systematic_rms": sys_rms, "ms_ratio_rgb": ms_ratio.tolist(),
                                "mean_ratio_gpu_over_pt_rgb": (gpu.mean(axis=0) / pt.mean(axis=0)).tolist()}
                if model == "sh" and preset == V4B_GATED_PRESET and scenario in V4B_GATED:
                    ok = all(sys_rms[k] < SKY_DELTA_E_TOLERANCE for k in ("sky_sun_side", "sky_anti_sun")
                             if (scenario, k) not in V4B_KNOWN_LIMITS)
                    known = [k for k in ("sky_sun_side", "sky_anti_sun") if (scenario, k) in V4B_KNOWN_LIMITS]
                    if known:
                        entry["known_limitation"] = {k: sys_rms[k] for k in known}
            gated = preset == V4B_GATED_PRESET and scenario in V4B_GATED
            entry.update({"tolerance": {"sky_delta_e_systematic_rms": SKY_DELTA_E_TOLERANCE} if gated else None,
                          "pass": ok if gated else True, "info": None if gated else "recorded only"})
            results.append(entry)
    return results


# ---------------------------------------------------------------------- 地面を照らした画面
def check_views(out: Path) -> list[dict]:
    report = json.loads((out / "views" / "report.json").read_text(encoding="utf-8"))
    table = {(e["scenario"], e["side"], e["variant"]): e for e in report}
    results = []
    for scenario in ("day", "low_sun", "sunset", "day_10km", "altitude_3km"):
        for side in ("front", "back"):
            gpu, iso = table[(scenario, side, "gpu_sh")], table[(scenario, side, "current")]
            py = table.get((scenario, side, "sh"))
            entry = {"check": f"Views lit ground ({scenario}, {side})"}
            ok = True
            limits = VIEW_LIMITS.get(scenario)
            for region in ("sky", "ground"):
                g = gpu[region]["delta_e_systematic_rms"]
                i = iso[region]["delta_e_systematic_rms"]
                entry[region] = {"gpu_sh": g, "isotropic": i,
                                 "python_sh": py[region]["delta_e_systematic_rms"] if py else None,
                                 "mean_ratio_gpu_sh_rgb": gpu[region]["mean_ratio_approx_over_pt_rgb"]}
                if limits is not None:
                    ok &= g < limits[0][region] and (not limits[1] or g <= 0.5 * i)
            entry.update({"tolerance": ({"limits": limits[0], "at_most_half_of_isotropic": limits[1]}
                                        if limits else None),
                          "pass": bool(ok), "info": None if limits else "recorded only"})
            results.append(entry)
    return results


def print_line(r: dict) -> str:
    if r["check"].startswith("Views"):
        return " | ".join(f"{k} SH {r[k]['gpu_sh']:.1f} (Python {r[k]['python_sh']:.1f}, iso {r[k]['isotropic']:.1f})"
                          for k in ("sky", "ground"))
    if r["check"].startswith("V4b-2"):
        parts = []
        for model in ("iso", "sh"):
            d = r[model]["delta_e_systematic_rms"]
            parts.append(f"{model} sky {d['sky_sun_side']:.1f}/{d['sky_anti_sun']:.1f} all {d['sun_side']:.1f}/"
                         f"{d['anti_sun']:.1f} MS {' '.join(f'{x:.2f}' for x in r[model]['ms_ratio_rgb'])}")
        return " | ".join(parts)
    if r["check"].startswith("V3k-SH"):
        return (f"raymarch vs Python dE mean {r['raymarch_vs_python_delta_e_mean']:.3f} "
                f"p95 {r['raymarch_vs_python_delta_e_p95']:.2f} | LUT vs raymarch "
                f"{r['lut_vs_raymarch_delta_e_mean_info']:.3f} | SH vs iso (sky) {r['sh_vs_iso_lut_delta_e_mean_sky_info']:.2f}")
    if r["check"].startswith("V3j-SH SH"):
        return (f"max {r['max_rel_error_over_c00']:.2e} at {r['at_row_col']} | default "
                f"{r['default_settings_max_rel_error_info']:.2e}")
    if r["check"].startswith("chi"):
        return f"max rel {r['max_rel_diff']:.2e}"
    if r["check"].startswith("MSsh"):
        return (f"min det {r['min_det']:.3f} | F00 spread in row {r['f00_max_relative_spread_in_row']:.1%} | "
                f"ground F00 {r['f00_ground_row_mean']:.3f}")
    return json.dumps({k: v for k, v in r.items() if k not in ("check", "pass")})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=RELEASE_EXE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pt", type=Path, default=ROOT / "dumps" / "v4b")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="chi V3j-SH MSsh V3k-SH V4b Views")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if not args.skip_run:
        run_app(args.exe, "multiscattering_sh", args.out)
    checks = {"chi": lambda: check_chi(args.out), "V3j-SH": lambda: check_v3j_sh(args.out, args.workers),
              "MSsh": lambda: check_ms_debug(args.out), "V3k-SH": lambda: check_v3k_sh(args.out),
              "V4b": lambda: check_v4b(args.out, args.pt), "Views": lambda: check_views(args.out)}
    results = []
    for name, fn in checks.items():
        if args.only is None or name in args.only:
            results += fn()
    report = args.out / ("report.json" if args.only is None else f"report_{'_'.join(args.only)}.json")
    report.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<58} {print_line(r)}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
