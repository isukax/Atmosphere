"""V4b: 現実的なエアロゾルでの多重散乱の近似の検証（docs/design.md §5.8、§8.2 V4b）。

    reference/.venv/Scripts/python.exe reference/validate_v4b.py

  1. research.md §12 の変種（C / C+A+B / C+A+B+D）を表の媒質にして dumps/v4b/media に書き出す。
  2. アプリの pathtrace_aerosol で、表の媒質のパストレース（散乱 1 回 / 32 回）を描く。
  3. 同じ表の媒質で、Python の近似（Hillaire 2020 の多重散乱、aerosol_experiment.py）と単散乱を描いて比べる。
     V4b-1: 散乱 1 回のパストレース vs Python の単散乱。GPU が同じ媒質を見ているかの確認。
            判定は高度帯ごとの偏り（|偏り| < 3 × 標準誤差 + 0.5%。地球の影は単散乱がほぼ 0 なので V4a と同じ基準）。V4a と同じ基準（全画素の RMSE とノイズの比）も
            記録する。地面向きの短い光線は衝突が少なく、ノイズの裾が重い（画素ごとに ±30%）ので、RMSE の基準は
            偏りがなくても外れることがある（2026-10-17 の実行で 1 件）。
     V4b-2: Python の近似 vs パストレース（散乱 32 回）。V4 と同じ指標 + GT7 SDR・自動露出での ΔE_ITP。
            [POINT] パストレースのノイズ自体が ΔE で 1〜4 あるので、ΔE はノイズを差し引いた RMS で判定する:
              ΔE_sys = √(mean(ΔE²) − mean(ΔE_halves²)/4)（偶数・奇数フレームの平均の差の分散は全体の 4 倍）。
            基準（§12 Q18）: ΔE_sys < 3（太陽側・反対側の半球とも、地球の影を除く）、
            多重散乱成分の比（近似 / パストレース）が 0.7〜1.3（RGB 各）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerosol_experiment as ae  # noqa: E402
from atmosphere_ref import sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from pngio import write_png  # noqa: E402
from validate_phase1 import DEFAULT_EXE, ROOT, delta_e_itp, run_app  # noqa: E402
from validate_phase3 import closeup_directions, path_trace_noise, rel_rmse, rgb  # noqa: E402

VARIANTS = (("C", "C"), ("C+A+B", "C_A_B"), ("C+A+B+D", "C_A_B_D"))  # (実験の名前, ファイル名)
SCENARIOS = ("day", "low_sun", "sunset", "earth_shadow", "day_10km", "altitude_3km")
GATED = ("day", "low_sun", "sunset", "day_10km", "altitude_3km")
CLOSEUPS = ("low_sun", "sunset")
DELTA_E_TOLERANCE = 3.0
MS_RATIO_RANGE = (0.7, 1.3)
V4_ABS_RMSE_PER_LUX_TOLERANCE = 1e-2  # V4 と同じ（記録）
VIEW_STEPS = 128
ELEVATION_BANDS_DEG = ((-90, -30), (-30, -5), (-5, 0), (0, 5), (5, 30), (30, 90))


def build_media(out_dir: Path) -> dict[str, ae.Atmosphere]:
    """変種を表の媒質にして書き出し、Python 側で使う（同じ表の）大気を返す。"""

    coeff = json.loads((ROOT / "data" / "atmosphere_coefficients.json").read_text(encoding="utf-8"))
    hillaire = next(s for s in coeff["sets"] if s["name"] == "Hillaire2020")
    rayleigh = np.asarray(hillaire["rayleigh_scattering_per_km"])
    ozone = np.asarray(hillaire["ozone_absorption_per_km"])
    fine = ae.mode_optics(ae.FINE, ae.MEDIUM_PHASE_COUNT)
    coarse = ae.mode_optics(ae.COARSE, ae.MEDIUM_PHASE_COUNT)
    variants = ae.build_variants(rayleigh, ozone, fine, coarse)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    index = []
    atmospheres = {}
    for name, file_stem in VARIANTS:
        atmo = ae.tabulated(variants[name])
        table = ae.export_medium(atmo)
        (media_dir / f"{file_stem}.bin").write_bytes(table.astype("<f4").tobytes())
        index.append({"name": file_stem, "file": f"{file_stem}.bin", "floats": int(table.size), "variant": name,
                      "aod_rgb": atmo.aod().tolist()})
        atmospheres[file_stem] = atmo
    (media_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")
    return atmospheres


def display_nits(per_lux: np.ndarray, exposure: float) -> np.ndarray:
    return ae.to_display(per_lux, exposure)[0]


def scenario_exposure(pt: np.ndarray, dirs: np.ndarray) -> float:
    """パストレースの上半球の中央値の輝度が fb = 0.6 になる露出（近似とパストレースで同じ値を使う）"""

    return ae.exposure_for(pt[dirs[:, 1] > 0.0])


def hemisphere_masks(dirs: np.ndarray, sun: np.ndarray) -> dict[str, np.ndarray]:
    """太陽側・反対側の半球（水平面での方位で分ける）"""

    horizontal_sun = np.array([sun[0], 0.0, sun[2]])
    horizontal_sun /= max(np.linalg.norm(horizontal_sun), 1e-12)
    side = dirs @ horizontal_sun
    sky = dirs[:, 1] > 0.0
    # sky_*（地平線より上だけ）は記録用。判定は V4 と同じく全方向（sun_side / anti_sun）で行う
    return {"sun_side": side >= 0.0, "anti_sun": side < 0.0, "sky_sun_side": (side >= 0.0) & sky,
            "sky_anti_sun": (side < 0.0) & sky}


def heat(de: np.ndarray) -> np.ndarray:
    x = np.clip(de / 10.0, 0.0, 1.0)
    return np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 0.6, 0, 1), np.clip(3 * x - 1.8, 0, 1)], axis=-1)


def check(out_dir: Path, atmospheres: dict[str, ae.Atmosphere]) -> tuple[list[dict], dict]:
    results = []
    images = {}
    for file_stem, atmo in atmospheres.items():
        tt = ae.TransmittanceTable(atmo)
        psi = ae.MultiScatteringTable(atmo, tt, ground_albedo=0.0)  # V4 と同じく地表アルベド 0
        rows = []
        for scenario in SCENARIOS:
            stem = out_dir / f"{file_stem}_{scenario}"
            pt_img, meta = read_exr(stem.with_name(stem.name + "_pt.exr"))
            pt_single_img, _ = read_exr(stem.with_name(stem.name + "_pt_single.exr"))
            dirs_img, _ = read_exr(out_dir / f"{scenario}_directions.exr")
            h, w = pt_img.shape[:2]
            dirs = dirs_img[..., :3].reshape(-1, 3).astype(np.float64)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            a = meta["atmosphere"]
            # [POINT] Rayleigh・オゾンは GPU の大気（サイドカー）と同じ値であること（表の媒質に入れていない成分）
            assert np.allclose(a["rayleigh_scattering_per_km"], atmo.rayleigh, rtol=1e-6)
            assert np.allclose(a["ozone_absorption_per_km"], atmo.ozone, rtol=1e-6)
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            origin = np.array([0.0, a["bottom_radius_km"] + meta["camera_altitude_km"], 0.0])
            approx, _ = ae.integrate(atmo, tt, psi, origin, dirs, sun, VIEW_STEPS, None)
            single, _ = ae.integrate(atmo, tt, None, origin, dirs, sun, VIEW_STEPS, None)
            pt, pt_single = rgb(pt_img), rgb(pt_single_img)

            # V4b-1: 散乱 1 回のパストレース vs Python の単散乱（V4a と同じ基準）
            noise_s, _ = path_trace_noise(stem.with_name(stem.name + "_pt_single"))
            noise_rel_s = np.sqrt(np.mean(noise_s ** 2, axis=0)) / np.mean(single, axis=0)
            diff_rel = rel_rmse(pt_single, single)
            bias = np.mean(pt_single - single, axis=0) / np.mean(single, axis=0)
            bias_se = noise_rel_s / np.sqrt(single.shape[0])
            v4a_like = bool(np.all(diff_rel < 1.5 * noise_rel_s + 1e-3) and np.all(np.abs(bias) < 4.0 * bias_se + 1e-3))
            # 高度帯ごとの偏り（帯の中の画素の散らばりから標準誤差を見積もる）
            elevation = np.degrees(np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0)))
            bands = []
            ok1 = True
            for lo, hi in ELEVATION_BANDS_DEG:
                sel = (elevation >= lo) & (elevation < hi) & np.all(single > 0.0, axis=1)
                if sel.sum() < 16:
                    continue
                r = pt_single[sel] / single[sel] - 1.0
                b, se = r.mean(axis=0), r.std(axis=0) / np.sqrt(sel.sum())
                band_ok = bool(np.all(np.abs(b) < 3.0 * se + 5e-3))
                ok1 &= band_ok
                bands.append({"elevation_deg": [lo, hi], "pixels": int(sel.sum()), "bias_rgb": b.tolist(),
                              "standard_error_rgb": se.tolist(), "pass": band_ok})
            # [POINT] 地球の影では単散乱がほぼ 0 で、帯ごとの相対的な偏りに意味がない。V4a と同じ基準で判定する
            if scenario == "earth_shadow":
                ok1 = v4a_like
            results.append({"check": f"V4b-1 PT 1 bounce vs single ({file_stem}, {scenario})",
                            "rel_rmse_rgb": np.asarray(diff_rel).tolist(), "noise_rel_rmse_rgb": noise_rel_s.tolist(),
                            "bias_rgb": bias.tolist(), "bias_standard_error_rgb": bias_se.tolist(),
                            "v4a_criterion_met": v4a_like, "elevation_bands": bands,
                            "tolerance": "per elevation band |bias| < 3 SE + 0.5% (V4a criterion recorded)",
                            "pass": ok1})

            # V4b-2: 近似 vs パストレース（散乱 32 回）
            noise, _ = path_trace_noise(stem.with_name(stem.name + "_pt"))
            noise_abs = np.sqrt(np.mean(noise ** 2, axis=0))
            approx_abs = np.sqrt(np.maximum(np.mean((approx - pt) ** 2, axis=0) - noise_abs ** 2, 0.0))
            ms_ratio = (approx - single).sum(axis=0) / (pt - pt_single).sum(axis=0)
            ms_share = (pt - pt_single).sum(axis=0) / pt.sum(axis=0)
            expo = scenario_exposure(pt, dirs)
            d_pt, d_ap = display_nits(pt, expo), display_nits(approx, expo)
            de = delta_e_itp(d_ap, d_pt)
            # ノイズの目安: 偶数・奇数フレームの平均どうしの ΔE（全体の平均のノイズの約 2 倍）
            acc_a, _ = read_exr(stem.with_name(stem.name + "_pt_accA.exr"))
            acc_b, _ = read_exr(stem.with_name(stem.name + "_pt_accB.exr"))
            half_a = rgb(acc_a) / acc_a[..., 3].reshape(-1, 1)
            half_b = rgb(acc_b) / acc_b[..., 3].reshape(-1, 1)
            de_noise = delta_e_itp(display_nits(half_a, expo), display_nits(half_b, expo))
            halves = hemisphere_masks(dirs, sun)
            de_stats = {k: {"mean": float(de[m].mean()), "p95": float(np.percentile(de[m], 95)),
                            "max": float(de[m].max()), "rms": float(np.sqrt(np.mean(de[m] ** 2))),
                            "noise_rms": float(np.sqrt(np.mean(de_noise[m] ** 2)) / 2.0),
                            "systematic_rms": float(np.sqrt(max(np.mean(de[m] ** 2) - np.mean(de_noise[m] ** 2) / 4.0,
                                                                0.0)))}
                        for k, m in halves.items()}
            gated = scenario in GATED
            ok2 = bool(all(de_stats[k]["systematic_rms"] < DELTA_E_TOLERANCE for k in ("sun_side", "anti_sun"))
                       and np.all((ms_ratio > MS_RATIO_RANGE[0]) & (ms_ratio < MS_RATIO_RANGE[1])))
            results.append({
                "check": f"V4b-2 approx vs PT ({file_stem}, {scenario})",
                "delta_e_itp_sdr_auto": de_stats,
                "delta_e_itp_pt_noise_halves_mean": float(de_noise.mean()),
                "multiple_scattering_ratio_approx_over_pt_rgb": ms_ratio.tolist(),
                "multiple_scattering_share_of_pt_rgb": ms_share.tolist(),
                "approx_abs_rmse_per_lux_rgb": approx_abs.tolist(),
                "approx_rel_rmse_rgb": (approx_abs / np.mean(pt, axis=0)).tolist(),
                "approx_bias_rgb": (np.mean(approx - pt, axis=0) / np.mean(pt, axis=0)).tolist(),
                "v4_abs_rmse_criterion_met": bool(np.all(approx_abs < V4_ABS_RMSE_PER_LUX_TOLERANCE)),
                "path_trace_noise_rel_rmse_rgb": (noise_abs / np.mean(pt, axis=0)).tolist(),
                "samples_per_channel": meta.get("samples_per_channel"),
                "tolerance": {"delta_e_systematic_rms": DELTA_E_TOLERANCE, "ms_ratio": MS_RATIO_RANGE} if gated else None,
                "pass": ok2 if gated else True,
                "meets_criteria": ok2,
                "info": None if gated else "recorded only (earth shadow)",
            })
            # 画像: 近似 | パストレース | ΔE（正距円筒、2 倍に拡大）
            tile = lambda img: np.repeat(np.repeat(img.reshape(h, w, 3), 2, 0), 2, 1)  # noqa: E731
            rows.append(np.concatenate([tile(ae.to_display(approx, expo)[2]), np.ones((2 * h, 3, 3)),
                                        tile(ae.to_display(pt, expo)[2]), np.ones((2 * h, 3, 3)),
                                        tile(heat(de))], axis=1))
            rows.append(np.ones((3, rows[-1].shape[1], 3)))

            # 散乱回数の打ち切り（最大 64 回との差）
            pt64_path = stem.with_name(stem.name + "_pt64.exr")
            if pt64_path.exists():
                pt64 = rgb(read_exr(pt64_path)[0])
                de64 = delta_e_itp(display_nits(pt64, expo), d_pt)
                results.append({"check": f"V4b depth 32 vs 64 ({file_stem}, {scenario})",
                                "rel_mean_difference_rgb": (np.mean(pt64 - pt, axis=0) / np.mean(pt, axis=0)).tolist(),
                                "delta_e_itp_mean": float(de64.mean()), "delta_e_itp_max": float(de64.max()),
                                "tolerance": None, "pass": True, "info": "recorded only"})

            # 太陽の周り ±6°
            if scenario in CLOSEUPS:
                cu_img, cu_meta = read_exr(stem.with_name(stem.name + "_closeup_pt.exr"))
                n = cu_img.shape[0]
                cdirs = closeup_directions(cu_meta, n).reshape(-1, 3)
                cu_approx, _ = ae.integrate(atmo, tt, psi, origin, cdirs, sun, VIEW_STEPS, None)
                cu_pt = rgb(cu_img)
                # 太陽の周りは全天の露出だと白く飛ぶので、この画面のパストレースの中央値で露出を決める
                cu_expo = ae.exposure_for(cu_pt)
                de_cu = delta_e_itp(display_nits(cu_approx, cu_expo), display_nits(cu_pt, cu_expo))
                angle = np.degrees(np.arccos(np.clip(cdirs @ sun, -1.0, 1.0)))
                ring = {f"{lo}-{hi}deg": {"approx_over_pt_rgb": (cu_approx[(angle >= lo) & (angle < hi)].mean(axis=0)
                                                                / cu_pt[(angle >= lo) & (angle < hi)].mean(axis=0)).tolist()}
                        for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 6))}
                results.append({"check": f"V4b closeup approx vs PT ({file_stem}, {scenario})",
                                "delta_e_itp_mean": float(de_cu.mean()), "delta_e_itp_max": float(de_cu.max()),
                                "rings": ring, "tolerance": None, "pass": True, "info": "recorded only"})
            print(f"  {file_stem} {scenario}: ΔE_sys {de_stats['sun_side']['systematic_rms']:.2f}/"
                  f"{de_stats['anti_sun']['systematic_rms']:.2f}, "
                  f"MS ratio {' '.join(f'{x:.2f}' for x in ms_ratio)}")
        images[file_stem] = np.concatenate(rows[:-1])
    return results, images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "v4b")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    args = parser.parse_args()
    atmospheres = build_media(args.out)
    if not args.skip_run:
        run_app(args.exe, "pathtrace_aerosol", args.out)
    results, images = check(args.out, atmospheres)
    for name, img in images.items():
        write_png(ROOT / "docs" / "images" / f"v4b_{name}.png", img)
    (args.out / "report.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        if r["check"].startswith("V4b-2"):
            d = r["delta_e_itp_sdr_auto"]
            line = (f"ΔE_sys sun {d['sun_side']['systematic_rms']:.2f} anti {d['anti_sun']['systematic_rms']:.2f}"
                    f" (sky only {d['sky_sun_side']['systematic_rms']:.2f}/{d['sky_anti_sun']['systematic_rms']:.2f},"
                    f" noise {d['sun_side']['noise_rms']:.1f})"
                    f" | MS ratio {' '.join(f'{x:.2f}' for x in r['multiple_scattering_ratio_approx_over_pt_rgb'])}"
                    f" | MS share {' '.join(f'{x:.0%}' for x in r['multiple_scattering_share_of_pt_rgb'])}"
                    f" | rel RMSE {' '.join(f'{x:.1%}' for x in r['approx_rel_rmse_rgb'])}")
        elif r["check"].startswith("V4b-1"):
            worst = max(abs(x) for b in r["elevation_bands"] for x in b["bias_rgb"])
            line = ("rel RMSE " + " ".join(f"{x:.2%}" for x in r["rel_rmse_rgb"])
                    + " | noise " + " ".join(f"{x:.2%}" for x in r["noise_rel_rmse_rgb"])
                    + f" | V4a {'ok' if r['v4a_criterion_met'] else 'NG'} | worst band bias {worst:.2%}")
        elif "rings" in r:
            line = f"ΔE mean {r['delta_e_itp_mean']:.2f} max {r['delta_e_itp_max']:.1f} | " + " ".join(
                f"{k}: {np.mean(v['approx_over_pt_rgb']):.2f}" for k, v in r["rings"].items())
        else:
            line = f"ΔE mean {r['delta_e_itp_mean']:.3f} max {r['delta_e_itp_max']:.2f} | rel diff " + " ".join(
                f"{x:+.2%}" for x in r["rel_mean_difference_rgb"])
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<52} {line}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
