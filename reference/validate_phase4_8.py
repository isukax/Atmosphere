"""Phase 4.8 の検証（docs/design.md §5.9、エアロゾルのプリセット・1 成分。基準は §12 Q20）。

    reference/.venv/Scripts/python.exe reference/validate_phase4_8.py [--skip-run]

アプリの aerosol_presets（dumps/phase4_8）を、同じプリセット（data/atmosphere_coefficients.json の float32 の値）の
Python と比べる。
  V3i: Transmittance LUT（500 区間）と単散乱の入れ子レイマーチ（GPU）vs 倍精度の求積（atmosphere_ref.py）。
       V1・V2 と同じ基準（相対誤差 < 1e-3、リファレンス自身の収束 < 1e-6）。
  V3j: Multi-scattering LUT（入れ子 256 区間・視線 256 区間）vs 倍精度の Ψ_ms。最大相対誤差 < 1%。
       既定の設定（20 区間・Transmittance LUT）の誤差も記録する。
  V3k: 全天（128×64）で、(1) LUT 経由の描画 vs 画面のレイマーチ（記録、V3 と同じ）、
       (2) GPU のレイマーチ vs Python の近似の描画（aerosol_experiment.py の integrate、Ψ_ms は Python で作り直したもの）。
       (2) は GT7 SDR の ΔE_ITP の平均 < 0.5（露出は Python の上半球の中央値 → fb 0.6）。
  V3b: 太陽の周り ±6°（低い太陽・夕方）の LUT 描画 vs レイマーチの光輪（記録）。
  V4b: 地表アルベド 0 の GPU の LUT 描画 vs Phase 4.7 のパストレースのダンプ（dumps/v4b、C と C+A+B）。記録のみ
       （既知の限界、§14 Phase 4.7）。ΔE_sys は validate_v4b.py と同じ（ノイズを差し引いた RMS）。
  V9: SpectralFit709 / 2020 の C+B・C+A+B・base+B・base+A+B の単散乱（GPU）vs 分光リファレンス（Mie 理論の σ(λ)・p(θ, λ) を 5 nm 刻みで
      使い、XYZ → 作業色空間）。記録のみ（今の V9 と同じ扱い）。
  MSsteps: Multi-scattering LUT の区間数・配置（20〜128、等間隔 / 2 次）ごとの誤差（GPU の検証用の LUT との比）と、
      Python の近似の描画の Ψ_ms だけを差し替えたときの ΔE_ITP（記録）。
  V3i には、区間数を増やした GPU の結果（Transmittance LUT 4000 区間、太陽方向 1024 区間）の誤差も記録する
  （AOD 0.1 で誤差が区間数によるものか、実装の誤りかを分けるため）。
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
import aerosol_experiment as ae  # noqa: E402
import aerosol_models as am  # noqa: E402
import derive_coefficients as dc  # noqa: E402
import spectral_data as sd  # noqa: E402
import validate_v4b as v4b  # noqa: E402
from atmosphere_ref import Atmosphere, sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from validate_phase1 import ROOT, delta_e_itp, run_app  # noqa: E402
from validate_phase3 import (WORKING_SPACES, closeup_directions, halo_statistics, path_trace_noise,  # noqa: E402
                             rel_rmse, rgb)

REL_TOLERANCE = 1e-3          # V1・V2 と同じ
REFERENCE_SELF_TOLERANCE = 1e-6
MS_REL_TOLERANCE = 1e-2       # §12 Q20
DELTA_E_TOLERANCE = 0.5       # §12 Q20（GPU のレイマーチ vs Python の近似）
SKY_SCENARIOS = ("day", "low_sun", "sunset", "day_10km", "altitude_3km")
CLOSEUP_SCENARIOS = ("low_sun", "sunset")
V4B_SCENARIOS = ("day", "low_sun", "sunset", "earth_shadow", "day_10km", "altitude_3km")
V4B_GATED = ("day", "low_sun", "sunset", "day_10km", "altitude_3km")
# Phase48Tests.cpp の kCombos と同じ。B を含まないもの（base+A・C・C+A）は色空間で Mie が変わらないので Hillaire2020 だけ
ALL_COMBOS = (("Hillaire2020", "C"), ("Hillaire2020", "C+A"), ("Hillaire2020", "C+B"), ("Hillaire2020", "C+A+B"),
              ("SpectralFit709", "C+B"), ("SpectralFit709", "C+A+B"),
              ("SpectralFit2020", "C+B"), ("SpectralFit2020", "C+A+B"),
              ("Hillaire2020", "base+A"), ("Hillaire2020", "base+B"), ("Hillaire2020", "base+A+B"),
              ("SpectralFit709", "base+B"), ("SpectralFit709", "base+A+B"),
              ("SpectralFit2020", "base+B"), ("SpectralFit2020", "base+A+B"))
# 倍精度の求積（V3i・V3j）は時間がかかるので、色空間で変わる分は C+A+B・base+A+B だけにする
REFERENCE_COMBOS = tuple(c for c in ALL_COMBOS if c[0] == "Hillaire2020" or c[1] in ("C+A+B", "base+A+B"))
V9_PRESETS = ("C+B", "C+A+B", "base+B", "base+A+B")
GROUND_ALBEDO = 0.3           # アプリの既定（Multi-scattering LUT の地表）


def stem(combo) -> str:
    return f"{combo[0]}_{combo[1].replace('+', '_')}"


def relative_error(gpu: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(gpu.astype(np.float64) - ref) / np.abs(ref)


# ---------------------------------------------------------------------- V3i
def check_transmittance(out: Path, combo) -> dict:
    lut, meta = read_exr(out / "transmittance" / f"{stem(combo)}.exr")
    r_mu, _ = read_exr(out / "transmittance" / "r_mu.exr")
    atm = Atmosphere.from_meta(meta)
    r = r_mu[..., 0].astype(np.float64).ravel()
    mu = r_mu[..., 1].astype(np.float64).ravel()
    ref = atm.transmittance_to_top(r, mu, panels=32, order=4)
    fine = atm.transmittance_to_top(r, mu, panels=64, order=4)
    self_err = float(np.max(np.abs(ref - fine) / fine))
    # [POINT] AOD 0.1 では地平線の下向きの光線の透過率が 1e-30 程度まで小さくなる。float32 の相対誤差の意味が
    #         なくなる値（< 1e-20）は除く（GPU の LUT は使われ方として、その値を引いても輝度に寄与しない）
    sig = np.all(ref > 1e-20, axis=1)
    err = relative_error(lut[..., :3].reshape(-1, 3), ref)[sig]
    # 区間数を 8 倍（4000）にした GPU の LUT の誤差（積分の区間数の誤差か、実装の誤りかを分ける）
    fine_path = out / "transmittance" / f"{stem(combo)}_steps4000.exr"
    err_fine = relative_error(read_exr(fine_path)[0][..., :3].reshape(-1, 3), ref)[sig] if fine_path.exists() else None
    return {"check": f"V3i transmittance LUT ({stem(combo)})", "max_rel_error": float(err.max()),
            "mean_rel_error": float(err.mean()), "texels_excluded_below_1e-20": int((~sig).sum()),
            "steps4000_max_rel_error_info": None if err_fine is None else float(err_fine.max()),
            "reference_self_error": self_err, "tolerance": REL_TOLERANCE,
            "pass": bool(err.max() < REL_TOLERANCE and self_err < REFERENCE_SELF_TOLERANCE)}


def check_single(out: Path, combo) -> list[dict]:
    results = []
    dirs, _ = read_exr(out / "single" / "directions.exr")
    d = dirs[..., :3].reshape(-1, 3).astype(np.float64)
    for altitude in (0.2, 10.0):
        for sun_zenith in (30.0, 60.0, 85.0):
            gpu, meta = read_exr(out / "single" / f"{stem(combo)}_h{altitude:g}_sz{sun_zenith:g}.exr")
            atm = Atmosphere.from_meta(meta)
            a = meta["atmosphere"]
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            camera = np.array([0.0, atm.bottom_radius_km + meta["camera_altitude_km"], 0.0])
            ref = atm.single_scattering_per_lux(camera, d, sun, outer_panels=32, inner_panels=8)
            subset = slice(0, None, 37)
            fine = atm.single_scattering_per_lux(camera, d[subset], sun, outer_panels=64, inner_panels=16)
            self_err = float(np.max(np.abs(ref[subset] - fine) / fine))
            err = relative_error(gpu[..., :3].reshape(-1, 3), ref)
            worst = int(np.unravel_index(np.argmax(err), err.shape)[0])
            # 太陽方向の区間数を 4 倍（1024）にした GPU の結果の誤差（収束の確認）
            fine_path = out / "single" / f"{stem(combo)}_h{altitude:g}_sz{sun_zenith:g}_nested1024.exr"
            err_fine = relative_error(read_exr(fine_path)[0][..., :3].reshape(-1, 3), ref) if fine_path.exists() else None
            results.append({"check": f"V3i single scattering ({stem(combo)}, h={altitude:g}km sz={sun_zenith:g}deg)",
                            "max_rel_error": float(err.max()), "mean_rel_error": float(err.mean()),
                            "nested1024_max_rel_error_info": None if err_fine is None else float(err_fine.max()),
                            "worst_direction": d[worst].tolist(), "reference_self_error": self_err,
                            "tolerance": REL_TOLERANCE,
                            "pass": bool(err.max() < REL_TOLERANCE and self_err < REFERENCE_SELF_TOLERANCE)})
    return results


# ---------------------------------------------------------------------- V3j
def check_ms(out: Path, combo) -> dict:
    r_mu, _ = read_exr(out / "ms" / "r_musun.exr")
    gpu_ref, meta = read_exr(out / "ms" / f"{stem(combo)}_reference.exr")
    gpu_def, _ = read_exr(out / "ms" / f"{stem(combo)}_default.exr")
    atm = Atmosphere.from_meta(meta)
    albedo = np.asarray(meta["atmosphere"]["ground_albedo"], dtype=np.float64)
    picks = [0, 4, 9, 13, 18, 22, 27, 31]
    ref, g_ref, g_def = [], [], []
    for y in picks:
        for x in picks:
            ref.append(atm.multiple_scattering_psi(float(r_mu[y, x, 0]), float(r_mu[y, x, 1]), albedo))
            g_ref.append(gpu_ref[y, x, :3])
            g_def.append(gpu_def[y, x, :3])
    ref = np.asarray(ref)

    def rel(gpu):  # 分母の下限は LUT の最大値の 1e-6（validate_phase3.py と同じ理由）
        return np.abs(np.asarray(gpu, dtype=np.float64) - ref) / np.maximum(ref, 1e-6 * ref.max())

    err, err_def = rel(g_ref), rel(g_def)
    significant = np.all(ref >= 1e-3 * ref.max(), axis=1)
    return {"check": f"V3j multi-scattering LUT ({stem(combo)})", "max_rel_error": float(err.max()),
            "mean_rel_error": float(err.mean()),
            "default_settings_max_rel_error_info": float(err_def.max()),
            "default_settings_mean_rel_error_significant_info": float(err_def[significant].mean()),
            "psi_max_per_lux": float(ref.max()), "tolerance": MS_REL_TOLERANCE,
            "pass": bool(err.max() < MS_REL_TOLERANCE)}


# ---------------------------------------------------------------------- Multi-scattering LUT の区間数（記録）
class GpuMultiScatteringTable:
    """GPU の Multi-scattering LUT を、GPU と同じ双線形で引く（テクセルの (r, μs) は r_musun.exr）。
    aerosol_experiment.integrate の Ψ_ms の代わりに使い、Ψ_ms 以外が同じ描画で LUT の違いの影響を見る。"""

    def __init__(self, path: Path, r_mu_path: Path):
        lut, _ = read_exr(path)
        r_mu, _ = read_exr(r_mu_path)
        self.psi = lut[..., :3].astype(np.float64)  # (y = r, x = μs)
        self.h = r_mu[:, 0, 0].astype(np.float64) - ae.BOTTOM_KM
        self.mu = r_mu[0, :, 1].astype(np.float64)

    def lookup(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        def frac(nodes, x):
            x = np.clip(x, nodes[0], nodes[-1])
            i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, nodes.size - 2)
            return i, ((x - nodes[i]) / (nodes[i + 1] - nodes[i]))[..., None]

        i, th = frac(self.h, r - ae.BOTTOM_KM)
        j, tm = frac(self.mu, mu)
        p = self.psi
        return (1 - th) * ((1 - tm) * p[i, j] + tm * p[i, j + 1]) + th * ((1 - tm) * p[i + 1, j] + tm * p[i + 1, j + 1])


def check_ms_steps(out: Path, coeff: dict) -> list[dict]:
    """区間数・配置ごとの Multi-scattering LUT（GPU）を、検証用の LUT（入れ子 256 区間、GPU）と比べる。
    見た目への影響は、Python の近似の描画の Ψ_ms だけを差し替えて ΔE_ITP で見る（全天、GT7 SDR）。"""

    results = []
    dirs_img, _ = read_exr(out / "sky" / "directions.exr")
    dirs = dirs_img[..., :3].reshape(-1, 3).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r_mu_path = out / "ms" / "r_musun.exr"
    for combo in (("Hillaire2020", "C"), ("Hillaire2020", "C+A+B")):
        atmo = python_atmosphere(coeff, combo)
        table = ae.TransmittanceTable(atmo)
        ref_path = out / "ms" / f"{stem(combo)}_reference.exr"
        ref_lut = read_exr(ref_path)[0][..., :3].astype(np.float64)
        significant = np.all(ref_lut >= 1e-3 * ref_lut.max(), axis=-1)
        ref_table = GpuMultiScatteringTable(ref_path, r_mu_path)
        renders = {}
        for scenario in SKY_SCENARIOS:
            _, meta = read_exr(out / "sky" / f"{stem(combo)}_{scenario}_lut.exr")
            a = meta["atmosphere"]
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            origin = np.array([0.0, a["bottom_radius_km"] + meta["camera_altitude_km"], 0.0])
            ref_img, _ = ae.integrate(atmo, table, ref_table, origin, dirs, sun, 256, None)
            renders[scenario] = (origin, sun, ref_img, ae.exposure_for(ref_img[dirs[:, 1] > 0.0]))
        for path in sorted((out / "ms_steps").glob(f"{stem(combo)}_lut*.exr")):
            lut, meta = read_exr(path)
            e = np.abs(lut[..., :3].astype(np.float64) - ref_lut) / np.maximum(ref_lut, 1e-6 * ref_lut.max())
            entry = {"check": f"MS steps {path.stem}", "max_rel_error_significant": float(e[significant].max()),
                     "mean_rel_error_significant": float(e[significant].mean()), "delta_e_mean": {},
                     "delta_e_mean_sky": {}, "tolerance": None, "pass": True, "info": "recorded only"}
            test_table = GpuMultiScatteringTable(path, r_mu_path)
            for scenario, (origin, sun, ref_img, expo) in renders.items():
                img, _ = ae.integrate(atmo, table, test_table, origin, dirs, sun, 256, None)
                de = delta_e_itp(ae.to_display(img, expo)[0], ae.to_display(ref_img, expo)[0])
                entry["delta_e_mean"][scenario] = float(de.mean())
                entry["delta_e_mean_sky"][scenario] = float(de[dirs[:, 1] > 0.0].mean())
            results.append(entry)
    return results


# ---------------------------------------------------------------------- V3k / V3b
def python_atmosphere(coeff: dict, combo) -> am.Atmosphere:
    s = next(x for x in coeff["sets"] if x["name"] == combo[0])
    return am.atmosphere_from_json(s, combo[1])


def check_sky(out: Path, combo, coeff: dict) -> list[dict]:
    atmo = python_atmosphere(coeff, combo)
    table = ae.TransmittanceTable(atmo)
    psi = ae.MultiScatteringTable(atmo, table, ground_albedo=GROUND_ALBEDO)
    dirs_img, _ = read_exr(out / "sky" / "directions.exr")
    dirs = dirs_img[..., :3].reshape(-1, 3).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    results = []
    for scenario in SKY_SCENARIOS:
        lut, meta = read_exr(out / "sky" / f"{stem(combo)}_{scenario}_lut.exr")
        ray, _ = read_exr(out / "sky" / f"{stem(combo)}_{scenario}_raymarch.exr")
        a = meta["atmosphere"]
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        origin = np.array([0.0, a["bottom_radius_km"] + meta["camera_altitude_km"], 0.0])
        approx, _ = ae.integrate(atmo, table, psi, origin, dirs, sun, 256, None)
        g_lut, g_ray = rgb(lut), rgb(ray)
        expo = ae.exposure_for(approx[dirs[:, 1] > 0.0])
        de = delta_e_itp(ae.to_display(g_ray, expo)[0], ae.to_display(approx, expo)[0])
        de_lut = delta_e_itp(ae.to_display(g_lut, expo)[0], ae.to_display(g_ray, expo)[0])
        sky = dirs[:, 1] > 0.0
        results.append({
            "check": f"V3k sky ({stem(combo)}, {scenario})",
            "raymarch_vs_python_delta_e_mean": float(de.mean()),
            "raymarch_vs_python_delta_e_p95": float(np.percentile(de, 95)),
            "raymarch_vs_python_delta_e_mean_sky": float(de[sky].mean()),
            "raymarch_vs_python_rel_rmse_rgb": rel_rmse(g_ray, approx).tolist(),
            "lut_vs_raymarch_rel_rmse_rgb_info": rel_rmse(g_lut, g_ray).tolist(),
            "lut_vs_raymarch_delta_e_mean_info": float(de_lut.mean()),
            "tolerance": {"raymarch_vs_python_delta_e_mean": DELTA_E_TOLERANCE},
            "pass": bool(de.mean() < DELTA_E_TOLERANCE)})
    return results


def check_closeup(out: Path, combo) -> list[dict]:
    results = []
    for scenario in CLOSEUP_SCENARIOS:
        lut, meta = read_exr(out / "closeup" / f"{stem(combo)}_{scenario}_lut.exr")
        ray, _ = read_exr(out / "closeup" / f"{stem(combo)}_{scenario}_raymarch.exr")
        a = meta["atmosphere"]
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        dirs = closeup_directions(meta, lut.shape[0])
        s_lut, s_ray = halo_statistics(lut, dirs, sun), halo_statistics(ray, dirs, sun)
        angle = np.degrees(np.arccos(np.clip(dirs @ sun, -1.0, 1.0)))
        rings = {f"{lo}-{hi}deg": float(lut[(angle >= lo) & (angle < hi), :3].mean()
                                        / ray[(angle >= lo) & (angle < hi), :3].mean())
                 for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 6))}
        results.append({"check": f"V3b sun aureole ({stem(combo)}, {scenario})",
                        "peak_ratio": float(s_lut["peak"] / s_ray["peak"]),
                        "sigma_ratio_h_v": [float(s_lut["sigma_h_deg"] / s_ray["sigma_h_deg"]),
                                            float(s_lut["sigma_v_deg"] / s_ray["sigma_v_deg"])],
                        "ring_mean_ratio_lut_over_raymarch": rings, "lut": s_lut, "raymarch": s_ray,
                        "tolerance": None, "pass": True, "info": "recorded only"})
    return results


# ---------------------------------------------------------------------- V4b
def check_v4b(out: Path, pt_dir: Path) -> list[dict]:
    previous = {}
    report = pt_dir / "report.json"
    if report.exists():
        for r in json.loads(report.read_text(encoding="utf-8")):
            if r["check"].startswith("V4b-2"):
                previous[r["check"]] = r["delta_e_itp_sdr_auto"]
    results = []
    for preset in ("C", "C_A_B"):
        for scenario in V4B_SCENARIOS:
            lut, meta = read_exr(out / "v4b" / f"{preset}_{scenario}_lut.exr")
            pt_stem = pt_dir / f"{preset}_{scenario}_pt"
            pt_img, _ = read_exr(pt_stem.with_name(pt_stem.name + ".exr"))
            dirs_img, _ = read_exr(pt_dir / f"{scenario}_directions.exr")
            dirs = dirs_img[..., :3].reshape(-1, 3).astype(np.float64)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            a = meta["atmosphere"]
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            pt, gpu = rgb(pt_img), rgb(lut)
            expo = v4b.scenario_exposure(pt, dirs)
            de = delta_e_itp(v4b.display_nits(gpu, expo), v4b.display_nits(pt, expo))
            acc_a, _ = read_exr(pt_stem.with_name(pt_stem.name + "_accA.exr"))
            acc_b, _ = read_exr(pt_stem.with_name(pt_stem.name + "_accB.exr"))
            half_a = rgb(acc_a) / acc_a[..., 3].reshape(-1, 1)
            half_b = rgb(acc_b) / acc_b[..., 3].reshape(-1, 1)
            de_noise = delta_e_itp(v4b.display_nits(half_a, expo), v4b.display_nits(half_b, expo))
            masks = v4b.hemisphere_masks(dirs, sun)
            sys_rms = {k: float(np.sqrt(max(np.mean(de[m] ** 2) - np.mean(de_noise[m] ** 2) / 4.0, 0.0)))
                       for k, m in masks.items()}
            prev = previous.get(f"V4b-2 approx vs PT ({preset}, {scenario})")
            results.append({"check": f"V4b GPU LUT vs PT ({preset}, {scenario})",
                            "delta_e_systematic_rms": sys_rms,
                            "python_approx_delta_e_systematic_rms_phase4_7":
                                {k: v["systematic_rms"] for k, v in prev.items()} if prev else None,
                            "mean_ratio_gpu_over_pt_rgb": (gpu.mean(axis=0) / pt.mean(axis=0)).tolist(),
                            "tolerance": None, "pass": True, "info": "recorded only (known limitation)"})
    return results


# ---------------------------------------------------------------------- V9
def spectral_mie(preset: str, wl: np.ndarray):
    """5 nm 刻みの Mie（混ぜた 1 成分）の σs(λ)・σt(λ)・位相関数の表 (U, L)（A を含むとき）または None（C+B・base+B）。
    量はプリセットの AOD550（base 系は base の量、C 系は 0.1）"""

    mie_wl = np.arange(360.0, 830.0 + dc.MIE_WAVELENGTH_STEP_NM / 2, dc.MIE_WAVELENGTH_STEP_NM)
    col = am.column_km(am.exponential_profile(am.BASE_MIE_SCALE_HEIGHT))
    aod550 = am.preset_aod550(preset)
    sigma_s = np.zeros(wl.size)
    sigma_t = np.zeros(wl.size)
    weighted = np.zeros((am.PHASE_TABLE_COUNT, wl.size))
    interp = dc.interpolation_matrix(wl, mie_wl)  # (L, 10 nm の節点)
    for mode, frac in ((am.FINE, am.FINE_FRACTION_OF_AOD), (am.COARSE, 1.0 - am.FINE_FRACTION_OF_AOD)):
        spec = dc.spectral_mode(mode, mie_wl)
        i550 = int(np.argmin(np.abs(mie_wl - 550.0)))
        ext550 = frac * aod550 / col
        s = ext550 * (interp @ spec.c_sca) / spec.c_ext[i550]
        sigma_s += s
        sigma_t += ext550 * (interp @ spec.c_ext) / spec.c_ext[i550]
        weighted += (interp @ spec.phase).T * s  # 位相関数も 10 nm の節点の線形補間
    phase = weighted / sigma_s if "A" in preset.split("+") else None
    return sigma_s, sigma_t, phase


def check_v9(out: Path) -> list[dict]:
    wl = np.arange(360.0, 830.0 + 1e-9, 5.0)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
        colour.SpectralShape(360.0, 830.0, 5.0)).values
    solar = sd.solar_irradiance(wl)
    dirs, _ = read_exr(out / "single" / "directions.exr")
    d = dirs[..., :3].reshape(-1, 3).astype(np.float64)[::4]  # 128 方向（今の V9 と同じ数）
    results = []
    for preset in V9_PRESETS:
        sigma_s, sigma_t, phase = spectral_mie(preset, wl)
        for sun_zenith in (30.0, 60.0, 85.0):
            ref_xyz = None
            for set_name in ("SpectralFit709", "SpectralFit2020"):
                gpu, meta = read_exr(out / "single" / f"{set_name}_{preset.replace('+', '_')}_h0.2_sz{sun_zenith:g}.exr")
                a = meta["atmosphere"]
                if ref_xyz is None:
                    atm = dataclasses.replace(Atmosphere.spectral(meta, wl), mie_scattering_per_km=sigma_s,
                                              mie_extinction_per_km=sigma_t, mie_phase_table=phase,
                                              mie_phase_g=am.BASE_G)
                    camera = np.array([0.0, atm.bottom_radius_km + float(meta["camera_altitude_km"]), 0.0])
                    sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
                    per_irradiance = atm.single_scattering_per_lux(camera, d, sun, outer_panels=16, inner_panels=4,
                                                                   chunk=2)
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
                results.append({"check": f"V9 {set_name} {preset} sz={sun_zenith:g}deg",
                                "mean_delta_e_itp": float(de.mean()), "max_delta_e_itp": float(de.max()),
                                "mean_luminance_rel_error": float(lum_err.mean()),
                                "max_abs_luminance_rel_error": float(np.abs(lum_err).max()),
                                "tolerance": None, "pass": True, "info": "recorded only (design.md V9)"})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe")
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase4_8")
    parser.add_argument("--pt", type=Path, default=ROOT / "dumps" / "v4b", help="Phase 4.7 のパストレースのダンプ")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    parser.add_argument("--only", nargs="*", default=None, help="V3i V3j V3k V3b V4b V9 のうち実行するもの")
    args = parser.parse_args()
    if not args.skip_run:
        run_app(args.exe, "aerosol_presets", args.out)
    coeff = json.loads((ROOT / "data" / "atmosphere_coefficients.json").read_text(encoding="utf-8"))
    run = (lambda k: args.only is None or k in args.only)  # noqa: E731
    results = []
    for combo in ALL_COMBOS:
        if combo in REFERENCE_COMBOS and run("V3i"):
            results.append(check_transmittance(args.out, combo))
            results += check_single(args.out, combo)
        if combo in REFERENCE_COMBOS and run("V3j"):
            results.append(check_ms(args.out, combo))
        if run("V3k"):
            results += check_sky(args.out, combo, coeff)
        if run("V3b"):
            results += check_closeup(args.out, combo)
        print(f"  {stem(combo)} done", flush=True)
    if run("MSsteps"):
        results += check_ms_steps(args.out, coeff)
    if run("V4b"):
        results += check_v4b(args.out, args.pt)
    if run("V9"):
        results += check_v9(args.out)
    tag = "" if args.only is None else "_" + "_".join(args.only)
    (args.out / f"report{tag}.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        keys = [k for k in r if k not in ("check", "pass", "tolerance", "info", "lut", "raymarch")]
        brief = ", ".join(f"{k}={r[k]:.3g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys[:3])
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<60} {brief}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
