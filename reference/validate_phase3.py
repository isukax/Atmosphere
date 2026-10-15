"""Phase 3 の数値検証（docs/design.md §8.2 V3 / V4 / V9 と Multi-scattering LUT の照合）。

    reference/.venv/Scripts/python.exe reference/validate_phase3.py

  MS : Multi-scattering LUT（GPU、太陽方向は入れ子 256 区間・視線 256 区間）vs 倍精度の同じ式（64 方向）
       8×8 テクセルで相対誤差 < 1e-3。実用設定（20 区間・Transmittance LUT）の誤差は参考値。
  V3 : Sky-View LUT 経由の描画 vs ピクセルごとのレイマーチ（同じ Transmittance / Multi-scattering LUT）
       = Sky-View LUT の解像度・パラメタ化・補間・30 区間による誤差。記録のみ。
  V4a: パストレース（散乱 1 回）vs 単散乱の入れ子レイマーチ（Phase 2 で検証済み）
       = パストレーサー自体の正しさ。差がモンテカルロのノイズで説明できること。
  V4 : LUT 描画・レイマーチ vs パストレース（散乱 32 回）= 多重散乱の近似（等方仮定）の誤差。
       昼・夕方・高度 10 km で、単位照度あたりの絶対 RMSE（ノイズ補正後）が 1e-2 未満（論文の値
       (0.94, 1.74, 5.07)e-3 と同じオーダー、design.md V4）。地球の影のシナリオは記録のみ。
       相対 RMSE と、多重散乱の成分だけの比（近似 / パストレース）も記録する。
  V9 : 係数セットごとの単散乱（GPU、作業色空間の RGB）vs 分光計算（5 nm、XYZ → 同じ作業色空間）
       ΔE_ITP と輝度の相対誤差を記録する（合否はなし）。
  V3b: 太陽の周りの光輪（画角 6°）の LUT 描画 vs レイマーチ。g = 0.8 と、g = 0.99 で太陽天頂角 34〜36°。
       ピーク値の比が 0.95〜1.05、光輪の広がり（標準偏差）の比が 0.95〜1.05、中心のずれ < 0.05°。
       Mie の位相関数を LUT の外で掛ける変更（design.md §14 の調査）の回帰検出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import colour
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spectral_data as sd  # noqa: E402
from atmosphere_ref import Atmosphere, sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from validate_phase1 import DEFAULT_EXE, ROOT, delta_e_itp, run_app  # noqa: E402

MS_REL_TOLERANCE = 1e-3
V4_ABS_RMSE_PER_LUX_TOLERANCE = 1e-2
SCENARIOS = ("day", "sunset", "earth_shadow", "day_10km")
V4_GATED = ("day", "sunset", "day_10km")
WORKING_SPACES = {"Rec709": "ITU-R BT.709", "Rec2020": "ITU-R BT.2020"}


def rgb(image: np.ndarray) -> np.ndarray:
    return image[..., :3].astype(np.float64).reshape(-1, 3)


def rel_rmse(test: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """チャンネルごとの RMSE / 平均（相対 RMSE）"""

    return np.sqrt(np.mean((test - ref) ** 2, axis=0)) / np.mean(ref, axis=0)


# ---------------------------------------------------------------------- MS
def check_multiscattering(out_dir: Path) -> list[dict]:
    r_mu, meta = read_exr(out_dir / "ms_r_musun.exr")
    reference_gpu, _ = read_exr(out_dir / "ms_reference.exr")
    default_gpu, _ = read_exr(out_dir / "ms_default.exr")
    atm = Atmosphere.from_meta(meta)
    albedo = np.asarray(meta["atmosphere"]["ground_albedo"], dtype=np.float64)
    picks = [0, 4, 9, 13, 18, 22, 27, 31]
    ref, gpu_ref, gpu_def = [], [], []
    for y in picks:
        for x in picks:
            r_km, mu_sun = float(r_mu[y, x, 0]), float(r_mu[y, x, 1])
            ref.append(atm.multiple_scattering_psi(r_km, mu_sun, albedo))
            gpu_ref.append(reference_gpu[y, x, :3])
            gpu_def.append(default_gpu[y, x, :3])
    ref = np.asarray(ref)

    def rel(gpu):
        # [POINT] 相対誤差の分母の下限を「LUT の最大値の 1e-6」にする。地球の影の境界付近（μs ≈ −0.4）では
        #         Ψ_ms が最大値の 1e-9 程度まで小さくなり、そこでは float32 の積算誤差で相対誤差が 50% を超えるが、
        #         絶対値としては無意味（太陽が真下の点では Ψ_ms = 0 で、0/0 にもなる）。
        gpu = np.asarray(gpu, dtype=np.float64)
        return np.abs(gpu - ref) / np.maximum(ref, 1e-6 * ref.max())

    err = rel(gpu_ref)
    err_def = rel(gpu_def)
    # リファレンス自身の収束（太陽が地平線の上にある 1 テクセルで区間数を倍に。誤差の分母の下限は rel と同じ）
    i = 5
    y, x = picks[i], picks[i]
    fine = atm.multiple_scattering_psi(float(r_mu[y, x, 0]), float(r_mu[y, x, 1]), albedo, outer_panels=64,
                                       inner_panels=16)
    self_err = float(np.max(np.abs(ref[i * len(picks) + i] - fine) / np.maximum(fine, 1e-6 * ref.max())))
    # 区間の配置（2 次 vs 等間隔）と区間数による誤差（記録のみ）。
    # 有意な値（LUT の最大値の 1e-3 以上）のテクセルだけで平均・RMSE を取り、影の境界の極小値に引きずられないようにする。
    significant = np.all(ref >= 1e-3 * ref.max(), axis=1)
    mu_of_pick = np.array([float(r_mu[y, x, 1]) for y in picks for x in picks])
    distribution_results = []
    for path in sorted(out_dir.glob("ms_lut*_*.exr")):
        img, meta_v = read_exr(path)
        gpu = np.array([img[y, x, :3] for y in picks for x in picks])
        e = rel(gpu)
        signed = (np.asarray(gpu, dtype=np.float64) - ref) / np.maximum(ref, 1e-6 * ref.max())
        distribution_results.append({
            "check": f"MS step distribution {meta_v['step_distribution']} {meta_v['step_count']} steps",
            "max_rel_error": float(e.max()),
            "mean_rel_error_significant": float(e[significant].mean()),
            "mean_signed_rel_error_significant_rgb": signed[significant].mean(axis=0).tolist(),
            "max_rel_error_sun_above_horizon": float(e[mu_of_pick > 0.2].max()),
            "max_rel_error_near_terminator": float(e[np.abs(mu_of_pick) <= 0.2].max()),
            "tolerance": None,
            "pass": True,
            "info": "recorded only (step distribution comparison)",
        })

    return distribution_results + [{
        "check": "MS Multi-scattering LUT (256 steps, nested)",
        "max_rel_error": float(err.max()),
        "mean_rel_error": float(err.mean()),
        "default_settings_max_rel_error_info": float(err_def.max()),
        "default_settings_mean_rel_error_info": float(err_def.mean()),
        "psi_range_per_lux": [float(ref.min()), float(ref.max())],
        "reference_self_error": self_err,
        "tolerance": MS_REL_TOLERANCE,
        "pass": bool(err.max() < MS_REL_TOLERANCE and self_err < 1e-6),
    }]


# ---------------------------------------------------------------------- V3
def check_sky_lut(out_dir: Path) -> list[dict]:
    results = []
    for name in SCENARIOS:
        lut, meta = read_exr(out_dir / f"{name}_lut.exr")
        ref, _ = read_exr(out_dir / f"{name}_raymarch.exr")
        dirs, _ = read_exr(out_dir / f"{name}_directions.exr")
        elevation = np.degrees(np.arcsin(np.clip(dirs[..., 1].astype(np.float64), -1.0, 1.0)))
        a, b = rgb(lut), rgb(ref)
        rel = np.abs(a - b) / np.maximum(b, 1e-12)
        horizon = np.abs(elevation.ravel()) < 5.0
        worst = int(np.argmax(rel.max(axis=1)))
        # 太陽の方位（φ ≈ 0、x = 0 の列）での仰角プロファイル
        profile = {
            "elevation_deg": elevation[:, 0].tolist(),
            "lut": lut[:, 0, :3].tolist(),
            "raymarch": ref[:, 0, :3].tolist(),
        }
        results.append({
            "check": f"V3 sky-view LUT vs raymarch ({name})",
            "rel_rmse_rgb": rel_rmse(a, b).tolist(),
            "max_rel_error": float(rel.max()),
            "max_rel_error_elevation_deg": float(elevation.ravel()[worst]),
            "horizon_band_max_rel_error": float(rel[horizon].max()),
            "sun_zenith_deg": meta["atmosphere"]["sun_zenith_deg"],
            "profile_sun_azimuth": profile,
            "tolerance": None,
            "pass": True,
            "info": "recorded only (design.md V3)",
        })
    return results


# ---------------------------------------------------------------------- V4
def path_trace_noise(stem: Path) -> tuple[np.ndarray, np.ndarray]:
    """偶数・奇数フレームの平均の差から、パストレースの平均のノイズ（ピクセルごとの標準偏差の推定）を返す。"""

    acc_a, _ = read_exr(stem.with_name(stem.name + "_accA.exr"))
    acc_b, _ = read_exr(stem.with_name(stem.name + "_accB.exr"))
    mean_a = rgb(acc_a) / acc_a[..., 3].reshape(-1, 1)
    mean_b = rgb(acc_b) / acc_b[..., 3].reshape(-1, 1)
    # [POINT] 全体の平均 m = (m_A + m_B)/2 の分散は σ²/N、m_A − m_B の分散は 4σ²/N なので、
    #         m のノイズ ≈ |m_A − m_B| / 2。
    noise = np.abs(mean_a - mean_b) / 2.0
    return noise, (mean_a + mean_b) / 2.0


def check_path_trace(out_dir: Path) -> list[dict]:
    results = []
    for name in SCENARIOS:
        single, _ = read_exr(out_dir / f"{name}_single.exr")
        pt_single, _ = read_exr(out_dir / f"{name}_pt_single.exr")
        noise_s, _ = path_trace_noise(out_dir / f"{name}_pt_single")
        ref_s, test_s = rgb(single), rgb(pt_single)
        noise_rel = np.sqrt(np.mean(noise_s ** 2, axis=0)) / np.mean(ref_s, axis=0)
        diff_rel = rel_rmse(test_s, ref_s)
        bias = np.mean(test_s - ref_s, axis=0) / np.mean(ref_s, axis=0)
        bias_se = noise_rel / np.sqrt(ref_s.shape[0])
        # [POINT] パストレーサーが正しければ、単散乱との差はノイズ程度（比 ≈ 1）で、平均の偏りは
        #         標準誤差の数倍以内に収まる。
        ok = bool(np.all(diff_rel < 1.5 * noise_rel + 1e-3) and np.all(np.abs(bias) < 4.0 * bias_se + 1e-3))
        results.append({
            "check": f"V4a path trace (1 bounce) vs single scattering ({name})",
            "rel_rmse_rgb": diff_rel.tolist(),
            "noise_rel_rmse_rgb": noise_rel.tolist(),
            "bias_rgb": bias.tolist(),
            "bias_standard_error_rgb": bias_se.tolist(),
            "tolerance": "rel RMSE < 1.5 x noise + 1e-3, |bias| < 4 SE + 1e-3",
            "pass": ok,
        })

        lut, meta = read_exr(out_dir / f"{name}_lut.exr")
        raymarch, _ = read_exr(out_dir / f"{name}_raymarch.exr")
        pt, pt_meta = read_exr(out_dir / f"{name}_pt.exr")
        noise, _ = path_trace_noise(out_dir / f"{name}_pt")
        ref = rgb(pt)
        noise_abs = np.sqrt(np.mean(noise ** 2, axis=0))
        noise_rel = noise_abs / np.mean(ref, axis=0)
        # [POINT] 測った RMSE² = 近似の誤差² + パストレースのノイズ²（独立）なので、ノイズ分を差し引いた
        #         値を「近似の誤差」とする。
        lut_abs = np.sqrt(np.maximum(np.mean((rgb(lut) - ref) ** 2, axis=0) - noise_abs ** 2, 0.0))
        ray_abs = np.sqrt(np.maximum(np.mean((rgb(raymarch) - ref) ** 2, axis=0) - noise_abs ** 2, 0.0))
        lut_rel = lut_abs / np.mean(ref, axis=0)
        ray_rel = ray_abs / np.mean(ref, axis=0)
        # 多重散乱の成分だけの比較: (レイマーチ − 単散乱) / (パストレース − 散乱 1 回のパストレース)
        ms_ratio = (rgb(raymarch) - ref_s).sum(axis=0) / (ref - test_s).sum(axis=0)
        gated = name in V4_GATED
        results.append({
            "check": f"V4 LUT / raymarch vs path trace ({name})",
            "lut_abs_rmse_per_lux_rgb": lut_abs.tolist(),
            "raymarch_abs_rmse_per_lux_rgb": ray_abs.tolist(),
            "lut_rel_rmse_rgb": lut_rel.tolist(),
            "raymarch_rel_rmse_rgb": ray_rel.tolist(),
            "multiple_scattering_ratio_approx_over_pt_rgb": ms_ratio.tolist(),
            "multiple_scattering_share_of_pt_rgb": ((ref - test_s).sum(axis=0) / ref.sum(axis=0)).tolist(),
            "mean_per_lux_rgb": np.mean(ref, axis=0).tolist(),
            "lut_bias_rgb": (np.mean(rgb(lut) - ref, axis=0) / np.mean(ref, axis=0)).tolist(),
            "raymarch_bias_rgb": (np.mean(rgb(raymarch) - ref, axis=0) / np.mean(ref, axis=0)).tolist(),
            "path_trace_noise_rel_rmse_rgb": noise_rel.tolist(),
            "samples_per_channel": pt_meta.get("samples_per_channel"),
            "sun_zenith_deg": meta["atmosphere"]["sun_zenith_deg"],
            "tolerance": V4_ABS_RMSE_PER_LUX_TOLERANCE if gated else None,
            "pass": bool(np.all(lut_abs < V4_ABS_RMSE_PER_LUX_TOLERANCE)) if gated else True,
            "info": None if gated else "recorded only (earth shadow)",
        })
    return results


# ---------------------------------------------------------------------- V3b
def closeup_directions(meta: dict, n: int) -> np.ndarray:
    """sun_closeup のカメラ（Camera::viewRotation + Reverse-Z 射影と同じ）の各ピクセルの視線方向"""

    yaw, pitch = np.radians(meta["camera_yaw_deg"]), np.radians(meta["camera_pitch_deg"])
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    t = np.tan(np.radians(meta["vertical_fov_deg"]) / 2.0)
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y = np.meshgrid(c, -c)
    dirs = forward + x[..., None] * t * right + y[..., None] * t * up
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def halo_statistics(image: np.ndarray, dirs: np.ndarray, sun: np.ndarray) -> dict:
    """背景（太陽から 2.5° 以上の中央値）を引いた光輪の、ピーク値・広がり（水平・鉛直の標準偏差 [°]）・中心"""

    lum = image[..., :3].mean(axis=-1).astype(np.float64)
    angle = np.degrees(np.arccos(np.clip(dirs @ sun, -1.0, 1.0)))
    lum = np.clip(lum - np.median(lum[angle > 2.5]), 0.0, None)
    elevation = np.degrees(np.arcsin(dirs[..., 1])) - np.degrees(np.arcsin(sun[1]))
    azimuth = np.degrees(np.arctan2(dirs[..., 2], dirs[..., 0])) * np.cos(np.arcsin(dirs[..., 1]))
    w = lum.sum()
    cx, cy = (lum * azimuth).sum() / w, (lum * elevation).sum() / w
    return {
        "peak": float(lum.max()),
        "sigma_h_deg": float(np.sqrt((lum * (azimuth - cx) ** 2).sum() / w)),
        "sigma_v_deg": float(np.sqrt((lum * (elevation - cy) ** 2).sum() / w)),
        "center_h_deg": float(cx),
        "center_v_deg": float(cy),
    }


def check_sun_closeup(out_dir: Path) -> list[dict]:
    results = []
    for path in sorted(out_dir.glob("closeup_*_lut.exr")):
        lut, meta = read_exr(path)
        ref, _ = read_exr(path.with_name(path.name.replace("_lut", "_raymarch")))
        a = meta["atmosphere"]
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        dirs = closeup_directions(meta, lut.shape[0])
        s_lut, s_ref = halo_statistics(lut, dirs, sun), halo_statistics(ref, dirs, sun)
        peak_ratio = s_lut["peak"] / s_ref["peak"]
        sigma_ratio = [s_lut["sigma_h_deg"] / s_ref["sigma_h_deg"], s_lut["sigma_v_deg"] / s_ref["sigma_v_deg"]]
        center_offset = float(np.hypot(s_lut["center_h_deg"] - s_ref["center_h_deg"],
                                       s_lut["center_v_deg"] - s_ref["center_v_deg"]))
        ok = (0.95 < peak_ratio < 1.05 and all(0.95 < r < 1.05 for r in sigma_ratio) and center_offset < 0.05)
        results.append({
            "check": f"V3b sun aureole g={a['mie_phase_g']:.2f} sz={a['sun_zenith_deg']:.2f}deg",
            "peak_ratio": float(peak_ratio),
            "sigma_ratio_h_v": [float(r) for r in sigma_ratio],
            "center_offset_deg": center_offset,
            "lut": s_lut,
            "raymarch": s_ref,
            "tolerance": "peak and sigma ratio 0.95-1.05, center offset < 0.05 deg",
            "pass": bool(ok),
        })
    return results


# ---------------------------------------------------------------------- V9
def check_spectral(out_dir: Path) -> list[dict]:
    wl = np.arange(360.0, 830.0 + 1e-9, 5.0)
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
        colour.SpectralShape(360.0, 830.0, 5.0)).values
    solar = sd.solar_irradiance(wl)
    sets = sorted({p.name.split("_sz")[0] for p in out_dir.glob("*_sz*_directions.exr")})
    results = []
    for sun_zenith in (30.0, 60.0, 85.0):
        _, meta0 = read_exr(out_dir / f"{sets[0]}_sz{sun_zenith:g}.exr")
        dirs, _ = read_exr(out_dir / f"{sets[0]}_sz{sun_zenith:g}_directions.exr")
        d = dirs[..., :3].reshape(-1, 3).astype(np.float64)
        atm = Atmosphere.spectral(meta0, wl)
        a = meta0["atmosphere"]
        camera = np.array([0.0, atm.bottom_radius_km + float(meta0["camera_altitude_km"]), 0.0])
        sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
        # 波長ごとの L/E_e(λ) [sr⁻¹] → 分光放射輝度 L_e(λ) [W·m⁻²·sr⁻¹·nm⁻¹] → XYZ [cd/m²]
        per_irradiance = atm.single_scattering_per_lux(camera, d, sun, outer_panels=16, inner_panels=4, chunk=2)
        # [POINT] 5 nm 刻みの和で求めた太陽照度を、アプリの照度（JSON の 1 nm 刻みの値）に合わせる。
        #         波長刻みの違いによる一律の差（0.1% 程度）を、色空間の誤差に混ぜないため。
        spectral_lux = 683.0 * np.sum(solar * cmfs[:, 1]) * 5.0
        scale = float(a["sun_illuminance_lux"]) / spectral_lux
        xyz_ref = 683.0 * (per_irradiance * solar) @ cmfs * 5.0 * scale
        # リファレンス自身の収束（数方向で区間数を倍に）
        fine = atm.single_scattering_per_lux(camera, d[::31], sun, outer_panels=32, inner_panels=8, chunk=2)
        self_err = float(np.max(np.abs(per_irradiance[::31] - fine) / fine))

        for set_name in sets:
            gpu, meta = read_exr(out_dir / f"{set_name}_sz{sun_zenith:g}.exr")
            space = meta["color_space"]
            cs = colour.RGB_COLOURSPACES[WORKING_SPACES[space]]
            sun_rgb = float(meta["atmosphere"]["sun_illuminance_lux"]) * np.asarray(meta["atmosphere"]["sun_color"])
            gpu_rgb = rgb(gpu) * sun_rgb                                   # 作業色空間の RGB [cd/m²]
            gpu_xyz = gpu_rgb @ np.asarray(cs.matrix_RGB_to_XYZ).T
            to_2020 = colour.matrix_RGB_to_RGB(cs, colour.RGB_COLOURSPACES["ITU-R BT.2020"])
            xyz_to_2020 = np.asarray(colour.RGB_COLOURSPACES["ITU-R BT.2020"].matrix_XYZ_to_RGB)
            de = delta_e_itp(gpu_rgb @ to_2020.T, xyz_ref @ xyz_to_2020.T)
            lum_err = (gpu_xyz[:, 1] - xyz_ref[:, 1]) / xyz_ref[:, 1]
            results.append({
                "check": f"V9 {set_name} sz={sun_zenith:g}deg",
                "working_space": space,
                "mean_delta_e_itp": float(de.mean()),
                "max_delta_e_itp": float(de.max()),
                "mean_luminance_rel_error": float(lum_err.mean()),
                "max_abs_luminance_rel_error": float(np.abs(lum_err).max()),
                "reference_self_error": self_err,
                "tolerance": None,
                "pass": bool(self_err < 1e-4),
                "info": "recorded only (design.md V9)",
            })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "phase3")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    args = parser.parse_args()

    if not args.skip_run:
        for test in ("multiscattering", "sky_lut", "sun_closeup", "pathtrace", "spectral"):
            run_app(args.exe, test, args.out / test)

    results = (check_multiscattering(args.out / "multiscattering") + check_sky_lut(args.out / "sky_lut")
               + check_sun_closeup(args.out / "sun_closeup")
               + check_path_trace(args.out / "pathtrace") + check_spectral(args.out / "spectral"))
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    for r in results:
        if "mean_rel_error_significant" in r:
            line = (f"max {r['max_rel_error']:.2e} | mean {r['mean_rel_error_significant']:.2e}"
                    f" | sun>0.2 max {r['max_rel_error_sun_above_horizon']:.2e}"
                    f" | terminator max {r['max_rel_error_near_terminator']:.2e}"
                    " | bias " + " ".join(f"{v:+.2e}" for v in r["mean_signed_rel_error_significant_rgb"]))
        elif "max_rel_error" in r and "default_settings_max_rel_error_info" in r:
            line = f"max={r['max_rel_error']:.2e} (default settings {r['default_settings_max_rel_error_info']:.2e})"
        elif "peak_ratio" in r:
            line = (f"peak LUT/ref {r['peak_ratio']:.3f} | sigma H/V ratio "
                    + " ".join(f"{v:.3f}" for v in r["sigma_ratio_h_v"])
                    + f" | center offset {r['center_offset_deg']:.3f} deg")
        elif "rel_rmse_rgb" in r and "noise_rel_rmse_rgb" in r:
            line = ("relRMSE " + " ".join(f"{v:.2e}" for v in r["rel_rmse_rgb"])
                    + " | noise " + " ".join(f"{v:.2e}" for v in r["noise_rel_rmse_rgb"]))
        elif "rel_rmse_rgb" in r:
            line = ("relRMSE " + " ".join(f"{v:.2e}" for v in r["rel_rmse_rgb"])
                    + f" | max {r['max_rel_error']:.2e} @ {r['max_rel_error_elevation_deg']:.1f}deg"
                    + f" | horizon max {r['horizon_band_max_rel_error']:.2e}")
        elif "lut_rel_rmse_rgb" in r:
            line = ("LUT abs " + " ".join(f"{v:.1e}" for v in r["lut_abs_rmse_per_lux_rgb"])
                    + " rel " + " ".join(f"{v:.1%}" for v in r["lut_rel_rmse_rgb"])
                    + " | MS approx/PT " + " ".join(f"{v:.2f}" for v in r["multiple_scattering_ratio_approx_over_pt_rgb"]))
        else:
            line = (f"dE_ITP mean {r['mean_delta_e_itp']:.2f} max {r['max_delta_e_itp']:.2f}"
                    f" | Y err mean {r['mean_luminance_rel_error']:+.2%} max {r['max_abs_luminance_rel_error']:.2%}")
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']:<58} {line}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
