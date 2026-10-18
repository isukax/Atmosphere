"""V4b の見た目の比較: 目標の大気（research.md §12 の C+A+B+D）で、多重散乱の近似とパストレースを画面で並べる。

    reference/.venv/Scripts/python.exe reference/compare_v4b_views.py

  1. 表の媒質を dumps/v4b_views/media に書き出す（validate_v4b.py と同じ）。
  2. アプリの pathtrace_aerosol_views で、5 シナリオ × 太陽の方 / 反対の画面（画角 50°、320×180）をパストレースする
     （散乱最大 32 回、チャンネルあたり 16,384 経路、地表アルベド 0.3）。
  3. 同じ画面を Python の近似（aerosol_experiment.py）で描いて比べる。近似は 2 通り:
     - current: 今の Hillaire 2020 の Ψ_ms（GPU と同じ 64 方向・等方）
     - option1: 改善案 H 案 1（research.md §11.4）。L₂ の太陽光の 1 回目の散乱に実際の位相関数を使い、
                方向は太陽中心の格子（θ = π·u²、32 × 方位 32 = 1,024 方向、重みは立体角）。f_ms と、描画時の
                最後の散乱は等方のまま

[POINT] V4b-2（地表アルベド 0、全天）と違い、地面を照らす条件で見た目を比べる。
  - Ψ_ms の地表アルベドも 0.3（GPU の Multi-scattering LUT と同じ）。
  - 近似側の地面 = 太陽の直達光 + 空からの照度（その近似の空を地表で半球積分した表）のランバート反射。
    GPU の描画（SH の照度）とは別の作り方。パストレースは地面と大気の間の往復をすべて含む。
  - 太陽ディスクは描かない。露出はシナリオごとに、パストレースの太陽の方の画面の中央値で決め、同じシナリオの
    すべての画面に同じ値を使う。
  判定はしない（記録）。ΔE_ITP はパストレースのノイズを差し引いた RMS（validate_v4b.py と同じ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerosol_experiment as ae  # noqa: E402
import validate_v4b as v4b  # noqa: E402
from atmosphere_ref import sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from pngio import write_png  # noqa: E402
from validate_phase1 import ROOT, delta_e_itp, run_app  # noqa: E402
from validate_phase3 import rgb  # noqa: E402

MEDIUM = "C_A_B_D"
SCENARIOS = ("day", "low_sun", "sunset", "day_10km", "altitude_3km")
SIDES = ("front", "back")
VARIANTS = ("current", "option1")
GROUND_ALBEDO = 0.3
VIEW_STEPS = 128


class Option1MultiScatteringTable(ae.MultiScatteringTable):
    """改善案 H 案 1 の Ψ_ms: L₂ だけ実際の位相関数、太陽中心の方向の格子（θ = π·u²）"""

    def __init__(self, atmo, table, size: int = 32, steps: int = 48, ground_albedo: float = ae.GROUND_ALBEDO_MS,
                 n_theta: int = 32, n_phi: int = 32):
        u0, u1 = np.arange(n_theta) / n_theta, (np.arange(n_theta) + 1) / n_theta
        theta = np.pi * (0.5 * (u0 + u1)) ** 2
        d_cos = np.cos(np.pi * u0 ** 2) - np.cos(np.pi * u1 ** 2)
        phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
        th, ph = np.meshgrid(theta, phi, indexing="ij")
        local = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1, 3)
        weight = np.repeat(d_cos[:, None], n_phi, axis=1).reshape(-1) * (2.0 * np.pi / n_phi)  # 立体角（和 = 4π）
        self.h = np.linspace(0.0, 100.0, size)
        self.mu = np.linspace(-1.0, 1.0, size)
        self.psi = np.zeros((size, size, 3))
        for i, hh in enumerate(self.h):
            origin = np.array([0.0, ae.BOTTOM_KM + max(hh, 0.001), 0.0])
            for j, ms in enumerate(self.mu):
                sun = np.array([np.sqrt(max(0.0, 1.0 - ms * ms)), ms, 0.0])
                # 太陽を +Z に置いた局所座標（太陽は XY 平面内なので、世界の Z 軸は太陽に直交する）
                x = np.cross([0.0, 0.0, 1.0], sun)
                x /= np.linalg.norm(x)
                dirs = local @ np.stack([x, np.cross(sun, x), sun])
                # f（∫T·σs dt）は位相関数によらないので、L₂ と同じ呼び出しで得られる
                lum, f = ae.integrate(atmo, table, None, origin, dirs, sun, steps,
                                      ground_albedo if ground_albedo > 0.0 else None, isotropic=False)
                second = (weight[:, None] * lum).sum(axis=0) / (4.0 * np.pi)
                fms = (weight[:, None] * f).sum(axis=0) / (4.0 * np.pi)
                self.psi[i, j] = second / (1.0 - fms)


def camera_directions(meta: dict, w: int, h: int) -> np.ndarray:
    """Camera::viewRotation + 射影（PathTrace.hlsl の CameraViewDirection）と同じ視線方向。(h·w, 3)"""

    yaw, pitch = np.radians(meta["camera_yaw_deg"]), np.radians(meta["camera_pitch_deg"])
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    t = np.tan(np.radians(meta["vertical_fov_deg"]) / 2.0)
    x = ((np.arange(w) + 0.5) / w * 2.0 - 1.0) * t * w / h
    y = (1.0 - (np.arange(h) + 0.5) / h * 2.0) * t
    xx, yy = np.meshgrid(x, y)
    d = forward + xx[..., None] * right + yy[..., None] * up
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)


class GroundSkyIrradiance:
    """地表（高度 0）での空からの照度 [単位照度あたり]。近似の空（単散乱 + Ψ_ms）を上半球で積分した、μ_s の表。"""

    def __init__(self, atmo, table, psi, count: int = 65):
        n_t, n_p = 16, 32
        ct = 1.0 - (np.arange(n_t) + 0.5) / n_t  # cosθ が一様（上半球）
        ph = 2.0 * np.pi * (np.arange(n_p) + 0.5) / n_p
        c, p = np.meshgrid(ct, ph, indexing="ij")
        s = np.sqrt(1.0 - c * c)
        dirs = np.stack([s * np.cos(p), c, s * np.sin(p)], axis=-1).reshape(-1, 3)
        weight = (2.0 * np.pi / dirs.shape[0]) * dirs[:, 1]  # dω·cosθ
        origin = np.array([0.0, ae.BOTTOM_KM + 0.001, 0.0])
        self.mu = np.linspace(-1.0, 1.0, count)
        self.e = np.zeros((count, 3))
        for i, ms in enumerate(self.mu):
            sun = np.array([np.sqrt(max(0.0, 1.0 - ms * ms)), ms, 0.0])
            lum, _ = ae.integrate(atmo, table, psi, origin, dirs, sun, 64, None)
            self.e[i] = (weight[:, None] * lum).sum(axis=0)

    def __call__(self, mu: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(mu, self.mu, self.e[:, c]) for c in range(3)], axis=-1)


def render_approx(atmo, table, psi, sky_e, origin, dirs, sun):
    """近似の描画（単位照度あたり）と、地面に当たる視線のマスク"""

    lum, _ = ae.integrate(atmo, table, psi, origin, dirs, sun, VIEW_STEPS, GROUND_ALBEDO)
    # 地面の空からの照度の項（integrate の地面は太陽の直達光だけ）。視線の透過率は integrate と同じ区間で求める
    t_end, hit = ae.ray_extent(origin, dirs)
    s = (np.arange(VIEW_STEPS) + 0.5) / VIEW_STEPS
    d = dirs[hit]
    t = t_end[hit, None] * s * s
    dt = t_end[hit, None] * 2.0 * s / VIEW_STEPS
    p = origin + t[..., None] * d[:, None, :]
    _, _, ext = atmo.coefficients(np.linalg.norm(p, axis=-1) - ae.BOTTOM_KM)
    view_t = np.exp(-np.sum(ext * dt[..., None], axis=1))
    gp = origin + t_end[hit, None] * d
    mu_sun = gp @ sun / np.linalg.norm(gp, axis=1)
    lum[hit] += view_t * sky_e(mu_sun) * GROUND_ALBEDO / np.pi
    return lum, hit


def systematic(de: np.ndarray, de_noise: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(max(np.mean(de[mask] ** 2) - np.mean(de_noise[mask] ** 2) / 4.0, 0.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=ROOT / "build" / "vs2026-x64" / "Release" / "atmosphere.exe")
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "v4b_views")
    parser.add_argument("--skip-run", action="store_true", help="既存のダンプだけを比較する")
    args = parser.parse_args()
    atmo = v4b.build_media(args.out)[MEDIUM]
    if not args.skip_run:
        run_app(args.exe, "pathtrace_aerosol_views", args.out)

    table = ae.TransmittanceTable(atmo)
    psis = {"current": ae.MultiScatteringTable(atmo, table, ground_albedo=GROUND_ALBEDO),
            "option1": Option1MultiScatteringTable(atmo, table, ground_albedo=GROUND_ALBEDO)}
    sky_es = {k: GroundSkyIrradiance(atmo, table, psi) for k, psi in psis.items()}
    report = []
    rows = {side: [] for side in SIDES}
    for scenario in SCENARIOS:
        exposure = None
        for side in SIDES:
            stem = args.out / f"{MEDIUM}_{scenario}_{side}"
            pt_img, meta = read_exr(stem.with_name(stem.name + "_pt.exr"))
            h, w = pt_img.shape[:2]
            a = meta["atmosphere"]
            assert np.allclose(a["ground_albedo"], GROUND_ALBEDO)
            sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
            origin = np.array([0.0, a["bottom_radius_km"] + meta["camera_altitude_km"], 0.0])
            dirs = camera_directions(meta, w, h)
            pt = rgb(pt_img)
            if exposure is None:
                exposure = ae.exposure_for(pt)  # 太陽の方の画面（front が先）
            d_pt = v4b.display_nits(pt, exposure)
            acc_a, _ = read_exr(stem.with_name(stem.name + "_pt_accA.exr"))
            acc_b, _ = read_exr(stem.with_name(stem.name + "_pt_accB.exr"))
            half_a = rgb(acc_a) / acc_a[..., 3].reshape(-1, 1)
            half_b = rgb(acc_b) / acc_b[..., 3].reshape(-1, 1)
            de_noise = delta_e_itp(v4b.display_nits(half_a, exposure), v4b.display_nits(half_b, exposure))

            images, heats = [], []
            for variant in VARIANTS:
                approx, ground = render_approx(atmo, table, psis[variant], sky_es[variant], origin, dirs, sun)
                de = delta_e_itp(v4b.display_nits(approx, exposure), d_pt)
                entry = {"scenario": scenario, "side": side, "variant": variant,
                         "exposure_per_lux_to_fb": exposure, "samples_per_channel": meta.get("samples_per_channel")}
                for name, m in (("all", np.ones_like(ground)), ("sky", ~ground), ("ground", ground)):
                    if not np.any(m):
                        continue
                    entry[name] = {
                        "pixels_fraction": float(m.mean()),
                        "delta_e_systematic_rms": systematic(de, de_noise, m),
                        "delta_e_mean": float(de[m].mean()),
                        "delta_e_p95": float(np.percentile(de[m], 95)),
                        "delta_e_noise_rms": float(np.sqrt(np.mean(de_noise[m] ** 2)) / 2.0),
                        "luminance_ratio_approx_over_pt": float(ae.luminance(approx[m]).mean()
                                                                / ae.luminance(pt[m]).mean()),
                        "mean_ratio_approx_over_pt_rgb": (approx[m].mean(axis=0) / pt[m].mean(axis=0)).tolist(),
                    }
                report.append(entry)
                images.append(ae.to_display(approx, exposure)[2].reshape(h, w, 3))
                heats.append(v4b.heat(de).reshape(h, w, 3))
                line = "  ".join(f"{k} ΔE_sys {entry[k]['delta_e_systematic_rms']:.1f}"
                                 f" ratio {' '.join(f'{x:.2f}' for x in entry[k]['mean_ratio_approx_over_pt_rgb'])}"
                                 for k in ("sky", "ground") if k in entry)
                print(f"{scenario:<13} {side:<5} {variant:<8} {line}")
            # 1 行: 今の近似 | 案 1 | パストレース | ΔE（今の近似） | ΔE（案 1）
            gap = np.ones((h, 3, 3))
            tiles = [images[0], gap, images[1], gap, ae.to_display(pt, exposure)[2].reshape(h, w, 3),
                     np.ones((h, 12, 3)), heats[0], gap, heats[1]]
            rows[side] += [np.concatenate(tiles, axis=1), np.ones((3, sum(t.shape[1] for t in tiles), 3))]
    for side in SIDES:
        write_png(ROOT / "docs" / "images" / f"v4b_views_{MEDIUM}_{side}.png", np.concatenate(rows[side][:-1]))
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
