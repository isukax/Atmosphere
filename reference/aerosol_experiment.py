"""エアロゾルのモデルの違いが見た目にどれだけ効くかを測る実験（research.md §11 の改善案 A / B / C / D）。

    reference/.venv/Scripts/python.exe reference/aerosol_experiment.py

GPU の実装の前に、Python だけで「どの案が見た目に効くか」を数字で決めるためのもの。リファレンス（正解）ではない。

描画: RGB（Hillaire2020 の係数セット、680 / 550 / 440 nm を Rec.709 の RGB とみなす。太陽は白）。
  - 単散乱: 視線上 N 点（t = T·u²、中点則）。太陽方向の透過率は (高度, μ) の表を補間。惑星の影あり。
  - 多重散乱: Hillaire 2020 の近似（GPU の Multi-scattering LUT と同じ 64 方向・等方・地表アルベド 0.3）を
              この実験の大気で作り直し、σs_total·Ψ_ms を足す。
  - 地表に当たる視線: 地表（アルベド 0.18）の太陽の直達光の反射だけ（空からの照度は入れない）。
  - 太陽ディスクは描かない（光輪を見るため）。
エアロゾルは「Mie の成分」のリストで表す。成分ごとに σs / σa（RGB）、鉛直分布、位相関数（RGB の表 or Cornette-Shanks）。
  同じ鉛直分布の粒子はまとめて 1 成分にする（σs で重みを付けた位相関数の混合。design の相談の「LUT は鉛直分布ごとに 1 枚」）。

変種（案の切り替え）:
  base        今の既定（Hillaire 2020: AOD550 ≈ 0.005、灰色、Cornette-Shanks g = 0.8、指数 H = 1.2 km）
  base+A      base の量（AOD550）のまま、位相関数を Mie（二峰性の粒径分布、550 nm の値を全チャンネルに）に替える
  base+B      base の量のまま、Mie の係数の波長依存（位相関数は Cornette-Shanks のまま）
  base+A+B    base の量のまま、Mie の位相関数と係数をチャンネルごとに
              （base 系は C 系と同じ作り方で、単散乱アルベドは Mie 理論の値。2026-10-18 までの base+A は base の
              σs・σa のまま位相関数だけを替えていた。research.md §12.3 の表はその値）
  C           量だけを現実的に（AOD550 = 0.1）。ほかは base と同じ
  C+A         C + Mie の位相関数（550 nm の値を全チャンネルに）
  C+B         C + Mie の係数の波長依存（位相関数は Cornette-Shanks のまま）
  C+D         C + 鉛直分布を境界層の形に（fine / coarse で別の高さ）
  C+A+B       C + Mie の位相関数と係数をチャンネルごとに
  C+A+B+D     すべて
比較: GT7 のトーンマップ（SDR / HDR 1000 nit）の出力で ΔE_ITP（BT.2124、1 ≈ 弁別閾）。
  露出は (1) シナリオで固定（base の正面の中央値で決める）と (2) 変種ごとに中央値を合わせる（自動露出）の 2 通り。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mie  # noqa: E402
# [POINT] エアロゾルのモデル（仮定・鉛直分布・位相関数・プリセット）は aerosol_models.py にある（derive_coefficients.py と共有）。
#         ここからも同じ名前で使えるようにする（validate_v4b.py などが ae.* で参照する）。
from aerosol_models import (  # noqa: E402,F401
    AOD550_REALISTIC, BASE_G, BASE_MIE_ABSORPTION, BASE_MIE_SCALE_HEIGHT, BASE_MIE_SCATTERING, COARSE, FINE,
    FINE_FRACTION_OF_AOD, G_CHANNEL, WAVELENGTHS_UM, Atmosphere, CornetteShanks, MieComponent, ModeOptics, PhaseTable,
    Profile, boundary_layer_profile, build_variants, column_km, cornette_shanks, exponential_profile, mie_mixture,
    mode_optics, rayleigh_phase,
)
from gt7_tone_mapping import tone_mapper  # noqa: E402
from pngio import write_png  # noqa: E402
from validate_phase1 import delta_e_itp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOTTOM_KM, TOP_KM = 6360.0, 6460.0
GROUND_ALBEDO_SURFACE = 0.18
GROUND_ALBEDO_MS = 0.3
REC709_TO_REC2020 = np.array([[0.627403895934699, 0.329283038377884, 0.043313065687417],
                              [0.069097289358232, 0.919540395075459, 0.011362315566309],
                              [0.016391438875150, 0.088013307877226, 0.895595253247624]])
REC2020_TO_REC709 = np.linalg.inv(REC709_TO_REC2020)

# ---------------------------------------------------------------------- 透過率の表（大気上端まで）
class TransmittanceTable:
    """(高度, μ) → 大気上端までの光学的厚さ τ（RGB）。μ は 0 付近（地平線）を細かく刻む。
    地表に当たる方向は、地平線の値で埋めて補間をなめらかにする（引くときに惑星の影として 0 にする）。"""

    def __init__(self, atmo: Atmosphere, heights: int = 160, mus: int = 1201, steps: int = 256):
        v = np.linspace(0.0, 1.0, heights)
        self.h = 100.0 * v * v
        u = np.linspace(-1.0, 1.0, mus)
        self.mu = np.sign(u) * u * u
        r = BOTTOM_KM + self.h[:, None]
        mu_h = -np.sqrt(np.maximum(1.0 - (BOTTOM_KM / r) ** 2, 0.0))
        mu = np.maximum(self.mu[None, :], mu_h + 1e-6)                    # 地表に当たる方向は地平線で代用
        tau = optical_depth_to_top(atmo, np.broadcast_to(r, mu.shape).ravel(), mu.ravel(), steps)
        self.tau = tau.reshape(heights, mus, 3)

    def transmittance(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        h = np.clip(r - BOTTOM_KM, 0.0, 100.0)
        mu = np.clip(mu, -1.0, 1.0)
        # 高度の格子は h = 100·v² なので v = √(h/100) が等間隔
        fh = np.sqrt(h / 100.0) * (self.h.size - 1)
        i0 = np.clip(np.floor(fh).astype(int), 0, self.h.size - 2)
        th = fh - i0
        uu = np.sign(mu) * np.sqrt(np.abs(mu))
        fm = (uu + 1.0) * 0.5 * (self.mu.size - 1)
        j0 = np.clip(np.floor(fm).astype(int), 0, self.mu.size - 2)
        tm = fm - j0
        t = self.tau
        tau = ((1 - th)[..., None] * ((1 - tm)[..., None] * t[i0, j0] + tm[..., None] * t[i0, j0 + 1])
               + th[..., None] * ((1 - tm)[..., None] * t[i0 + 1, j0] + tm[..., None] * t[i0 + 1, j0 + 1]))
        shadow = (mu < 0.0) & (r * r * (mu * mu - 1.0) + BOTTOM_KM ** 2 >= 0.0)
        return np.where(shadow[..., None], 0.0, np.exp(-tau))


def distance_to_top(r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, -r * mu + np.sqrt(np.maximum(r * r * (mu * mu - 1.0) + TOP_KM ** 2, 0.0)))


def optical_depth_to_top(atmo: Atmosphere, r: np.ndarray, mu: np.ndarray, steps: int, chunk: int = 4096) -> np.ndarray:
    out = np.zeros((r.size, 3))
    s = (np.arange(steps) + 0.5) / steps
    for a in range(0, r.size, chunk):
        rr, mm = r[a:a + chunk], mu[a:a + chunk]
        d = distance_to_top(rr, mm)
        t = d[:, None] * s * s                                           # 始点の近く（濃い所）を細かく
        dt = d[:, None] * 2.0 * s / steps
        ri = np.sqrt(t * t + 2.0 * rr[:, None] * mm[:, None] * t + rr[:, None] ** 2)
        _, _, ext = atmo.coefficients(ri - BOTTOM_KM)
        out[a:a + chunk] = np.einsum("nm,nmc->nc", dt, ext)
    return out


# ---------------------------------------------------------------------- 視線の積分
def ray_extent(origin: np.ndarray, dirs: np.ndarray):
    """視線の終点（大気上端 or 地表）までの距離と、地表に当たるか。origin は大気の中にあること。"""

    r = np.linalg.norm(origin)
    b = dirs @ origin
    c_top = (r - TOP_KM) * (r + TOP_KM)
    t_top = -b + np.sqrt(np.maximum(b * b - c_top, 0.0))
    c_g = (r - BOTTOM_KM) * (r + BOTTOM_KM)
    disc = b * b - c_g
    t_ground = np.where(disc >= 0.0, -b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
    hit = (t_ground > 0.0) & (t_ground < t_top)
    return np.where(hit, t_ground, t_top), hit


def integrate(atmo: Atmosphere, table: TransmittanceTable, psi: "MultiScatteringTable | None", origin: np.ndarray,
              dirs: np.ndarray, sun: np.ndarray, steps: int, ground_albedo: float | None, isotropic: bool = False,
              chunk: int = 2048):
    """視線ごとの (L, F)。L = 単位照度あたりの輝度 [sr⁻¹]、F = ∫ T_view·σs dt（f_ms 用）。
    psi: 等方の Ψ_ms の表（lookup(r, μs)、光源 = σs·Ψ_ms）か、光源を作る表（source(...)、ms_sh.py の SH）。"""

    n = dirs.shape[0]
    lum = np.zeros((n, 3))
    fms = np.zeros((n, 3))
    s = (np.arange(steps) + 0.5) / steps
    for a in range(0, n, chunk):
        d = dirs[a:a + chunk]
        t_end, hit = ray_extent(origin, d)
        t = t_end[:, None] * s * s
        dt = t_end[:, None] * 2.0 * s / steps
        p = origin + t[..., None] * d[:, None, :]
        r = np.linalg.norm(p, axis=-1)
        h = r - BOTTOM_KM
        s_r, s_m, ext = atmo.coefficients(h)
        seg = ext * dt[..., None]
        depth = np.cumsum(seg, axis=1) - 0.5 * seg                       # 中点までの光学的厚さ
        view_t = np.exp(-depth)
        mu_sun = np.einsum("nmk,k->nm", p, sun) / r
        sun_t = table.transmittance(r, mu_sun)
        cos_theta = d @ sun
        if isotropic:
            source = (s_r + sum(s_m)) / (4.0 * np.pi)
        else:
            source = s_r * rayleigh_phase(cos_theta)[:, None, None]
            for comp, sigma in zip(atmo.mie, s_m):
                source = source + sigma * comp.phase(cos_theta)[:, None, :]
        scattering = s_r + sum(s_m)
        integrand = view_t * source * sun_t
        if psi is not None:
            if getattr(psi, "source", None) is not None:
                # SH の多重散乱（ms_sh.py）: 光源が視線の向きと成分ごとの χ によるので、表が光源を作る
                integrand = integrand + view_t * psi.source(p, r, mu_sun, d, sun, [s_r] + list(s_m))
            else:
                integrand = integrand + view_t * scattering * psi.lookup(r, mu_sun)
        lum[a:a + chunk] = np.einsum("nm,nmc->nc", dt, integrand)
        fms[a:a + chunk] = np.einsum("nm,nmc->nc", dt, view_t * scattering)
        if ground_albedo is not None and np.any(hit):
            gp = origin + t_end[hit, None] * d[hit]
            gr = np.linalg.norm(gp, axis=1)
            cos_sun = np.maximum(gp @ sun / gr, 0.0)
            total_t = np.exp(-np.sum(seg[hit], axis=1))
            g_sun = table.transmittance(gr, gp @ sun / gr)
            lum[a:a + chunk][hit] += total_t * g_sun * cos_sun[:, None] * ground_albedo / np.pi
    return lum, fms


class MultiScatteringTable:
    """Hillaire 2020 の Ψ_ms（GPU の Multi-scattering LUT と同じ 64 方向・等方・地表アルベド 0.3）。(高度, μ_s) の表。"""

    def __init__(self, atmo: Atmosphere, table: TransmittanceTable, size: int = 32, steps: int = 48,
                 ground_albedo: float = GROUND_ALBEDO_MS):
        k = np.arange(64)
        theta = 2.0 * np.pi * (k % 8 + 0.5) / 8.0
        phi = np.arccos(1.0 - 2.0 * (k // 8 + 0.5) / 8.0)
        dirs = np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], axis=1)
        self.h = np.linspace(0.0, 100.0, size)
        self.mu = np.linspace(-1.0, 1.0, size)
        self.psi = np.zeros((size, size, 3))
        for i, hh in enumerate(self.h):
            origin = np.array([0.0, BOTTOM_KM + max(hh, 0.001), 0.0])
            for j, ms in enumerate(self.mu):
                sun = np.array([np.sqrt(max(0.0, 1.0 - ms * ms)), ms, 0.0])
                lum, f = integrate(atmo, table, None, origin, dirs, sun, steps,
                                   ground_albedo if ground_albedo > 0.0 else None, isotropic=True)
                second = lum.mean(axis=0)
                fms = f.mean(axis=0)
                self.psi[i, j] = second / (1.0 - fms)

    def lookup(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        fh = np.clip((r - BOTTOM_KM) / 100.0, 0.0, 1.0) * (self.h.size - 1)
        fm = (np.clip(mu, -1.0, 1.0) + 1.0) * 0.5 * (self.mu.size - 1)
        i0 = np.clip(np.floor(fh).astype(int), 0, self.h.size - 2)
        j0 = np.clip(np.floor(fm).astype(int), 0, self.mu.size - 2)
        th, tm = (fh - i0)[..., None], (fm - j0)[..., None]
        p = self.psi
        return ((1 - th) * ((1 - tm) * p[i0, j0] + tm * p[i0, j0 + 1])
                + th * ((1 - tm) * p[i0 + 1, j0] + tm * p[i0 + 1, j0 + 1]))


# ---------------------------------------------------------------------- 表の媒質（パストレーサー用、design.md §5.8）
# [POINT] shaders/atmosphere/PathTrace.hlsl の「表の媒質」と同じ並び・同じ補間。V4b では Python の近似もこの表で
#         描くので、GPU のパストレースと Python の近似はまったく同じ媒質を見る。
MEDIUM_HEADER_FLOATS = 16
MEDIUM_PROFILE_COUNT = 256
MEDIUM_PHASE_COUNT = 2049
MEDIUM_MAX_COMPONENTS = 2


class TabulatedProfile:
    """高度 h_i = 100·(i/255)² km の表を、v = √(h/100) で線形補間する鉛直分布（GPU と同じ）。"""

    def __init__(self, profile: Callable[[np.ndarray], np.ndarray]):
        v = np.linspace(0.0, 1.0, MEDIUM_PROFILE_COUNT)
        self.values = np.asarray(profile(100.0 * v * v), dtype=np.float32).astype(np.float64)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        f = np.sqrt(np.clip(np.asarray(h, dtype=np.float64) / 100.0, 0.0, 1.0)) * (MEDIUM_PROFILE_COUNT - 1)
        return np.interp(f, np.arange(MEDIUM_PROFILE_COUNT), self.values)


def tabulated_phase(phase: Callable[[np.ndarray], np.ndarray]) -> PhaseTable:
    """位相関数を θ = π·u²（2,049 点）の表にする（Cornette-Shanks も同じ表で扱うため）。float32 に丸める。"""

    if isinstance(phase, PhaseTable) and phase.values.shape[0] == MEDIUM_PHASE_COUNT:
        values = phase.values
    else:
        u = np.linspace(0.0, 1.0, MEDIUM_PHASE_COUNT)
        values = phase(np.cos(np.pi * u * u))
    return PhaseTable(np.asarray(values, dtype=np.float32).astype(np.float64))


def tabulated(atmo: Atmosphere) -> Atmosphere:
    """Mie の成分の鉛直分布と位相関数を、GPU と同じ表に置き換えた大気。係数は float32 に丸める。"""

    comps = [MieComponent(c.name, np.float32(c.scattering).astype(np.float64), np.float32(c.absorption).astype(np.float64),
                          TabulatedProfile(c.profile), tabulated_phase(c.phase)) for c in atmo.mie]
    return Atmosphere(atmo.rayleigh, atmo.ozone, comps)


def phase_cdf(values: np.ndarray) -> np.ndarray:
    """表の区間ごとの確率（台形則、x = 1 − cosθ で積分）の累積。形状 (U, 3)、先頭 0・末尾 1。"""

    u = np.linspace(0.0, 1.0, MEDIUM_PHASE_COUNT)
    x = 2.0 * np.sin(0.5 * np.pi * u * u) ** 2
    dx = np.diff(x)[:, None]
    prob = 2.0 * np.pi * 0.5 * (values[1:] + values[:-1]) * dx
    cdf = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(prob, axis=0)], axis=0)
    return cdf / cdf[-1]


def export_medium(atmo: Atmosphere) -> np.ndarray:
    """表の媒質を float32 の並びにする（PathTrace.hlsl の説明と同じ並び）。atmo は tabulated() の結果であること。"""

    if len(atmo.mie) > MEDIUM_MAX_COMPONENTS:
        raise ValueError("too many Mie components")
    header = np.zeros(MEDIUM_HEADER_FLOATS)
    header[0] = len(atmo.mie)
    blocks = [header]
    for c in atmo.mie:
        assert isinstance(c.profile, TabulatedProfile) and isinstance(c.phase, PhaseTable)
        head = np.zeros(8)
        head[0:3] = c.scattering
        head[3] = c.profile.values.max()
        head[4:7] = c.absorption
        phase = c.phase.values                                               # (U, 3)
        blocks += [head, c.profile.values, phase.T.ravel(), phase_cdf(phase).T.ravel()]
    return np.concatenate(blocks).astype(np.float32)


# ---------------------------------------------------------------------- 視点
@dataclass(frozen=True)
class Scenario:
    name: str
    height_km: float
    sun_zenith_deg: float
    front_pitch_deg: float


SCENARIOS = (
    Scenario("noon", 0.2, 45.0, 10.0),
    Scenario("low_sun", 0.2, 80.0, 5.0),
    Scenario("sunset", 0.2, 88.0, 3.0),
    Scenario("altitude_3km", 3.0, 80.0, -3.0),
)


def perspective_dirs(yaw_deg: float, pitch_deg: float, vfov_deg: float, w: int, h: int) -> np.ndarray:
    """+Y が上。yaw 0 で +X を向く（太陽の方位を 0 に置く）。"""

    tan = np.tan(np.radians(vfov_deg) * 0.5)
    x = ((np.arange(w) + 0.5) / w * 2 - 1) * tan * w / h
    y = (1 - (np.arange(h) + 0.5) / h * 2) * tan
    yy, xx = np.meshgrid(y, x, indexing="ij")
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.sin(pitch), np.cos(pitch) * np.sin(yaw)])
    right = np.array([-np.sin(yaw), 0.0, np.cos(yaw)])
    up = np.cross(right, fwd)
    d = fwd + xx[..., None] * right + yy[..., None] * up
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)


def views(sc: Scenario, scale: int):
    sun_elev = 90.0 - sc.sun_zenith_deg
    w, h = 64 * scale, 36 * scale
    return {
        "front": (perspective_dirs(0.0, sc.front_pitch_deg, 50.0, w, h), (h, w)),
        "back": (perspective_dirs(180.0, sc.front_pitch_deg, 50.0, w, h), (h, w)),
        # 太陽の周り ±6°
        "closeup": (perspective_dirs(0.0, sun_elev, 12.0, 24 * scale, 24 * scale), (24 * scale, 24 * scale)),
    }


def sun_vector(zenith_deg: float) -> np.ndarray:
    th = np.radians(zenith_deg)
    return np.array([np.sin(th), np.cos(th), 0.0])


# ---------------------------------------------------------------------- 表示（GT7）と比較
SDR = tone_mapper("sdr")
HDR = tone_mapper("hdr1000")


def to_display(per_lux: np.ndarray, exposure: float):
    """単位照度あたりの輝度（Rec.709）→ fb → Rec.2020 → GT7。戻り値: (SDR [nit], HDR [nit], sRGB 0〜1)。"""

    fb = per_lux * exposure
    fb2020 = np.maximum(fb @ REC709_TO_REC2020.T, 0.0)
    sdr = SDR.apply(fb2020).astype(np.float64)                 # 0〜1（1 = 250 nit）
    hdr = HDR.apply(fb2020).astype(np.float64)                 # fb（1 = 100 nit）
    lin709 = np.clip(sdr @ REC2020_TO_REC709.T, 0.0, 1.0)
    srgb = np.where(lin709 <= 0.0031308, 12.92 * lin709, 1.055 * np.power(lin709, 1 / 2.4) - 0.055)
    return sdr * 250.0, hdr * 100.0, srgb


def luminance(rgb709: np.ndarray) -> np.ndarray:
    return rgb709 @ np.array([0.2126, 0.7152, 0.0722])


def exposure_for(per_lux: np.ndarray, target_fb: float = 0.6) -> float:
    """単位照度あたりの輝度 → fb の係数（= E_sun × 露出 k）。中央値の輝度が fb = target_fb になるように選ぶ。
    固定露出では、どの変種にも同じ係数を掛けるので、E_sun の値は比較に影響しない。"""

    med = float(np.median(luminance(per_lux)))
    return target_fb / max(med, 1e-12)


def de_stats(a: np.ndarray, b: np.ndarray) -> dict:
    de = delta_e_itp(a, b)
    return {"mean": float(de.mean()), "p95": float(np.percentile(de, 95)), "max": float(de.max()),
            "frac_gt_3": float(np.mean(de > 3.0)), "frac_gt_1": float(np.mean(de > 1.0))}


def heat(de: np.ndarray) -> np.ndarray:
    """ΔE_ITP を色にする: 0 = 黒、3 = 黄、10 以上 = 白。"""

    x = np.clip(de / 10.0, 0.0, 1.0)
    return np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 0.6, 0, 1), np.clip(3 * x - 1.8, 0, 1)], axis=-1)


# ---------------------------------------------------------------------- 実行
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "dumps" / "aerosol_experiment")
    parser.add_argument("--scale", type=int, default=5, help="画像の大きさ（front/back は 64×36 の倍数）")
    parser.add_argument("--steps", type=int, default=96, help="視線上の点の数")
    parser.add_argument("--scenarios", nargs="*", default=[s.name for s in SCENARIOS])
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    coeff = json.loads((ROOT / "data" / "atmosphere_coefficients.json").read_text(encoding="utf-8"))
    hillaire = next(s for s in coeff["sets"] if s["name"] == "Hillaire2020")
    rayleigh = np.asarray(hillaire["rayleigh_scattering_per_km"])
    ozone = np.asarray(hillaire["ozone_absorption_per_km"])

    t0 = time.time()
    fine = mode_optics(FINE, 2049)
    coarse = mode_optics(COARSE, 2049)
    optics_report = {}
    for name, o in (("fine", fine), ("coarse", coarse)):
        ang = -np.log(o.c_ext[2] / o.c_ext[0]) / np.log(WAVELENGTHS_UM[2] / WAVELENGTHS_UM[0])
        u = np.linspace(0, 1, o.phase.shape[0])
        sample = {f"{deg}deg": [float(np.interp(np.sqrt(np.radians(deg) / np.pi), u, o.phase[:, c])) for c in range(3)]
                  for deg in (0, 0.5, 1, 3, 10, 30, 90, 150, 180)}
        optics_report[name] = {"g_rgb": o.g.tolist(), "ssa_rgb": o.ssa.tolist(), "angstrom_440_680": float(ang),
                               "c_ext_ratio_rgb_over_g": (o.c_ext / o.c_ext[G_CHANNEL]).tolist(), "phase_sr-1": sample}
    cs = cornette_shanks(BASE_G)
    optics_report["cornette_shanks_g0.8"] = {f"{deg}deg": float(cs(np.cos(np.radians(deg)))[0])
                                             for deg in (0, 0.5, 1, 3, 10, 30, 90, 150, 180)}
    print(f"Mie optics: {time.time() - t0:.1f} s")

    variants = build_variants(rayleigh, ozone, fine, coarse)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    report = {"assumptions": {"fine": vars(FINE) | {"refractive_index": str(FINE.refractive_index)},
                              "coarse": vars(COARSE) | {"refractive_index": str(COARSE.refractive_index)},
                              "aod550_realistic": AOD550_REALISTIC, "fine_fraction_of_aod": FINE_FRACTION_OF_AOD,
                              "wavelengths_um": WAVELENGTHS_UM.tolist(), "steps": args.steps},
              "optics": optics_report,
              "aod_rgb": {k: v.aod().tolist() for k, v in variants.items()}, "scenarios": {}}
    for k, v in variants.items():
        print(f"  AOD {k:<9} " + " ".join(f"{x:.4f}" for x in v.aod()))

    tables = {}
    for name, atmo in variants.items():
        t1 = time.time()
        tt = TransmittanceTable(atmo)
        tables[name] = (tt, MultiScatteringTable(atmo, tt))
        print(f"tables {name}: {time.time() - t1:.1f} s")

    for sc in (s for s in SCENARIOS if s.name in args.scenarios):
        origin = np.array([0.0, BOTTOM_KM + sc.height_km, 0.0])
        sun = sun_vector(sc.sun_zenith_deg)
        renders = {}
        for vname, atmo in variants.items():
            tt, psi = tables[vname]
            for view, (dirs, shape) in views(sc, args.scale).items():
                t1 = time.time()
                lum, _ = integrate(atmo, tt, psi, origin, dirs, sun, args.steps, GROUND_ALBEDO_SURFACE)
                renders[(vname, view)] = lum.reshape(shape + (3,))
            print(f"{sc.name} {vname}: {time.time() - t1:.1f} s (last view)")
        np.savez_compressed(args.out / f"{sc.name}{args.tag}.npz",
                            **{f"{v}__{w}": img for (v, w), img in renders.items()})
        # 露出: (1) シナリオで固定（base の正面の中央値）、(2) 変種ごと（その変種の正面の中央値）
        ref_name = "base" if "base" in variants else next(iter(variants))
        fixed = exposure_for(renders[(ref_name, "front")])
        sc_report = {"fixed_exposure_fb_per_perlux": fixed, "pairs": {}}
        disp = {}
        for (vname, view), img in renders.items():
            auto = exposure_for(renders[(vname, "front")])
            disp[(vname, view, "fixed")] = to_display(img, fixed)
            disp[(vname, view, "auto")] = to_display(img, auto)
        pairs = [("base", "base+A"), ("base", "base+B"), ("base", "base+A+B"), ("base", "C"), ("C", "C+A"),
                 ("C", "C+B"), ("C", "C+D"), ("C", "C+A+B"), ("C", "C+A+B+D"), ("base", "C+A+B+D")]
        for a, b in pairs:
            if a not in variants or b not in variants:
                continue
            for view in ("front", "back", "closeup"):
                for expo in ("fixed", "auto"):
                    da, db = disp[(a, view, expo)], disp[(b, view, expo)]
                    key = f"{a} -> {b} | {view} | {expo}"
                    sc_report["pairs"][key] = {"sdr": de_stats(da[0], db[0]), "hdr1000": de_stats(da[1], db[1])}
        report["scenarios"][sc.name] = sc_report
        # 画像: 行 = 変種、列 = 正面・背面・太陽の周り（SDR、露出は固定）+ C との ΔE（SDR、自動露出）
        rows = []
        for vname in variants:
            tiles = []
            for view in ("front", "back", "closeup"):
                tiles += [disp[(vname, view, "fixed")][2], np.ones((disp[(vname, view, "fixed")][2].shape[0], 3, 3))]
            ref = "C" if vname.startswith("C") and vname != "C" else "base"
            de = delta_e_itp(disp[(ref, "front", "auto")][0], disp[(vname, "front", "auto")][0]) \
                if vname != ref and ref in variants else np.zeros(disp[(vname, "front", "auto")][0].shape[:2])
            tiles.append(heat(de))
            h_max = max(t.shape[0] for t in tiles)
            tiles = [np.pad(t, ((0, h_max - t.shape[0]), (0, 0), (0, 0)), constant_values=1.0) for t in tiles]
            rows += [np.concatenate(tiles, axis=1), np.ones((3, sum(t.shape[1] for t in tiles), 3))]
        write_png(args.out / f"{sc.name}{args.tag}.png", np.concatenate(rows[:-1]))
        print(f"{sc.name} done")

    (args.out / f"report{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print_summary(report)
    return 0


def print_summary(report: dict) -> None:
    print("\nΔE_ITP (SDR GT7, mean / p95 / >3 の画素の割合)  [fixed exposure | auto exposure]")
    for sc, data in report["scenarios"].items():
        print(f"== {sc}")
        keys = sorted({k.rsplit(" | ", 1)[0] for k in data["pairs"]}, key=lambda s: list(data["pairs"]).index(s + " | fixed"))
        for k in keys:
            f, a = data["pairs"][k + " | fixed"]["sdr"], data["pairs"][k + " | auto"]["sdr"]
            print(f"  {k:<32} {f['mean']:6.2f} {f['p95']:6.2f} {f['frac_gt_3']:5.0%} | "
                  f"{a['mean']:6.2f} {a['p95']:6.2f} {a['frac_gt_3']:5.0%}")


if __name__ == "__main__":
    sys.exit(main())
