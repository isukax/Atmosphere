"""大気散乱の倍精度リファレンス（RGB、GPU と同じ媒質モデル）。

単位はシェーダーと同じ: 距離 km（惑星中心が原点）、係数 km⁻¹、輝度は単位照度あたり [sr⁻¹]。

積分は複合 Gauss-Legendre 求積で行う（区間を panels 個に分け、各区間に order 点）。
被積分関数は指数関数の組み合わせで滑らかだが、オゾンのテント関数は高度 10 / 25 / 40 km で
折れ曲がる（微分が不連続）。折れ目をまたぐ区間があると求積の収束が O(h²) に落ちるので、
光線がその高度の球面を横切る点で積分区間を分割する（kink_breakpoints）。
精度は区間数を倍にした結果との差で確認する（validate_phase2.py）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Atmosphere:
    bottom_radius_km: float
    top_radius_km: float
    rayleigh_scattering_per_km: np.ndarray  # (3,)
    rayleigh_scale_height_km: float
    mie_scattering_per_km: np.ndarray       # (3,)
    mie_extinction_per_km: np.ndarray       # (3,)
    mie_scale_height_km: float
    mie_phase_g: float
    ozone_absorption_per_km: np.ndarray     # (3,)
    ozone_center_km: float
    ozone_half_width_km: float

    @classmethod
    def from_meta(cls, meta: dict) -> "Atmosphere":
        """アプリが書いたサイドカーの "atmosphere"（AtmosphereParams の JSON）から作る。"""

        a = meta["atmosphere"]
        mie_scattering = np.asarray(a["mie_scattering_per_km"], dtype=np.float64)
        return cls(
            bottom_radius_km=float(a["bottom_radius_km"]),
            top_radius_km=float(a["top_radius_km"]),
            rayleigh_scattering_per_km=np.asarray(a["rayleigh_scattering_per_km"], dtype=np.float64),
            rayleigh_scale_height_km=float(a["rayleigh_scale_height_km"]),
            mie_scattering_per_km=mie_scattering,
            mie_extinction_per_km=mie_scattering + np.asarray(a["mie_absorption_per_km"], dtype=np.float64),
            mie_scale_height_km=float(a["mie_scale_height_km"]),
            mie_phase_g=float(a["mie_phase_g"]),
            ozone_absorption_per_km=np.asarray(a["ozone_absorption_per_km"], dtype=np.float64),
            ozone_center_km=float(a["ozone_center_km"]),
            ozone_half_width_km=float(a["ozone_half_width_km"]),
        )

    # ------------------------------------------------------------------ 媒質
    def densities(self, altitude_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.asarray(altitude_km, dtype=np.float64)
        rayleigh = np.exp(-h / self.rayleigh_scale_height_km)
        mie = np.exp(-h / self.mie_scale_height_km)
        ozone = np.maximum(0.0, 1.0 - np.abs(h - self.ozone_center_km) / self.ozone_half_width_km)
        return rayleigh, mie, ozone

    def extinction(self, altitude_km: np.ndarray) -> np.ndarray:
        """σt(h) [km⁻¹]、形状 (..., 3)。"""

        r, m, o = self.densities(altitude_km)
        return (r[..., None] * self.rayleigh_scattering_per_km + m[..., None] * self.mie_extinction_per_km
                + o[..., None] * self.ozone_absorption_per_km)

    # ------------------------------------------------------------------ 幾何
    def distance_to_top(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        disc = r * r * (mu * mu - 1.0) + self.top_radius_km ** 2
        return np.maximum(0.0, -r * mu + np.sqrt(np.maximum(disc, 0.0)))

    def ray_intersects_ground(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return (mu < 0.0) & (r * r * (mu * mu - 1.0) + self.bottom_radius_km ** 2 >= 0.0)

    def kink_breakpoints(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """光線（半径 r、天頂角の余弦 mu）がオゾン分布の折れ目の高度を横切る距離 t、形状 (M, 6)。
        横切らない分は NaN。半径 ρ の球面との交点は t = −r·mu ± √(r²mu² − r² + ρ²)。"""

        radii = self.bottom_radius_km + self.ozone_center_km + np.array(
            [-self.ozone_half_width_km, 0.0, self.ozone_half_width_km])
        disc = (r * mu)[:, None] ** 2 - r[:, None] ** 2 + radii[None, :] ** 2
        s = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
        return np.concatenate([-(r * mu)[:, None] - s, -(r * mu)[:, None] + s], axis=1)

    # ------------------------------------------------------------------ 透過率
    def optical_depth_to_top(self, r: np.ndarray, mu: np.ndarray, panels: int, order: int) -> np.ndarray:
        """(r, mu) から大気上端までの光学的厚さ、形状 (M, 3)。panels は折れ目で分けた各区間の分割数。"""

        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
        d = self.distance_to_top(r, mu)
        edges = segment_edges(np.zeros_like(d), d, self.kink_breakpoints(r, mu), panels)
        t, w = gauss_legendre_from_edges(edges, order)
        ri = np.sqrt(t * t + 2.0 * r[:, None] * mu[:, None] * t + r[:, None] ** 2)
        return np.einsum("mn,mnc->mc", w, self.extinction(ri - self.bottom_radius_km))

    def transmittance_to_top(self, r: np.ndarray, mu: np.ndarray, panels: int = 32, order: int = 4) -> np.ndarray:
        return np.exp(-self.optical_depth_to_top(r, mu, panels, order))

    # ------------------------------------------------------------------ Transmittance LUT のパラメタ化
    def transmittance_r_mu_to_uv(self, r: np.ndarray, mu: np.ndarray, width: int, height: int):
        """シェーダーの TransmittanceRMuToUv と同じ式（倍精度）。"""

        big_h = np.sqrt(self.top_radius_km ** 2 - self.bottom_radius_km ** 2)
        rho = np.sqrt(np.maximum(r * r - self.bottom_radius_km ** 2, 0.0))
        d = self.distance_to_top(r, mu)
        d_min = self.top_radius_km - r
        d_max = rho + big_h
        x_mu = (d - d_min) / (d_max - d_min)
        x_r = rho / big_h
        u = (x_mu + 0.5 / width) * (width / (width + 1.0))
        v = (x_r + 0.5 / height) * (height / (height + 1.0))
        return u, v

    def transmittance_uv_to_r_mu(self, u: np.ndarray, v: np.ndarray, width: int, height: int):
        """シェーダーの TransmittanceUvToRMu と同じ式（倍精度）。"""

        x_mu = (u - 0.5 / width) * (width / (width - 1.0))
        x_r = (v - 0.5 / height) * (height / (height - 1.0))
        big_h = np.sqrt(self.top_radius_km ** 2 - self.bottom_radius_km ** 2)
        rho = big_h * x_r
        r = np.sqrt(rho * rho + self.bottom_radius_km ** 2)
        d_min = self.top_radius_km - r
        d_max = rho + big_h
        d = d_min + x_mu * (d_max - d_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.where(d == 0.0, 1.0, (big_h * big_h - rho * rho - d * d) / (2.0 * r * d))
        return r, np.clip(mu, -1.0, 1.0)

    # ------------------------------------------------------------------ 単散乱
    def single_scattering_per_lux(self, camera_km: np.ndarray, directions: np.ndarray, sun_dir: np.ndarray,
                                  outer_panels: int = 32, inner_panels: int = 8, order: int = 4,
                                  chunk: int = 8) -> np.ndarray:
        """単位照度あたりの単散乱輝度 [sr⁻¹]、形状 (N, 3)。太陽は点光源、惑星の影あり、地表反射なし。"""

        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        camera_km = np.asarray(camera_km, dtype=np.float64)
        sun_dir = np.asarray(sun_dir, dtype=np.float64)
        out = np.zeros((directions.shape[0], 3))
        for start in range(0, directions.shape[0], chunk):
            out[start:start + chunk] = self._single_scattering_chunk(
                camera_km, directions[start:start + chunk], sun_dir, outer_panels, inner_panels, order)
        return out

    def _single_scattering_chunk(self, c, dirs, sun, outer_panels, inner_panels, order):
        n = dirs.shape[0]
        t_start, t_end = np.zeros(n), np.zeros(n)
        for i, d in enumerate(dirs):
            top = ray_sphere(c, d, self.top_radius_km)
            ground = ray_sphere(c, d, self.bottom_radius_km)
            t_start[i] = max(top[0], 0.0)
            t_end[i] = top[1]
            if ground is not None and ground[0] > 0.0:
                t_end[i] = min(t_end[i], ground[0])

        # 視線上の求積点（オゾンの折れ目で区間を分割）
        cam_r = np.full(n, np.linalg.norm(c))
        cam_mu = (dirs @ c) / cam_r
        edges = segment_edges(t_start, t_end, self.kink_breakpoints(cam_r, cam_mu), outer_panels)
        outer_panels = edges.shape[1] - 1
        x, wx = np.polynomial.legendre.leggauss(order)
        half = 0.5 * (edges[:, 1:] - edges[:, :-1])                      # (n, P)
        mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
        t = (mid[:, :, None] + half[:, :, None] * x).reshape(n, -1)       # (n, P·order)
        w = (half[:, :, None] * wx).reshape(n, -1)

        def extinction_at(tt):  # tt: (n, M) の視線上の距離 → (n, M, 3)
            p = c + tt[..., None] * dirs[:, None, :]
            return self.extinction(np.linalg.norm(p, axis=-1) - self.bottom_radius_km)

        # カメラ → 求積点の光学的厚さ: 区間全体の積分を累積 + 区間内の部分積分（区間先頭 → 点）
        ext_nodes = extinction_at(t).reshape(n, outer_panels, order, 3)
        panel_depth = np.einsum("npk,npkc->npc", (half[:, :, None] * wx), ext_nodes)
        depth_before = np.concatenate([np.zeros((n, 1, 3)), np.cumsum(panel_depth, axis=1)[:, :-1]], axis=1)
        t_nodes = t.reshape(n, outer_panels, order)
        sub_half = 0.5 * (t_nodes - edges[:, :-1, None])                   # (n, P, order)
        sub_mid = 0.5 * (t_nodes + edges[:, :-1, None])
        sub_t = sub_mid[..., None] + sub_half[..., None] * x               # (n, P, order, order)
        sub_ext = extinction_at(sub_t.reshape(n, -1)).reshape(n, outer_panels, order, order, 3)
        partial = np.einsum("npk,npkj,npkjc->npkc", sub_half, np.broadcast_to(wx, sub_half.shape + (order,)), sub_ext)
        view_depth = (depth_before[:, :, None, :] + partial).reshape(n, -1, 3)

        # 求積点での散乱と太陽方向の透過率
        p = c + t[..., None] * dirs[:, None, :]
        r = np.linalg.norm(p, axis=-1)
        altitude = r - self.bottom_radius_km
        rayleigh_d, mie_d, _ = self.densities(altitude)
        cos_theta = dirs @ sun
        phase_r = rayleigh_phase(cos_theta)[:, None]
        phase_m = cornette_shanks_phase(self.mie_phase_g, cos_theta)[:, None]
        scattering = (rayleigh_d[..., None] * self.rayleigh_scattering_per_km * phase_r[..., None]
                      + mie_d[..., None] * self.mie_scattering_per_km * phase_m[..., None])
        mu_sun = (p @ sun) / r
        shadow = self.ray_intersects_ground(r, mu_sun)
        sun_t = self.transmittance_to_top(r.ravel(), mu_sun.ravel(), inner_panels, order).reshape(r.shape + (3,))
        sun_t[shadow] = 0.0
        return np.einsum("nm,nmc->nc", w, np.exp(-view_depth) * scattering * sun_t)


# ---------------------------------------------------------------------- 補助関数

def segment_edges(start: np.ndarray, end: np.ndarray, breakpoints: np.ndarray, panels_per_segment: int) -> np.ndarray:
    """[start, end] を breakpoints（NaN は無視）で区切り、各区間を panels_per_segment 等分した端点、
    形状 (M, (B+1)·k + 1)。区間外・NaN の点は start に寄せる（長さ 0 の区間になり重み 0）。"""

    b = np.where(np.isnan(breakpoints), start[:, None], breakpoints)
    b = np.clip(b, start[:, None], end[:, None])
    knots = np.sort(np.concatenate([start[:, None], b, end[:, None]], axis=1), axis=1)
    frac = np.linspace(0.0, 1.0, panels_per_segment + 1)[:-1]
    inner = knots[:, :-1, None] + (knots[:, 1:] - knots[:, :-1])[:, :, None] * frac
    return np.concatenate([inner.reshape(start.shape[0], -1), end[:, None]], axis=1)


def gauss_legendre_from_edges(edges: np.ndarray, order: int):
    """区間端点 (M, P+1) から、各区間 order 点の Gauss-Legendre 求積点と重み、形状 (M, P·order)。"""

    x, wx = np.polynomial.legendre.leggauss(order)
    half = 0.5 * (edges[:, 1:] - edges[:, :-1])
    mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
    t = (mid[:, :, None] + half[:, :, None] * x).reshape(edges.shape[0], -1)
    w = (half[:, :, None] * wx).reshape(edges.shape[0], -1)
    return t, w


def ray_sphere(origin: np.ndarray, direction: np.ndarray, radius: float):
    """光線と原点中心の球の交点 (t0, t1)。交差しなければ None。"""

    b = float(origin @ direction)
    r = float(np.linalg.norm(origin))
    c = (r - radius) * (r + radius)
    disc = b * b - c
    if disc < 0.0:
        return None
    s = np.sqrt(disc)
    q = -(b + (s if b >= 0.0 else -s))
    ta, tb = q, (c / q if q != 0.0 else -b)
    return min(ta, tb), max(ta, tb)


def rayleigh_phase(cos_theta: np.ndarray) -> np.ndarray:
    return 3.0 / (16.0 * np.pi) * (1.0 + cos_theta ** 2)


def cornette_shanks_phase(g: float, cos_theta: np.ndarray) -> np.ndarray:
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    return 3.0 / (8.0 * np.pi) * (1.0 - g2) * (1.0 + cos_theta ** 2) / ((2.0 + g2) * denom * np.sqrt(denom))


def sun_direction(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    th, ph = np.radians(zenith_deg), np.radians(azimuth_deg)
    return np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])
