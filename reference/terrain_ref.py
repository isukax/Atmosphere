"""地形・テスト球の幾何のリファレンス（Phase 4 の調査、Phase 4.5 の V3e / V3f）。

  - 地形の高さ: src/scene/Meshes.cpp の terrainHeightM を移植し、メッシュの格子点（513²）で評価して、
    メッシュと同じ三角形分割（各セルの対角線は (i+1, j)–(i, j+1)）で補間する。
  - 太陽の遮蔽: 点から太陽方向へ進み、地形の下に入るか、球に当たれば影（レイキャスト）。
  - 視線の復元: 深度ダンプ（Reverse-Z）とカメラ（位置・ヨー・ピッチ・画角）から、ピクセルの視線と地形の点。
  - 空気遠近法の単散乱: 視線上の点ごとに太陽の遮蔽を調べ、影に入る分を分けて積分する。
平面の世界で計算する（20 km で惑星の曲率による高度差 31 m、design.md §5.3）。
"""

from __future__ import annotations

import numpy as np

from atmosphere_ref import Atmosphere, cornette_shanks_phase, rayleigh_phase

# Meshes.h の TerrainSettings の既定値
TERRAIN_SIZE_M = 40000.0
TERRAIN_CELLS = 512
TERRAIN_MAX_HEIGHT_M = 3000.0
FLAT_RADIUS_M = 2000.0
BLEND_RADIUS_M = 8000.0


# ---------------------------------------------------------------------- 地形（Meshes.cpp の移植）
def hash2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = 0xFFFFFFFF
    h = ((x.astype(np.int64) & m) * 0x8DA6B343 & m) ^ ((y.astype(np.int64) & m) * 0xD8163841 & m)
    h ^= h >> 13
    h = (h * 0x85EBCA6B) & m
    h ^= h >> 16
    return h


def value_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    fx, fy = np.floor(x), np.floor(y)
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)

    def fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def lattice(a, b):
        return (hash2(a, b) & 0xFFFFFF) / 16777215.0

    u, v = fade(x - fx), fade(y - fy)
    a, b = lattice(ix, iy), lattice(ix + 1, iy)
    c, d = lattice(ix, iy + 1), lattice(ix + 1, iy + 1)
    return (a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v


def fbm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    total, amplitude, norm = 0.0, 0.5, 0.0
    for _ in range(6):
        total = total + amplitude * value_noise(x, y)
        norm += amplitude
        x, y, amplitude = x * 2.03, y * 2.03, amplitude * 0.5
    return total / norm


def terrain_height_m(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    t = np.clip((np.sqrt(x * x + z * z) - FLAT_RADIUS_M) / (BLEND_RADIUS_M - FLAT_RADIUS_M), 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    n = fbm(x / 9000.0 + 17.3, z / 9000.0 - 4.1)
    return TERRAIN_MAX_HEIGHT_M * n * n * mask


class HeightGrid:
    """メッシュの格子点の高さ（513²）を、メッシュと同じ三角形で補間して引く。格子の外は −∞（遮蔽なし）。"""

    def __init__(self):
        n = TERRAIN_CELLS + 1
        self.step = TERRAIN_SIZE_M / TERRAIN_CELLS
        self.origin = -0.5 * TERRAIN_SIZE_M
        coords = self.origin + self.step * np.arange(n)
        zz, xx = np.meshgrid(coords, coords, indexing="ij")
        self.h = terrain_height_m(xx, zz)  # [z, x]
        self.n = n

    def __call__(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        gx, gz = (x - self.origin) / self.step, (z - self.origin) / self.step
        inside = (gx >= 0) & (gx <= self.n - 1) & (gz >= 0) & (gz <= self.n - 1)
        gx, gz = np.clip(gx, 0, self.n - 1.001), np.clip(gz, 0, self.n - 1.001)
        ix, iz = gx.astype(int), gz.astype(int)
        fx, fz = gx - ix, gz - iz
        h = self.h
        a, b = h[iz, ix], h[iz, ix + 1]          # (i, j), (i+1, j)
        c, d = h[iz + 1, ix], h[iz + 1, ix + 1]  # (i, j+1), (i+1, j+1)
        # Meshes.cpp: 三角形 (a, c, b) と (b, c, d)。対角線 b–c で分ける
        lower = a + (b - a) * fx + (c - a) * fz
        upper = d + (c - d) * (1.0 - fx) + (b - d) * (1.0 - fz)
        v = np.where(fx + fz <= 1.0, lower, upper)
        return np.where(inside, v, -np.inf)


# ---------------------------------------------------------------------- 遮蔽
def ray_hits_spheres(points_m: np.ndarray, direction: np.ndarray, spheres: list[dict]) -> np.ndarray:
    """点から方向 direction（単位ベクトル）の半直線が、どれかの球に t > 0 で当たるか。"""

    hit = np.zeros(len(points_m), dtype=bool)
    for s in spheres:
        c = np.asarray(s["center_m"], dtype=np.float64)
        r = float(s["radius_m"])
        oc = points_m - c
        b = oc @ direction
        disc = b * b - (np.einsum("ij,ij->i", oc, oc) - r * r)
        t_far = -b + np.sqrt(np.maximum(disc, 0.0))
        hit |= (disc > 0.0) & (t_far > 1e-3)
    return hit


def in_shadow(points_m: np.ndarray, sun: np.ndarray, grid: HeightGrid, spheres: list[dict] | None = None,
              step_m: float = 20.0, chunk: int = 200000) -> np.ndarray:
    """点（形状 (M, 3) [m]）から太陽方向へ進み、地形の下に入るか球に当たれば影（True）。"""

    shadow = np.zeros(len(points_m), dtype=bool)
    max_steps = int(np.ceil(TERRAIN_MAX_HEIGHT_M / max(sun[1], 1e-3) / step_m)) + 1
    for c0 in range(0, len(points_m), chunk):
        pts = points_m[c0:c0 + chunk]
        sub = np.zeros(len(pts), dtype=bool)
        if spheres:
            sub |= ray_hits_spheres(pts, sun, spheres)
        active = np.nonzero(~sub)[0]
        for k in range(1, max_steps + 1):
            p = pts[active] + sun * (k * step_m)
            hit = p[:, 1] < grid(p[:, 0], p[:, 2])
            sub[active[hit]] = True
            # 影が決まった点、高さが最大標高を超えた点、地形の外へ出た点はやめる
            out = (np.abs(p[:, 0]) > 0.5 * TERRAIN_SIZE_M) | (np.abs(p[:, 2]) > 0.5 * TERRAIN_SIZE_M)
            active = active[~hit & (p[:, 1] <= TERRAIN_MAX_HEIGHT_M) & ~out]
            if active.size == 0:
                break
        shadow[c0:c0 + chunk] = sub
    return shadow


# ---------------------------------------------------------------------- 視線の復元
def camera_basis(yaw_deg: float, pitch_deg: float):
    """src/scene/Camera.cpp と同じ: yaw = 0 で −Z を向く、上向きが正のピッチ。"""

    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def reconstruct(depth_img: np.ndarray, near_m: float, camera_m: np.ndarray, yaw_deg: float, pitch_deg: float,
                fov_deg: float, ys: np.ndarray, xs: np.ndarray):
    """ピクセル (ys, xs) の視線方向（単位）、カメラ → 地形の点の距離 [m]、地形の点 [m]。"""

    h, w = depth_img.shape[:2]
    forward, right, up = camera_basis(yaw_deg, pitch_deg)
    tan_half = np.tan(0.5 * np.radians(fov_deg))
    ndc_x = (xs + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - (ys + 0.5) / h * 2.0
    view_depth_m = near_m / depth_img[ys, xs, 0].astype(np.float64)
    ray = (right[None, :] * (ndc_x * tan_half * w / h)[:, None] + up[None, :] * (ndc_y * tan_half)[:, None]
           + forward[None, :])  # ビュー空間の奥行き 1 あたり
    hit = camera_m + ray * view_depth_m[:, None]
    distance_m = np.linalg.norm(ray, axis=1) * view_depth_m
    return ray / np.linalg.norm(ray, axis=1, keepdims=True), distance_m, hit


def surface_normals(hit_m: np.ndarray, grid: HeightGrid, spheres: list[dict]) -> np.ndarray:
    """地形の点の法線。球の表面の点は球の法線、それ以外は地形の三角形の法線（数値微分）。"""

    e = 0.25
    hx = (grid(hit_m[:, 0] + e, hit_m[:, 2]) - grid(hit_m[:, 0] - e, hit_m[:, 2])) / (2 * e)
    hz = (grid(hit_m[:, 0], hit_m[:, 2] + e) - grid(hit_m[:, 0], hit_m[:, 2] - e)) / (2 * e)
    n = np.stack([-hx, np.ones_like(hx), -hz], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    for s in spheres:
        c = np.asarray(s["center_m"], dtype=np.float64)
        r = float(s["radius_m"])
        d = hit_m - c
        dist = np.linalg.norm(d, axis=1)
        on = np.abs(dist - r) < 0.01 * r + 0.05
        n[on] = d[on] / dist[on, None]
    return n


# ---------------------------------------------------------------------- 空気遠近法の単散乱
def aerial_single_scattering(atmo: Atmosphere, camera_m: np.ndarray, direction: np.ndarray, distance_m: np.ndarray,
                             sun: np.ndarray, grid: HeightGrid, spheres: list[dict] | None, samples: int = 64,
                             chunk_pixels: int = 1024):
    """カメラ → 地形の点の単散乱（Rayleigh + Mie、単位照度あたり [sr⁻¹]）を、全体と「地形・球の影に入る分」に分けて返す。
    戻り値: (total (N, 3), shadowed (N, 3), shadow_fraction_of_points (N,))"""

    frac = (np.arange(samples) + 0.5) / samples
    n = len(distance_m)
    total, shadowed = np.zeros((n, 3)), np.zeros((n, 3))
    fraction = np.zeros(n)
    for c0 in range(0, n, chunk_pixels):
        c = slice(c0, min(c0 + chunk_pixels, n))
        t_m = distance_m[c, None] * frac[None, :]
        points = camera_m + direction[c, None, :] * t_m[..., None]
        altitude_km = np.maximum(points[..., 1], 0.0) / 1000.0
        dt_km = (distance_m[c] / samples / 1000.0)[:, None]
        sigma_t = atmo.extinction(altitude_km)
        optical = np.cumsum(sigma_t * dt_km[..., None], axis=1) - 0.5 * sigma_t * dt_km[..., None]
        t_view = np.exp(-optical)
        planet = np.stack([points[..., 0] / 1000.0, atmo.bottom_radius_km + altitude_km, points[..., 2] / 1000.0],
                          axis=-1)
        t_sun = atmo.sun_transmittance(planet.reshape(-1, 3), sun, panels=16, order=4).reshape(points.shape)
        rho_r, rho_m, _ = atmo.densities(altitude_km)
        cos_theta = direction[c] @ sun
        integrand = (rho_m[..., None] * atmo.mie_scattering_per_km
                     * cornette_shanks_phase(atmo.mie_phase_g, cos_theta)[:, None, None]
                     + rho_r[..., None] * atmo.rayleigh_scattering_per_km * rayleigh_phase(cos_theta)[:, None, None]
                     ) * t_view * t_sun * dt_km[..., None]
        shadow = in_shadow(points.reshape(-1, 3), sun, grid, spheres).reshape(points.shape[:2])
        total[c] = integrand.sum(1)
        shadowed[c] = (integrand * shadow[..., None]).sum(1)
        fraction[c] = shadow.mean(1)
    return total, shadowed, fraction
