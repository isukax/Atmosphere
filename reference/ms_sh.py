"""多重散乱の近似（Phase 4.11）: l ≤ 2 の球面調和 + 閉じ方の地表の項（research.md §11.4 H の案 3 + 案 5）。

GPU の SH の Multi-scattering LUT（design.md §5.11 の予定）の Python のリファレンス。式は GPU と同じにする。

  表     : (h, μs) の 32×32（ともに等間隔。ae.MultiScatteringTable と同じ節点）。各点に、入ってくる散乱光の場
           （散乱 1 回以上）の SH の係数を 6 個 × RGB 持つ。
  局所の枠: z = 天頂（p/|p|）、x = 太陽を水平に射影した向き、y = z × x。太陽と天頂を含む面について鏡映対称なので
           cos(mφ) の項だけ（l ≤ 2 で 6 個）。
  基底   : 符号を固定した多項式（GPU と同じ。Condon-Shortley の位相は付けない）
             Y00 = 0.282095、Y10 = 0.488603·z、Y11 = 0.488603·x、
             Y20 = 0.315392(3z² − 1)、Y21 = 1.092548·x·z、Y22 = 0.546274(x² − y²)
           [POINT] 試作（scipy の lpmv）は Y11 と Y21 の符号が逆。描画の結果は同じだが、係数を比べるときは注意する。
  1 次の場: c₁_b = Σ_n w_n·Y_b(ω_n)·L₁(ω_n)。L₁ は太陽光の 1 回目の散乱（実際の位相関数）+ 地表での直達光の反射。
           方向は太陽中心の 1,024 方向（θ = π·u²、32 環 × 方位 32。compare_v4b_views.Option1MultiScatteringTable と同じ）。
  閉じ方 : c = (I − F)⁻¹·c₁（チャンネルごと）。
             F_ab = Σ_n w_n·Y_a·Y_b·Σ_k χ_{k,l(b)}·A_k(ω_n)、A_k = ∫ T·σs,k dt
             地表の項: 地面に当たる方向に (ρ/π)·T_g·e_b を足す（e_b = ∫_{z>0} Y_b·z dω、m = 0 の 3 つだけ 0 でない）。
           大気が天頂の周りに回転対称なので、F は m ごとのブロック（3×3、2×2、1×1）に分かれる。GPU と同じく
           既定はブロック対角で解く（block_diagonal=False で完全な 6×6）。
  描画   : 光源 J(v) = Σ_k σs,k Σ_b χ_{k,l(b)}·c_b(h, μs)·Y_b(v)（v は視線の向き、標本点の局所の枠）。クランプしない。
  χ      : Legendre のモーメント χ_l = 2π ∫ p(μ) P_l(μ) dμ（l = 0..2、RGB）。Rayleigh は (1, 0, 0.1)。
           位相関数の表は u（θ = π·u²）で台形則、Cornette-Shanks は閉じた式。χ₀ も測った値を使う（表の正規化の誤差を含める）。

反復（案 6）は入れない。ShMultiScatteringTable の gathers 引数が差し込み口（今は 0 以外で NotImplementedError）。
"""

from __future__ import annotations

import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerosol_experiment as ae  # noqa: E402
from aerosol_models import CornetteShanks, rayleigh_phase  # noqa: E402

COEFF_COUNT = 6
BAND_L = np.array([0, 1, 1, 2, 2, 2])          # 係数 b の l。並び (l, m) = (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)
BAND_M = np.array([0, 0, 1, 0, 1, 2])
BLOCKS = ((0, 1, 3), (2, 4), (5,))             # m = 0, 1, 2 のブロック
Y00 = 0.5 / math.sqrt(math.pi)                 # 0.282095
Y1 = math.sqrt(3.0 / (4.0 * math.pi))          # 0.488603
Y20 = 0.25 * math.sqrt(5.0 / math.pi)          # 0.315392
Y21 = 0.5 * math.sqrt(15.0 / math.pi)          # 1.092548
Y22 = 0.25 * math.sqrt(15.0 / math.pi)         # 0.546274
# e_b = ∫_{z>0} Y_b·z dω（地表の照度 E = Σ_b c_b·e_b）= (0.886227, 1.023327, 0, 0.495416, 0, 0)
HEMISPHERE_COSINE = np.array([math.pi * Y00, 2.0 * math.pi / 3.0 * Y1, 0.0, 0.5 * math.pi * Y20, 0.0, 0.0])
RAYLEIGH_MOMENTS = np.array([1.0, 0.0, 0.1])
TABLE_STEPS = 48
DEGENERATE_HORIZONTAL_SUN = 1e-7               # 太陽が天頂 / 天底: 局所の x が決まらないので m > 0 の項を 0 にする
DET_FALLBACK = 1e-4                            # ブロックの det(I − F) がこれ未満なら対角だけで閉じる（GPU と同じ）
DIAGONAL_FLOOR = 1e-3                          # 1 − F_aa の下限


# ---------------------------------------------------------------------- 基底
def basis(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """局所の枠での方向の成分 → (..., 6)"""

    x, y, z = np.broadcast_arrays(x, y, z)
    return np.stack([np.full(x.shape, Y00), Y1 * z, Y1 * x, Y20 * (3.0 * z * z - 1.0), Y21 * x * z,
                     Y22 * (x * x - y * y)], axis=-1)


def local_basis(points: np.ndarray, dirs: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """点ごとの局所の枠（z = 天頂、x = 太陽の水平の向き）で dirs の基底を評価する。points・dirs (..., 3) → (..., 6)。
    太陽が天頂・天底の点では m > 0 の項を 0 にする（その点の場は天頂の周りに回転対称）。"""

    up = points / np.linalg.norm(points, axis=-1, keepdims=True)
    xs = sun - (up @ sun)[..., None] * up
    n = np.linalg.norm(xs, axis=-1, keepdims=True)
    valid = n > DEGENERATE_HORIZONTAL_SUN
    xs = np.where(valid, xs / np.maximum(n, 1e-30), 0.0)
    ys = np.cross(up, xs)
    yb = basis((dirs * xs).sum(-1), (dirs * ys).sum(-1), (dirs * up).sum(-1))
    return np.where(valid | (BAND_M == 0), yb, 0.0)


# ---------------------------------------------------------------------- 方向の格子（表を作るとき）
def sun_grid(n_theta: int = 32, n_phi: int = 32):
    """太陽中心の格子（θ = π·u²、重みは立体角で和 = 4π）。compare_v4b_views.Option1MultiScatteringTable と同じ。
    返り値 make(sun) → (dirs (N, 3), weight (N,))。太陽中心の枠は x′ = Z × sun、y′ = sun × x′、z′ = sun
    （sun = (√(1−μ²), μ, 0) なら x′ = (−μ, √(1−μ²), 0)、y′ = +Z）。"""

    u0, u1 = np.arange(n_theta) / n_theta, (np.arange(n_theta) + 1) / n_theta
    theta = np.pi * (0.5 * (u0 + u1)) ** 2
    d_cos = np.cos(np.pi * u0 ** 2) - np.cos(np.pi * u1 ** 2)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1, 3)
    weight = np.repeat(d_cos[:, None], n_phi, axis=1).reshape(-1) * (2.0 * np.pi / n_phi)

    def make(sun: np.ndarray):
        x = np.cross([0.0, 0.0, 1.0], sun)
        x /= np.linalg.norm(x)
        return local @ np.stack([x, np.cross(sun, x), sun]), weight

    return make


# ---------------------------------------------------------------------- Legendre のモーメント χ
def legendre(lmax: int, x: np.ndarray) -> np.ndarray:
    p = np.empty((lmax + 1,) + np.shape(x))
    p[0] = 1.0
    if lmax >= 1:
        p[1] = x
    for l in range(1, lmax):
        p[l + 1] = ((2 * l + 1) * x * p[l] - l * p[l - 1]) / (l + 1)
    return p


def legendre_moments_cornette_shanks(g: float) -> np.ndarray:
    """Cornette-Shanks の χ₀..χ₂（閉じた式）。CS = 3/(2(2+g²))·(1+μ²)·HG で、HG の χ_l = g^l と
    μ²·P_l の Legendre 展開から:
      χ₁ = 3g(4+g²) / (5(2+g²))
      χ₂ = 3/(2(2+g²))·(2/15 + (32/21)g² + (12/35)g⁴)"""

    g2 = g * g
    return np.array([1.0, 3.0 * g * (4.0 + g2) / (5.0 * (2.0 + g2)),
                     1.5 / (2.0 + g2) * (2.0 / 15.0 + 32.0 / 21.0 * g2 + 12.0 / 35.0 * g2 * g2)])


def legendre_moments_numeric(phase, lmax: int = 2, samples: int = 400001) -> np.ndarray:
    """χ_l = 2π ∫ p(μ) P_l(μ) dμ（RGB）を、θ = π·u² で刻んだ u の台形則で求める → (lmax+1, 3)。
    位相関数の表（ae.PhaseTable、u で線形補間）は、この刻みで表の節点の間も細かく積む。"""

    u = np.linspace(0.0, 1.0, samples)
    theta = np.pi * u * u
    mu = np.cos(theta)
    dmu_du = np.sin(theta) * 2.0 * np.pi * u
    p = np.asarray(phase(mu))
    if p.ndim == 1:
        p = np.repeat(p[:, None], 3, axis=1)
    pl = legendre(lmax, mu)
    return 2.0 * np.pi * np.trapezoid(pl[:, :, None] * p[None] * dmu_du[None, :, None], u, axis=1)


def phase_moments(phase) -> np.ndarray:
    """位相関数の χ₀..χ₂（(3, 3) = (l, RGB)）。Cornette-Shanks は閉じた式、それ以外（表）は数値積分。"""

    if isinstance(phase, CornetteShanks):
        return np.repeat(legendre_moments_cornette_shanks(phase.g)[:, None], 3, axis=1)
    return legendre_moments_numeric(phase)


def medium_moments(atmo: ae.Atmosphere) -> np.ndarray:
    """成分ごとの χ（(K, 3, 3)、K = 0 が Rayleigh、以降 atmo.mie の順）"""

    return np.stack([np.repeat(RAYLEIGH_MOMENTS[:, None], 3, axis=1)] + [phase_moments(c.phase) for c in atmo.mie])


# ---------------------------------------------------------------------- 閉じ方
def solve_closure(F: np.ndarray, c1: np.ndarray, block_diagonal: bool = True):
    """c = (I − F)⁻¹·c₁。F (3, 6, 6) = (チャンネル, a, b)、c₁ (6, 3) → (c (6, 3), ブロックの det の最小値, 退避の有無)。
    block_diagonal なら m の違う要素を捨てて、m ごとのブロック（3×3、2×2、1×1）で解く（GPU と同じ）。
    det(I − F_block) < 1e-4 のブロックは対角だけで閉じる（1 − F_aa の下限 1e-3）。"""

    c = np.zeros_like(c1)
    min_det = np.inf
    fallback = False
    eye = np.eye(COEFF_COUNT)
    for ch in range(3):
        m = eye - F[ch]
        if not block_diagonal:
            min_det = min(min_det, float(np.linalg.det(m)))
            c[:, ch] = np.linalg.solve(m, c1[:, ch])
            continue
        for idx in BLOCKS:
            ix = np.ix_(idx, idx)
            det = float(np.linalg.det(m[ix]))
            min_det = min(min_det, det)
            if det < DET_FALLBACK:
                fallback = True
                c[list(idx), ch] = c1[list(idx), ch] / np.maximum(np.diag(m[ix]), DIAGONAL_FLOOR)
            else:
                c[list(idx), ch] = np.linalg.solve(m[ix], c1[list(idx), ch])
    return c, min_det, fallback


# ---------------------------------------------------------------------- 表の 1 点
def table_point(atmo: ae.Atmosphere, tt: ae.TransmittanceTable, chi: np.ndarray, h_km: float, mu_sun: float,
                ground_albedo: float, ground_term: bool, dirset, steps: int = TABLE_STEPS):
    """表の 1 点 (h, μs) の c₁ (6, 3) と F (3, 6, 6)。点は (0, r, 0)、太陽は (√(1−μs²), μs, 0)。
    SH の枠は z = +Y、x = +X、y = −Z（μs = ±1 でも退化しない。μs = ±1 では m > 0 の係数は対称性で 0 になる）。"""

    origin = np.array([0.0, ae.BOTTOM_KM + max(h_km, 0.001), 0.0])
    sun = np.array([math.sqrt(max(0.0, 1.0 - mu_sun * mu_sun)), mu_sun, 0.0])
    dirs, w = dirset(sun)
    t_end, hit = ae.ray_extent(origin, dirs)
    s = (np.arange(steps) + 0.5) / steps
    t = t_end[:, None] * s * s
    dt = t_end[:, None] * 2.0 * s / steps
    p = origin + t[..., None] * dirs[:, None, :]
    r = np.linalg.norm(p, axis=-1)
    s_r, s_m, ext = atmo.coefficients(r - ae.BOTTOM_KM)
    sig = [s_r] + list(s_m)
    seg = ext * dt[..., None]
    view_t = np.exp(-(np.cumsum(seg, axis=1) - 0.5 * seg))
    total_t = np.exp(-seg.sum(axis=1))

    # 1 回目の散乱（実際の位相関数）+ 地表での直達光の反射
    sun_t = tt.transmittance(r, (p @ sun) / r)
    cos = dirs @ sun
    phases = [np.repeat(rayleigh_phase(cos)[:, None], 3, axis=1)] + [c.phase(cos) for c in atmo.mie]
    src = sum(sg * ph[:, None, :] for sg, ph in zip(sig, phases))
    L = np.einsum("ns,nsc->nc", dt, view_t * src * sun_t)
    if ground_albedo > 0.0 and np.any(hit):
        gp = origin + t_end[hit, None] * dirs[hit]
        gr = np.linalg.norm(gp, axis=1)
        cos_sun = np.maximum(gp @ sun / gr, 0.0)
        L[hit] += total_t[hit] * tt.transmittance(gr, gp @ sun / gr) * cos_sun[:, None] * ground_albedo / np.pi

    yb = basis(dirs[:, 0], -dirs[:, 2], dirs[:, 1])                                   # (n, 6)
    c1 = np.einsum("n,nb,nc->bc", w, yb, L)
    A = np.stack([np.einsum("ns,nsc->nc", dt, view_t * sg) for sg in sig])            # (K, n, 3)
    B = np.einsum("kbc,knc->nbc", chi[:, BAND_L], A)                                  # (n, 6, 3)
    F = np.einsum("n,na,nb,nbc->cab", w, yb, yb, B)
    if ground_term and ground_albedo > 0.0:
        g = hit[:, None] * total_t * ground_albedo / np.pi                            # (n, 3)
        F = F + np.einsum("n,na,nc,b->cab", w, yb, g, HEMISPHERE_COSINE)
    return c1, F


_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    _CTX.clear()
    _CTX.update(ctx)
    _CTX["dirset"] = sun_grid(ctx["n_theta"], ctx["n_phi"])


def _table_row(i: int):
    c = _CTX
    return i, [table_point(c["atmo"], c["tt"], c["chi"], c["h"][i], mu, c["albedo"], c["ground_term"], c["dirset"],
                           c["steps"]) for mu in c["mu"]]


# ---------------------------------------------------------------------- 表
def _bilinear(h_nodes: np.ndarray, mu_nodes: np.ndarray, table: np.ndarray, h: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """table (H, M, ...) を節点 (h_nodes, mu_nodes) で双線形補間（端でクランプ）。GPU の双線形と同じ。"""

    def frac(nodes, x):
        x = np.clip(x, nodes[0], nodes[-1])
        i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, nodes.size - 2)
        return i, (x - nodes[i]) / (nodes[i + 1] - nodes[i])

    i, th = frac(h_nodes, h)
    j, tm = frac(mu_nodes, mu)
    extra = (None,) * (table.ndim - 2)
    th, tm = th[(...,) + extra], tm[(...,) + extra]
    t = table
    return (1 - th) * ((1 - tm) * t[i, j] + tm * t[i, j + 1]) + th * ((1 - tm) * t[i + 1, j] + tm * t[i + 1, j + 1])


class _ShSource:
    """係数の表 (H, M, 6, 3) から描画の光源 J を作る部分（Python で作った表と GPU のアトラスで共通）。"""

    h: np.ndarray
    mu: np.ndarray
    coefficients: np.ndarray
    chi: np.ndarray

    def lookup(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """(r, μs) → 係数 (..., 6, 3)"""

        return _bilinear(self.h, self.mu, self.coefficients, r - ae.BOTTOM_KM, mu)

    def source(self, p: np.ndarray, r: np.ndarray, mu_sun: np.ndarray, dirs: np.ndarray, sun: np.ndarray,
               sigmas: list[np.ndarray]) -> np.ndarray:
        """多重散乱の光源 J = Σ_k σs,k Σ_b χ_{k,l(b)}·c_b·Y_b(v)。p (n, m, 3)、dirs (n, 3)、sigmas は
        [σs_R, σs_mie_0, ...]（各 (n, m, 3)、χ と同じ順）→ (n, m, 3)。aerosol_experiment.integrate が使う。"""

        c = self.lookup(r, mu_sun)                                                    # (n, m, 6, 3)
        yb = local_basis(p, np.broadcast_to(dirs[:, None, :], p.shape), sun)          # (n, m, 6)
        band = yb[..., None] * c
        out = 0.0
        for k, sig in enumerate(sigmas):
            out = out + sig * np.einsum("nmbc,bc->nmc", band, self.chi[k][BAND_L])
        return out


class ShMultiScatteringTable(_ShSource):
    """SH（l ≤ 2）の多重散乱の表。coefficients (S, S, 6, 3) = (h, μs, b, RGB)。

    ground_term: 閉じ方の地表の項（案 5）。ground_albedo = 0 なら効かない。
    block_diagonal: F を m ごとのブロックで解く（GPU と同じ）。False なら完全な 6×6。
    gathers: 反復（案 6）の回数の差し込み口。今は 0 だけ。
    processes: 表を作るプロセス数（1 なら同じプロセスで作る）。"""

    def __init__(self, atmo: ae.Atmosphere, table: ae.TransmittanceTable, size: int = 32, steps: int = TABLE_STEPS,
                 ground_albedo: float = ae.GROUND_ALBEDO_MS, ground_term: bool = True, block_diagonal: bool = True,
                 gathers: int = 0, chi: np.ndarray | None = None, n_theta: int = 32, n_phi: int = 32,
                 processes: int = 1, raw: tuple[np.ndarray, np.ndarray] | None = None):
        if gathers != 0:
            # [POINT] 反復を足すときは、c_{n+1}(x) = ∫ T·Σ_k σs,k·J_n(y) dt + 地表 T·ρ/π·E_n(g) を、この表（前の次数）を
            #         光線上の点 y と地表の点 g で引いて求め、最後の次数を (I − F)⁻¹ の代わりに F(I − F)⁻¹ で閉じる
            #         （research.md §11.4 H 案 6）。c₁ と F はそのまま使える。
            raise NotImplementedError("gather iterations (research.md §11.4 H option 6) are not implemented")
        self.h = np.linspace(0.0, 100.0, size)
        self.mu = np.linspace(-1.0, 1.0, size)
        self.chi = medium_moments(atmo) if chi is None else np.asarray(chi, dtype=np.float64)
        self.ground_albedo = ground_albedo
        self.ground_term = ground_term
        self.block_diagonal = block_diagonal
        if raw is None:
            ctx = dict(atmo=atmo, tt=table, chi=self.chi, h=self.h, mu=self.mu, albedo=ground_albedo,
                       ground_term=ground_term, n_theta=n_theta, n_phi=n_phi, steps=steps)
            if processes > 1:
                with Pool(processes, initializer=_init_worker, initargs=(ctx,)) as pool:
                    rows = pool.map(_table_row, range(size), chunksize=1)
            else:
                _init_worker(ctx)
                rows = [_table_row(i) for i in range(size)]
            self.c1 = np.zeros((size, size, COEFF_COUNT, 3))
            self.F = np.zeros((size, size, 3, COEFF_COUNT, COEFF_COUNT))
            for i, row in rows:
                for j, (c1, F) in enumerate(row):
                    self.c1[i, j] = c1
                    self.F[i, j] = F
        else:
            self.c1, self.F = raw
        self.solve()

    def solve(self, block_diagonal: bool | None = None) -> None:
        """c₁ と F から係数を解き直す（block_diagonal を変えて比べるときに、表を作り直さずに使う）。"""

        if block_diagonal is not None:
            self.block_diagonal = block_diagonal
        S = self.h.size
        self.coefficients = np.zeros_like(self.c1)
        self.min_det = np.inf
        self.fallbacks = 0
        for i in range(S):
            for j in range(S):
                c, det, fb = solve_closure(self.F[i, j], self.c1[i, j], self.block_diagonal)
                self.coefficients[i, j] = c
                self.min_det = min(self.min_det, det)
                self.fallbacks += int(fb)

    def variant(self, block_diagonal: bool) -> "ShMultiScatteringTable":
        """同じ c₁・F で、解き方だけ違う表"""

        other = ShMultiScatteringTable.__new__(ShMultiScatteringTable)
        other.__dict__.update({k: v for k, v in self.__dict__.items()})
        other.solve(block_diagonal)
        return other

    def atlas(self) -> np.ndarray:
        """GPU のアトラスの並び (S, 6·S, 3)（行 = h、タイル b の列 = μs）。GpuShMultiScatteringTable の逆。"""

        S = self.h.size
        return np.concatenate([self.coefficients[:, :, b, :] for b in range(COEFF_COUNT)], axis=1).reshape(S, 6 * S, 3)


class GpuShMultiScatteringTable(_ShSource):
    """GPU の SH の Multi-scattering LUT（アトラス 192×32 = 32×32 のタイル 6 つ）を、GPU と同じ双線形で引く。

    タイル b（並び (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)）は列 [32b, 32b + 32)。GPU ではタイルの u =
    (MultiScatteringRMuSunToUv の x + b)/6 で、サブ UV なので隣のタイルは混ざらない。行 = r、タイル内の列 = μs。
    テクセルの (r, μs) は r_musun.exr（GpuMultiScatteringTable と同じ）。なければ等間隔の節点とみなす。
    χ は GPU と同じ値（Python で求めた medium_moments(atmo)、または CPU のサイドカーの値）を渡す。"""

    def __init__(self, path: Path, chi: np.ndarray, r_mu_path: Path | None = None):
        from exrio import read_exr

        lut, _ = read_exr(Path(path))
        self.coefficients = self.from_atlas(lut[..., :3].astype(np.float64))
        S_h, S_mu = self.coefficients.shape[:2]
        if r_mu_path is None:
            candidate = Path(path).with_name("r_musun.exr")
            r_mu_path = candidate if candidate.exists() else None
        if r_mu_path is not None:
            r_mu, _ = read_exr(Path(r_mu_path))
            self.h = r_mu[:, 0, 0].astype(np.float64) - ae.BOTTOM_KM
            self.mu = r_mu[0, :, 1].astype(np.float64)
        else:
            self.h = np.linspace(0.0, ae.TOP_KM - ae.BOTTOM_KM, S_h)
            self.mu = np.linspace(-1.0, 1.0, S_mu)
        self.chi = np.asarray(chi, dtype=np.float64)

    @staticmethod
    def from_atlas(atlas: np.ndarray) -> np.ndarray:
        """アトラス (S_h, 6·S_mu, 3) → 係数 (S_h, S_mu, 6, 3)"""

        S_h, width = atlas.shape[:2]
        if width % COEFF_COUNT != 0:
            raise ValueError(f"atlas width {width} is not a multiple of {COEFF_COUNT}")
        S_mu = width // COEFF_COUNT
        return np.stack([atlas[:, b * S_mu:(b + 1) * S_mu, :3] for b in range(COEFF_COUNT)], axis=2)
