"""Mie 散乱（球形粒子）と、対数正規の粒径分布での平均（エアロゾルの係数と位相関数）。

アルゴリズムは Bohren & Huffman, "Absorption and Scattering of Light by Small Particles" (1983) 付録 A の BHMIE
（B. T. Draine の改訂版と同じ手順）を、式から numpy で書き直したもの（コードの移植ではない）。
  - 対数微分 D_n(mx) は下向きの漸化式（安定）。Riccati-Bessel 関数 ψ_n(x)、χ_n(x) は上向き（n ≤ nstop で安定）。
  - 係数 a_n, b_n から、効率 Q_ext / Q_sca / Q_back と、散乱振幅 S1(θ), S2(θ) を求める。
  - 角度関数 π_n, τ_n は上向きの漸化式。

[POINT] 屈折率は m = n + ik（k > 0 で吸収）。大気の文献の "1.45 − 0.005i" は k = 0.005 として渡す。
[POINT] 複数の半径をまとめて計算する（ベクトル化）。半径ごとに級数の打ち切り nstop が違うので、n > nstop の項は
        マスクで落とす（上向きの漸化式は n > x で不安定になり、ゴミの値が出るため）。

単位: 半径・波長は µm。断面積は µm²。位相関数は sr⁻¹（全立体角の積分が 1）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MieResult:
    q_ext: np.ndarray    # (R,) 消散効率
    q_sca: np.ndarray    # (R,) 散乱効率
    q_back: np.ndarray   # (R,) 後方散乱効率
    s11: np.ndarray      # (R, A) (|S1|² + |S2|²) / 2（無次元）。∫ s11 dΩ = k²·C_sca


def mie_spheres(x: np.ndarray, m: complex, cos_theta: np.ndarray) -> MieResult:
    """サイズパラメータ x = 2πr/λ（形状 (R,)）の球の Mie 散乱。cos_theta は散乱角の余弦（形状 (A,)）。"""

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    mu = np.atleast_1d(np.asarray(cos_theta, dtype=np.float64))
    m = complex(m)
    y = m * x
    nstop = np.floor(x + 4.0 * np.cbrt(x) + 2.0).astype(int)              # (R,)
    n_max = int(nstop.max())
    nmx = int(max(n_max, np.max(np.abs(y)))) + 16

    # D_n(y) の下向きの漸化式: D_{n−1} = n/y − 1/(D_n + n/y)
    d = np.zeros((nmx + 1, x.size), dtype=np.complex128)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / y - 1.0 / (d[n] + n / y)

    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = psi1 - 1j * chi1
    pi0 = np.zeros(mu.size)
    pi1 = np.ones(mu.size)
    s1 = np.zeros((x.size, mu.size), dtype=np.complex128)
    s2 = np.zeros((x.size, mu.size), dtype=np.complex128)
    sum_sca = np.zeros(x.size)
    sum_ext = np.zeros(x.size)
    sum_back = np.zeros(x.size, dtype=np.complex128)
    for n in range(1, n_max + 1):
        en = float(n)
        active = n <= nstop                                                 # (R,)
        psi = (2.0 * en - 1.0) / x * psi1 - psi0
        chi = (2.0 * en - 1.0) / x * chi1 - chi0
        xi = psi - 1j * chi
        da = d[n] / m + en / x
        db = m * d[n] + en / x
        an = (da * psi - psi1) / (da * xi - xi1)
        bn = (db * psi - psi1) / (db * xi - xi1)
        an = np.where(active, an, 0.0)
        bn = np.where(active, bn, 0.0)
        sum_sca += (2.0 * en + 1.0) * (np.abs(an) ** 2 + np.abs(bn) ** 2)
        sum_ext += (2.0 * en + 1.0) * np.real(an + bn)
        sum_back += (2.0 * en + 1.0) * (-1.0) ** n * (an - bn)
        # 角度関数: τ_n = n·μ·π_n − (n+1)·π_{n−1}
        tau = en * mu * pi1 - (en + 1.0) * pi0
        fn = (2.0 * en + 1.0) / (en * (en + 1.0))
        s1 += fn * (an[:, None] * pi1[None, :] + bn[:, None] * tau[None, :])
        s2 += fn * (an[:, None] * tau[None, :] + bn[:, None] * pi1[None, :])
        # 次の n へ
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = psi1 - 1j * chi1
        pi_next = ((2.0 * en + 1.0) * mu * pi1 - (en + 1.0) * pi0) / en
        pi0, pi1 = pi1, pi_next

    x2 = x * x
    return MieResult(q_ext=2.0 / x2 * sum_ext, q_sca=2.0 / x2 * sum_sca, q_back=np.abs(sum_back) ** 2 / x2,
                     s11=0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2))


@dataclass(frozen=True)
class LogNormalMode:
    """対数正規の粒径分布の 1 モード。"""

    volume_median_radius_um: float  # 体積の中央半径 r_v [µm]（AERONET の流儀）
    sigma_ln: float                 # ln(幾何標準偏差)。AERONET の σ と同じ
    refractive_index: complex       # n + ik（k > 0 で吸収）

    @property
    def number_median_radius_um(self) -> float:
        # [POINT] 体積の分布と個数の分布は同じ σ で、中央値が r_n = r_v·exp(−3σ²) ずれる
        return self.volume_median_radius_um * np.exp(-3.0 * self.sigma_ln ** 2)


@dataclass
class EnsembleOptics:
    """粒径分布で平均した光学特性（個数密度 1 個あたり）。"""

    wavelength_um: float
    c_ext_um2: float     # 平均の消散断面積 [µm²]（1 個あたり）
    c_sca_um2: float     # 平均の散乱断面積 [µm²]
    cos_theta: np.ndarray  # (A,)
    phase: np.ndarray      # (A,) 位相関数 [sr⁻¹]（∫ p dΩ = 1）
    asymmetry_g: float

    @property
    def single_scattering_albedo(self) -> float:
        return self.c_sca_um2 / self.c_ext_um2


def ensemble_optics(mode: LogNormalMode, wavelength_um: float, cos_theta: np.ndarray,
                    radius_count: int = 240, sigma_span: float = 4.0, chunk: int = 48) -> EnsembleOptics:
    """対数正規の個数分布で平均した断面積と位相関数。積分は ln r で台形則（±sigma_span·σ の範囲）。"""

    rn = mode.number_median_radius_um
    # [POINT] 散乱は r² 以上の重みが付くので、個数の中央値より大きい側まで積分する（体積の中央値 + 余裕）
    ln_lo = np.log(rn) - sigma_span * mode.sigma_ln
    ln_hi = np.log(mode.volume_median_radius_um) + sigma_span * mode.sigma_ln
    ln_r = np.linspace(ln_lo, ln_hi, radius_count)
    r = np.exp(ln_r)
    weight = np.exp(-0.5 * ((ln_r - np.log(rn)) / mode.sigma_ln) ** 2) / (np.sqrt(2.0 * np.pi) * mode.sigma_ln)
    w = weight * np.gradient(ln_r)              # dN/d ln r · Δ ln r（台形則に近い重み）
    w[0] *= 0.5
    w[-1] *= 0.5
    w /= w.sum()                                 # 範囲外の裾を落とした分を正規化（個数 1 個あたり）
    k = 2.0 * np.pi / wavelength_um
    cos_theta = np.asarray(cos_theta, dtype=np.float64)
    c_ext = 0.0
    c_sca = 0.0
    s11_sum = np.zeros(cos_theta.size)
    for start in range(0, r.size, chunk):
        rr = r[start:start + chunk]
        res = mie_spheres(k * rr, mode.refractive_index, cos_theta)
        geom = np.pi * rr * rr
        ww = w[start:start + chunk]
        c_ext += float(np.sum(ww * res.q_ext * geom))
        c_sca += float(np.sum(ww * res.q_sca * geom))
        s11_sum += ww @ res.s11
    # [POINT] p(θ) = ⟨(|S1|²+|S2|²)/2⟩ / (k²·⟨C_sca⟩)。∫ (|S1|²+|S2|²)/2 dΩ = k²·C_sca なので ∫ p dΩ = 1。
    phase = s11_sum / (k * k * c_sca)
    g = float(2.0 * np.pi * _integrate_cos(cos_theta, phase * cos_theta))
    return EnsembleOptics(wavelength_um, c_ext, c_sca, cos_theta, phase, g)


def phase_integral(cos_theta: np.ndarray, phase: np.ndarray) -> float:
    """∫ p dΩ = 2π ∫ p d(cosθ)（cos_theta の格子で台形則）。"""

    return float(2.0 * np.pi * _integrate_cos(cos_theta, phase))


def _integrate_cos(cos_theta: np.ndarray, f: np.ndarray) -> float:
    order = np.argsort(cos_theta)
    return float(np.trapezoid(f[order], cos_theta[order]))


def forward_cos_grid(count: int) -> np.ndarray:
    """散乱角 θ = π·u²（u は 0〜1 の等間隔）の余弦。前方（θ = 0）付近を細かく刻む。"""

    u = np.linspace(0.0, 1.0, count)
    return np.cos(np.pi * u * u)


# ---------------------------------------------------------------------- 自己検証
def self_test() -> list[tuple[str, float, float, float]]:
    """(項目, 値, 期待値, 許容誤差) のリスト。"""

    out = []
    # 1) Bohren & Huffman 付録 A の例: m = 1.55、半径 0.525 µm、波長 0.6328 µm（x = 5.213）
    x = 2.0 * np.pi * 0.525 / 0.6328
    res = mie_spheres(np.array([x]), 1.55 + 0.0j, np.array([1.0, 0.0, -1.0]))
    out.append(("BH example Q_sca", float(res.q_sca[0]), 3.10543, 1e-4))
    out.append(("BH example Q_back", float(res.q_back[0]), 2.92534, 1e-4))
    out.append(("BH example Q_ext - Q_sca (no absorption)", float(res.q_ext[0] - res.q_sca[0]), 0.0, 1e-9))
    # 2) Rayleigh の極限（x = 0.01）: Q_sca = 8/3·x⁴·|(m²−1)/(m²+2)|²、位相関数 3/(16π)(1+cos²θ)
    m = 1.5 + 0.0j
    x = 0.01
    mu = np.linspace(-1.0, 1.0, 9)
    res = mie_spheres(np.array([x]), m, mu)
    rayleigh_q = 8.0 / 3.0 * x ** 4 * abs((m * m - 1.0) / (m * m + 2.0)) ** 2
    out.append(("Rayleigh limit Q_sca / theory", float(res.q_sca[0] / rayleigh_q), 1.0, 1e-3))
    k_c_sca = res.q_sca[0] * x * x / 1.0  # k²·C_sca = Q_sca·x²·π / π（r = 1 として k = x）
    phase = res.s11[0] / (k_c_sca * np.pi)
    out.append(("Rayleigh limit phase max rel. error", float(np.max(np.abs(phase / (3 / (16 * np.pi) * (1 + mu * mu)) - 1))),
                0.0, 1e-3))
    # 3) 大きな粒子（x = 2000、吸収なし）: Q_ext → 2（消散のパラドックス）
    res = mie_spheres(np.array([2000.0]), 1.33 + 0.0j, np.array([1.0]))
    out.append(("Large sphere Q_ext", float(res.q_ext[0]), 2.0, 2e-2))
    # 4) 粒径分布で平均した位相関数の正規化
    grid = forward_cos_grid(4097)
    ens = ensemble_optics(LogNormalMode(2.5, 0.6, 1.53 + 0.003j), 0.55, grid)
    out.append(("Ensemble phase integral (coarse mode)", phase_integral(grid, ens.phase), 1.0, 2e-3))
    return out


if __name__ == "__main__":
    ok = True
    for name, value, expected, tol in self_test():
        passed = abs(value - expected) <= tol
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:<44} {value:.6g} (expected {expected:g} +/- {tol:g})")
    raise SystemExit(0 if ok else 1)
