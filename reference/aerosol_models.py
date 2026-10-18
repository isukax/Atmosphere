"""エアロゾルのモデル（docs/design.md §5.9、research.md §12）。粒子の仮定・鉛直分布・位相関数・プリセットの定義。

aerosol_experiment.py（実験・V4b の近似）と derive_coefficients.py（アプリが読む JSON のプリセット）の両方が
ここから読む。同じ数値の出どころを 1 つにするため。

[POINT] プリセット = research.md §12 の変種。Mie の成分のリストで表す（成分ごとに σs / σa（RGB）、鉛直分布、
        位相関数）。同じ鉛直分布の粒子はまとめて 1 成分にする（σs で重みを付けた位相関数の混合）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import mie

WAVELENGTHS_UM = np.array([0.680, 0.550, 0.440])  # R, G, B（Hillaire2020 の係数と同じ規約）
G_CHANNEL = 1
PHASE_TABLE_COUNT = 2049  # 位相関数の表の点の数（θ = π·u²、GPU・パストレーサーと同じ）

# ---------------------------------------------------------------------- エアロゾルの仮定
# [POINT] 粒子の値は AERONET の逆解析でよく見る範囲の代表値として置いた仮定（特定の観測地点・論文の値ではない）。
FINE = mie.LogNormalMode(volume_median_radius_um=0.15, sigma_ln=0.45, refractive_index=1.45 + 0.005j)
COARSE = mie.LogNormalMode(volume_median_radius_um=2.5, sigma_ln=0.65, refractive_index=1.53 + 0.003j)
AOD550_REALISTIC = 0.1     # 晴れた日の AOD（550 nm）の目安
FINE_FRACTION_OF_AOD = 0.7  # AOD のうち fine-mode の割合（550 nm）
BASE_MIE_SCATTERING = 3.996e-3  # Hillaire 2020（km⁻¹、海面）
BASE_MIE_ABSORPTION = 0.444e-3
BASE_MIE_SCALE_HEIGHT = 1.2
BASE_G = 0.8
# 案 D の鉛直分布（仮定）: fine は境界層 1.5 km、coarse はより低い 1.0 km。上端から 0.3 km で減る。
# 晴れた日の昼の代表値として文献の範囲に入る（research.md §13）。自由大気の背景の項は Phase 4.9 で足す
# （design.md §5.9「自由大気の背景の項」）
FINE_LAYER_TOP_KM = 1.5
COARSE_LAYER_TOP_KM = 1.0
LAYER_DECAY_KM = 0.3

# base+A / base+B / base+A+B は base の量（AOD550 ≈ 0.005）で A・B だけを入れたもの（C 系と同じ作り方）
PRESET_NAMES = ("base", "base+A", "base+B", "base+A+B", "C", "C+A", "C+B", "C+A+B", "C+D", "C+A+B+D")


# ---------------------------------------------------------------------- 鉛直分布
@dataclass(frozen=True)
class Profile:
    """ρ(h) = exp(−max(h − h_top, 0) / H)（GPU の SampleMedium と同じ解析式）。
    h_top = 0 なら指数分布、h_top > 0 なら境界層（上端まで一様、その上は H で減る）。"""

    top_km: float
    scale_height_km: float

    def __call__(self, h: np.ndarray) -> np.ndarray:
        return np.exp(-np.maximum(np.asarray(h, dtype=np.float64) - self.top_km, 0.0) / self.scale_height_km)


def exponential_profile(scale_height_km: float) -> Profile:
    return Profile(0.0, scale_height_km)


def boundary_layer_profile(top_km: float, decay_km: float) -> Profile:
    """境界層: 上端まで一様、その上は decay_km で指数的に減る（よく混ざった層の上端で急に澄む形）。"""

    return Profile(top_km, decay_km)


def column_km(profile: Callable[[np.ndarray], np.ndarray]) -> float:
    """∫ ρ(h) dh [km]（鉛直の柱。平面近似）。AOD = σ_ext(海面) × 柱。"""

    h = np.linspace(0.0, 100.0, 200001)
    return float(np.trapezoid(profile(h), h))


# ---------------------------------------------------------------------- 位相関数
class PhaseTable:
    """散乱角 θ = π·u²（u 等間隔）で刻んだ RGB の位相関数の表。"""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float64)  # (U, 3)
        self.u = np.linspace(0.0, 1.0, self.values.shape[0])

    def __call__(self, cos_theta: np.ndarray) -> np.ndarray:
        u = np.sqrt(np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi)
        return np.stack([np.interp(u, self.u, self.values[:, c]) for c in range(3)], axis=-1)


@dataclass(frozen=True)
class CornetteShanks:
    """Cornette-Shanks（Hillaire 2020 の Mie の近似）。全チャンネル共通。"""

    g: float

    def __call__(self, cos_theta: np.ndarray) -> np.ndarray:
        g2 = self.g * self.g
        denom = 1.0 + g2 - 2.0 * self.g * cos_theta
        p = 3.0 / (8.0 * np.pi) * (1.0 - g2) * (1.0 + cos_theta ** 2) / ((2.0 + g2) * denom * np.sqrt(denom))
        return np.repeat(np.asarray(p)[..., None], 3, axis=-1)


def cornette_shanks(g: float) -> CornetteShanks:
    return CornetteShanks(g)


def rayleigh_phase(cos_theta: np.ndarray) -> np.ndarray:
    return 3.0 / (16.0 * np.pi) * (1.0 + cos_theta ** 2)


def mean_cosine(phase: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """非対称パラメータ g = 2π ∫ p cosθ d(cosθ)（RGB）"""

    c = np.cos(np.pi * np.linspace(0.0, 1.0, 20001) ** 2)[::-1]
    p = phase(c)
    return 2.0 * np.pi * np.trapezoid(p * c[:, None], c, axis=0)


# ---------------------------------------------------------------------- 媒質
@dataclass
class MieComponent:
    name: str
    scattering: np.ndarray   # (3,) km⁻¹（海面 = 分布の値 1 のところ）
    absorption: np.ndarray   # (3,) km⁻¹
    profile: Callable[[np.ndarray], np.ndarray]
    phase: Callable[[np.ndarray], np.ndarray]  # cosθ (...,) → (..., 3)


@dataclass
class Atmosphere:
    rayleigh: np.ndarray
    ozone: np.ndarray
    mie: list[MieComponent] = field(default_factory=list)

    def coefficients(self, h: np.ndarray):
        """高度 h [km]（任意の形）での σs_R, [σs_mie_k], σt。形状 (..., 3)。"""

        h = np.maximum(h, 0.0)
        s_r = np.exp(-h / 8.0)[..., None] * self.rayleigh
        ozone = np.maximum(0.0, 1.0 - np.abs(h - 25.0) / 15.0)[..., None] * self.ozone
        s_m = []
        ext = s_r + ozone
        for comp in self.mie:
            rho = comp.profile(h)[..., None]
            s_m.append(rho * comp.scattering)
            ext = ext + rho * (comp.scattering + comp.absorption)
        return s_r, s_m, ext

    def aod(self) -> np.ndarray:
        return sum(column_km(c.profile) * (c.scattering + c.absorption) for c in self.mie)


# ---------------------------------------------------------------------- エアロゾルの光学特性（Mie）
@dataclass
class ModeOptics:
    """1 つのモードの、チャンネルごとの光学特性（1 個あたり）と 550 nm の値。"""

    c_ext: np.ndarray       # (3,) [µm²]
    ssa: np.ndarray         # (3,)
    phase: np.ndarray       # (U, 3) θ = π·u² の表
    g: np.ndarray           # (3,)
    # [POINT] AOD は 550 nm で決める。Hillaire2020 の規約（680 / 550 / 440 nm のサンプル）では G = 550 nm だが、
    #         分光フィットのセットでは G チャンネルの値は 550 nm の値と違うので、別に持つ。
    c_ext_550: float = 0.0
    ssa_550: float = 0.0
    phase_550: np.ndarray | None = None  # (U,)


def mode_optics(mode: mie.LogNormalMode, u_count: int = PHASE_TABLE_COUNT) -> ModeOptics:
    """680 / 550 / 440 nm の Mie 理論の値を R / G / B にする（Hillaire2020 の規約）。"""

    cos_grid = np.cos(np.pi * np.linspace(0.0, 1.0, u_count) ** 2)
    res = [mie.ensemble_optics(mode, wl, cos_grid) for wl in WAVELENGTHS_UM]
    phase = np.stack([e.phase for e in res], axis=1)
    return ModeOptics(c_ext=np.array([e.c_ext_um2 for e in res]),
                      ssa=np.array([e.single_scattering_albedo for e in res]),
                      phase=phase, g=np.array([e.asymmetry_g for e in res]),
                      c_ext_550=res[G_CHANNEL].c_ext_um2, ssa_550=res[G_CHANNEL].single_scattering_albedo,
                      phase_550=phase[:, G_CHANNEL].copy())


@dataclass
class SpectralModeOptics:
    """1 つのモードの分光の光学特性（1 個あたり）。"""

    wavelengths_nm: np.ndarray  # (L,)
    c_ext: np.ndarray           # (L,) [µm²]
    c_sca: np.ndarray           # (L,) [µm²]
    phase: np.ndarray           # (L, U) θ = π·u² の表


def spectral_mode_optics(mode: mie.LogNormalMode, wavelengths_nm: np.ndarray,
                         u_count: int = PHASE_TABLE_COUNT) -> SpectralModeOptics:
    cos_grid = np.cos(np.pi * np.linspace(0.0, 1.0, u_count) ** 2)
    res = [mie.ensemble_optics(mode, wl * 1e-3, cos_grid) for wl in wavelengths_nm]
    return SpectralModeOptics(np.asarray(wavelengths_nm, dtype=np.float64),
                              np.array([e.c_ext_um2 for e in res]), np.array([e.c_sca_um2 for e in res]),
                              np.stack([e.phase for e in res]))


def mie_mixture(modes: list[tuple[ModeOptics, float]], spectral: bool):
    """同じ鉛直分布の粒子をまとめた 1 成分の (σs, σa, 位相関数の表)。modes = [(光学特性, 550 nm の海面の σ_ext)]。
    spectral = False なら 550 nm の値を全チャンネルに使う（波長依存なし = 案 B なし）。"""

    sigma_s = np.zeros(3)
    sigma_a = np.zeros(3)
    weighted_phase = 0.0
    for optics, ext550 in modes:
        scale = optics.c_ext / optics.c_ext_550 if spectral else np.ones(3)
        ssa = optics.ssa if spectral else np.full(3, optics.ssa_550)
        ext = ext550 * scale
        s = ext * ssa
        sigma_s += s
        sigma_a += ext - s
        phase = optics.phase if spectral else np.repeat(optics.phase_550[:, None], 3, axis=1)
        weighted_phase = weighted_phase + phase * s
    # [POINT] 粒子の集まりを混ぜた位相関数は σs で重みを付けた平均（同じ鉛直分布なら視線上で一定）
    return sigma_s, sigma_a, weighted_phase / sigma_s


# ---------------------------------------------------------------------- 変種（プリセット）
def base_aod550() -> float:
    """base（Hillaire 2020）の AOD = σt（海面）× 柱 ≈ 0.0053（灰色なので波長によらない）"""

    return (BASE_MIE_SCATTERING + BASE_MIE_ABSORPTION) * column_km(exponential_profile(BASE_MIE_SCALE_HEIGHT))


def preset_aod550(name: str) -> float:
    """プリセットの AOD550（base 系は base の量、C 系は AOD550_REALISTIC）"""

    return base_aod550() if name.startswith("base") else AOD550_REALISTIC


def build_variants(rayleigh: np.ndarray, ozone: np.ndarray, fine: ModeOptics, coarse: ModeOptics) -> dict[str, Atmosphere]:
    base_profile = exponential_profile(BASE_MIE_SCALE_HEIGHT)
    base_col = column_km(base_profile)
    cs = cornette_shanks(BASE_G)
    gray = np.ones(3)

    def gray_component(aod550: float, profile, phase, col=None) -> MieComponent:
        col = column_km(profile) if col is None else col
        ext = aod550 / col
        ssa = BASE_MIE_SCATTERING / (BASE_MIE_SCATTERING + BASE_MIE_ABSORPTION)
        return MieComponent("mie", ext * ssa * gray, ext * (1 - ssa) * gray, profile, phase)

    base_aod = base_aod550()
    fine_ext = lambda aod, col: FINE_FRACTION_OF_AOD * aod / col  # noqa: E731
    coarse_ext = lambda aod, col: (1 - FINE_FRACTION_OF_AOD) * aod / col  # noqa: E731

    def mixed(aod: float, spectral: bool, phase_from_mie: bool, profile=base_profile) -> MieComponent:
        col = column_km(profile)
        s, a, p = mie_mixture([(fine, fine_ext(aod, col)), (coarse, coarse_ext(aod, col))], spectral)
        if phase_from_mie:
            return MieComponent("mie", s, a, profile, PhaseTable(p))
        return MieComponent("mie", s, a, profile, cs)

    fine_profile = boundary_layer_profile(FINE_LAYER_TOP_KM, LAYER_DECAY_KM)
    coarse_profile = boundary_layer_profile(COARSE_LAYER_TOP_KM, LAYER_DECAY_KM)

    def split_d(aod: float, spectral: bool, phase_from_mie: bool) -> list[MieComponent]:
        comps = []
        for name, optics, frac, profile in (("fine", fine, FINE_FRACTION_OF_AOD, fine_profile),
                                            ("coarse", coarse, 1 - FINE_FRACTION_OF_AOD, coarse_profile)):
            ext550 = frac * aod / column_km(profile)
            if spectral or phase_from_mie:
                s, a, p = mie_mixture([(optics, ext550)], spectral)
                phase = PhaseTable(p) if phase_from_mie else cs
            else:
                ssa = BASE_MIE_SCATTERING / (BASE_MIE_SCATTERING + BASE_MIE_ABSORPTION)
                s, a, phase = ext550 * ssa * gray, ext550 * (1 - ssa) * gray, cs
            comps.append(MieComponent(name, s, a, profile, phase))
        return comps

    base_mie = MieComponent("mie", BASE_MIE_SCATTERING * gray, BASE_MIE_ABSORPTION * gray, base_profile, cs)
    aod = AOD550_REALISTIC
    # [POINT] base+A / base+B / base+A+B は C 系と同じ作り方（AOD550 を base にそろえ、単散乱アルベドは Mie 理論の値）。
    #         base との差には、A・B のほかに単散乱アルベドの差（0.90 → 0.94）も入る。C → C+A なども同じ
    #         （2026-10-18 まで、research.md §12 の実験の base+A は base の σs・σa のまま位相関数だけを替えていた）
    return {
        "base": Atmosphere(rayleigh, ozone, [base_mie]),
        "base+A": Atmosphere(rayleigh, ozone, [mixed(base_aod, False, True)]),
        "base+B": Atmosphere(rayleigh, ozone, [mixed(base_aod, True, False)]),
        "base+A+B": Atmosphere(rayleigh, ozone, [mixed(base_aod, True, True)]),
        "C": Atmosphere(rayleigh, ozone, [gray_component(aod, base_profile, cs)]),
        "C+A": Atmosphere(rayleigh, ozone, [mixed(aod, False, True)]),
        "C+B": Atmosphere(rayleigh, ozone, [mixed(aod, True, False)]),
        "C+D": Atmosphere(rayleigh, ozone, split_d(aod, False, False)),
        "C+A+B": Atmosphere(rayleigh, ozone, [mixed(aod, True, True)]),
        "C+A+B+D": Atmosphere(rayleigh, ozone, split_d(aod, True, True)),
    }


# ---------------------------------------------------------------------- JSON（data/atmosphere_coefficients.json）
def _f32(values) -> list:
    """float32 に丸めた値（アプリは float32 で読むので、Python の検証も同じ値を使う）"""

    return [float(v) for v in np.asarray(values, dtype=np.float32).ravel()]


def component_to_json(comp: MieComponent) -> dict:
    out = {"name": comp.name, "scattering_per_km": _f32(comp.scattering), "absorption_per_km": _f32(comp.absorption)}
    if not isinstance(comp.profile, Profile):
        raise TypeError("profile must be Profile (analytic, design.md §5.9)")
    out["boundary_layer_top_km"] = comp.profile.top_km
    out["scale_height_km"] = comp.profile.scale_height_km
    if isinstance(comp.phase, CornetteShanks):
        out["phase"] = {"model": "cornette_shanks", "g": comp.phase.g}
    elif isinstance(comp.phase, PhaseTable):
        values = np.asarray(comp.phase.values, dtype=np.float32)
        out["phase"] = {"model": "table", "parameterization": "theta = pi * u^2, u uniform in [0, 1]",
                        "count": int(values.shape[0]), "asymmetry_g": _f32(mean_cosine(comp.phase)),
                        "values_rgb": [[float(x) for x in row] for row in values]}
    else:
        raise TypeError("unknown phase function")
    return out


def component_from_json(j: dict) -> MieComponent:
    profile = Profile(float(j["boundary_layer_top_km"]), float(j["scale_height_km"]))
    ph = j["phase"]
    phase = CornetteShanks(float(ph["g"])) if ph["model"] == "cornette_shanks" else PhaseTable(np.asarray(ph["values_rgb"]))
    return MieComponent(j["name"], np.asarray(j["scattering_per_km"], dtype=np.float64),
                        np.asarray(j["absorption_per_km"], dtype=np.float64), profile, phase)


def atmosphere_from_json(coefficient_set: dict, preset: str) -> Atmosphere:
    """係数セット（JSON の 1 要素）とプリセット名から、アプリと同じ値の大気を作る"""

    p = next(x for x in coefficient_set["aerosol_presets"] if x["name"] == preset)
    return Atmosphere(np.asarray(coefficient_set["rayleigh_scattering_per_km"], dtype=np.float64),
                      np.asarray(coefficient_set["ozone_absorption_per_km"], dtype=np.float64),
                      [component_from_json(c) for c in p["components"]])
