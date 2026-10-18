"""作業色空間ごとの大気係数セットを分光データから導出する（docs/design.md §2.6）。

    reference/.venv/Scripts/python.exe reference/derive_coefficients.py

出力: data/atmosphere_coefficients.json（コミットする。アプリはこれを読むだけで係数を手書きしない）

[POINT] 散乱係数は「色」ではなく波長ごとの物理量なので、色空間の 3x3 行列では変換できない
        （透過率 exp(-σ·d) が σ について非線形）。そこで作業色空間の各チャンネル c について、
        太陽光を距離 d だけ通したときの分光透過光をそのチャンネルに射影した透過率
            T_c(d) = Σ_λ E_sun(λ)·c̄(λ)·exp(-σ(λ)·d) / Σ_λ E_sun(λ)·c̄(λ)
        に、RGB レンダラーの透過率 exp(-σ_c·d) が log 空間の最小二乗で一致する σ_c を求める。
        c̄(λ) は作業色空間の等色関数（CIE 1931 2° の x̄ȳz̄ を XYZ→RGB 行列で変換したもの）。
        d は「最大密度相当の距離」[km]。天頂（Rayleigh 約 8 km）から日没時の地平線方向
        （約 40 空気量 ≈ 300 km）までを範囲とする。

エアロゾルのプリセット（design.md §5.9、research.md §12 の変種）も各セットに書き出す。粒子の仮定と変種の定義は
aerosol_models.py にある。
  - Hillaire2020: 680 / 550 / 440 nm の Mie 理論の値を R / G / B にする（research.md §12 の実験と同じ規約）。
  - SpectralFit709 / 2020: Mie 理論の σt(λ)・σs(λ)・p(θ, λ)（10 nm 刻みで計算し 1 nm に線形補間）から、
      消散 σt_c は Rayleigh・オゾンと同じ透過率のフィット（モードごと、光学的距離はエアロゾルの範囲）、
      単散乱アルベド ω_c = Σ w_c·σs / Σ w_c·σt、位相関数 p_c = Σ w_c·σs·p / Σ w_c·σs
      （w_c = 太陽スペクトル × チャンネルの感度、負の値は 0。design.md §12 Q22）。
"""

from __future__ import annotations

import json
from pathlib import Path

import colour
import numpy as np

import aerosol_models as am
import spectral_data as sd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "atmosphere_coefficients.json"

WAVELENGTH_STEP_NM = 1.0
FIT_DISTANCE_KM = np.linspace(1.0, 300.0, 300)
MIN_TRANSMITTANCE_FOR_FIT = 1e-4
LUMINOUS_EFFICACY_LM_PER_W = 683.0
# [POINT] エアロゾルの光学的距離（海面の密度に換算）: 天頂（柱 ~1.2〜1.8 km）から地平線方向（~40 空気量）まで。
#         Rayleigh 用の範囲（~300 km）では AOD 0.1 の大気の透過率がほぼ 0 になり、フィットに使える点が偏る。
MIE_FIT_DISTANCE_KM = np.linspace(0.1, 60.0, 300)
MIE_WAVELENGTH_STEP_NM = 10.0
MIE_CACHE_DIR = ROOT / "dumps" / "cache"

COLOUR_SPACES = {
    "Rec709": "ITU-R BT.709",
    "Rec2020": "ITU-R BT.2020",
}


def cmfs_xyz(wavelengths_nm: np.ndarray) -> np.ndarray:
    """CIE 1931 2° 等色関数 x̄ȳz̄（形状 (N, 3)）。"""

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy()
    shape = colour.SpectralShape(wavelengths_nm[0], wavelengths_nm[-1], WAVELENGTH_STEP_NM)
    return cmfs.align(shape).values


def fit_channel_coefficients(sigma_per_km: np.ndarray, weights: np.ndarray,
                             distances_km: np.ndarray = FIT_DISTANCE_KM) -> tuple[np.ndarray, np.ndarray]:
    """チャンネルごとの等価係数 σ_c [km⁻¹] と、log 透過率の RMS 誤差を返す。

    sigma_per_km: 波長ごとの係数 (N,)、weights: E_sun·c̄（形状 (N, 3)）、distances_km: フィットの光学的距離。
    """

    transmitted = np.exp(-np.outer(distances_km, sigma_per_km))  # (D, N)
    t_channel = (transmitted @ weights) / weights.sum(axis=0)     # (D, 3)
    sigma_fit = np.zeros(3)
    rms = np.zeros(3)
    for c in range(3):
        valid = t_channel[:, c] > MIN_TRANSMITTANCE_FOR_FIT
        d = distances_km[valid]
        y = -np.log(t_channel[valid, c])
        # 原点を通る直線 y = σ·d の最小二乗解
        # [POINT] σ < 0（負の吸収 = 光の増幅）は物理的に不正なので 0 に制限する（非負制約付き最小二乗）。
        #         Rec.709 の B 等色関数は負の部分がオゾンの吸収帯（約 600 nm）にかかり、制約なしでは
        #         B チャンネルのオゾン係数が −2×10⁻⁵ km⁻¹ になる。
        sigma_fit[c] = max(0.0, float(np.dot(d, y) / np.dot(d, d)))
        rms[c] = float(np.sqrt(np.mean((y - sigma_fit[c] * d) ** 2)))
    return sigma_fit, rms


def interpolation_matrix(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """節点の値 → x での線形補間の値、の行列（形状 (len(x), len(nodes))）"""

    m = np.zeros((x.size, nodes.size))
    i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, nodes.size - 2)
    t = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
    m[np.arange(x.size), i] = 1.0 - t
    m[np.arange(x.size), i + 1] = t
    return m


def spectral_mode(mode, wavelengths_nm: np.ndarray) -> am.SpectralModeOptics:
    """Mie 理論の分光の光学特性（計算に数分かかるので dumps/cache に保存する）"""

    key = (f"{mode.volume_median_radius_um}_{mode.sigma_ln}_{mode.refractive_index.real}_{mode.refractive_index.imag}"
           f"_{wavelengths_nm[0]:.0f}_{wavelengths_nm[-1]:.0f}_{wavelengths_nm.size}_{am.PHASE_TABLE_COUNT}")
    path = MIE_CACHE_DIR / f"mie_spectral_{key}.npz"
    if path.exists():
        d = np.load(path)
        return am.SpectralModeOptics(d["wl"], d["c_ext"], d["c_sca"], d["phase"])
    spec = am.spectral_mode_optics(mode, wavelengths_nm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, wl=spec.wavelengths_nm, c_ext=spec.c_ext, c_sca=spec.c_sca, phase=spec.phase)
    return spec


def channel_mode_optics(spec: am.SpectralModeOptics, wl: np.ndarray, weights: np.ndarray,
                        sigma550_per_km: float) -> tuple[am.ModeOptics, np.ndarray]:
    """分光の光学特性 → チャンネルごとの値（design.md §5.9）。sigma550_per_km はフィットに使う海面の消散係数の目安。
    戻り値: (ModeOptics, 消散のフィットの log 透過率の RMS 誤差)"""

    interp = interpolation_matrix(wl, spec.wavelengths_nm)  # (N, L)
    c_ext = interp @ spec.c_ext
    c_sca = interp @ spec.c_sca
    i550 = int(np.argmin(np.abs(spec.wavelengths_nm - 550.0)))
    # [POINT] 透過率のフィットは σ について非線形なので、実際の大気に近い大きさの係数でフィットし、1 個あたりに戻す
    number = sigma550_per_km / spec.c_ext[i550]
    sigma_t, rms = fit_channel_coefficients(c_ext * number, weights, MIE_FIT_DISTANCE_KM)
    w = np.maximum(weights, 0.0)
    ssa = (w * c_sca[:, None]).sum(axis=0) / (w * c_ext[:, None]).sum(axis=0)
    # 位相関数: 1 nm の重み w·σs を、線形補間の係数で 10 nm の節点に配る（p は節点の値の線形補間なので同じ結果）
    coef = (w * c_sca[:, None]).T @ interp                        # (3, L)
    phase = ((coef @ spec.phase) / coef.sum(axis=1, keepdims=True)).T  # (U, 3)
    optics = am.ModeOptics(c_ext=sigma_t / number, ssa=ssa, phase=phase, g=am.mean_cosine(am.PhaseTable(phase)),
                           c_ext_550=float(spec.c_ext[i550]), ssa_550=float(spec.c_sca[i550] / spec.c_ext[i550]),
                           phase_550=spec.phase[i550].copy())
    return optics, rms


def aerosol_presets(rayleigh: np.ndarray, ozone: np.ndarray, fine: am.ModeOptics, coarse: am.ModeOptics) -> list[dict]:
    variants = am.build_variants(rayleigh, ozone, fine, coarse)
    return [{"name": name, "aod_rgb": [float(x) for x in variants[name].aod()],
             "components": [am.component_to_json(c) for c in variants[name].mie]} for name in am.PRESET_NAMES]


def optics_summary(o: am.ModeOptics) -> dict:
    return {"c_ext_over_550_rgb": (o.c_ext / o.c_ext_550).tolist(), "single_scattering_albedo_rgb": o.ssa.tolist(),
            "asymmetry_g_rgb": np.asarray(o.g).tolist()}


def dumps_with_compact_tables(output: dict) -> str:
    """indent=2 の JSON。ただし位相関数の表（values_rgb）は 1 行に 1 点（[r, g, b]）にする
    （indent=2 のままだと 1 値 1 行で 2,049 × 4 行になり、ファイルが読みにくく大きくなる）"""

    tables = []

    def placeholder(table):
        tables.append(table)
        return f"@@TABLE{len(tables) - 1}@@"

    def strip(node):
        if isinstance(node, dict):
            return {k: placeholder(v) if k == "values_rgb" else strip(v) for k, v in node.items()}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    text = json.dumps(strip(output), indent=2)
    for i, table in enumerate(tables):
        rows = ",\n".join(json.dumps(row) for row in table)
        text = text.replace(f'"@@TABLE{i}@@"', "[\n" + rows + "\n]", 1)
    return text


def effective_rayleigh_wavelength_nm(sigma_per_km: float) -> float:
    """Rayleigh の λ⁻⁴ 則で、係数 σ を与える波長 [nm]（解釈用の参考値）。"""

    return 1000.0 * (sd.RAYLEIGH_COEFF_PER_M * 1000.0 / sigma_per_km) ** 0.25


def main() -> None:
    wl = np.arange(360.0, 830.0 + WAVELENGTH_STEP_NM / 2, WAVELENGTH_STEP_NM)
    xyz_bar = cmfs_xyz(wl)
    sun = sd.solar_irradiance(wl)
    rayleigh = sd.rayleigh_scattering_per_km(wl)
    ozone = sd.ozone_absorption_per_km(wl)

    # 太陽の照度（大気圏外）。スペクトルから求めた値は参考としてアプリの既定値と比べる。
    sun_y = float(np.sum(sun * xyz_bar[:, 1]) * WAVELENGTH_STEP_NM)
    sun_illuminance_lux = LUMINOUS_EFFICACY_LM_PER_W * sun_y
    sun_xyz = (sun @ xyz_bar) * WAVELENGTH_STEP_NM

    mie_extinction = sd.mie_extinction_per_km()
    mie_scattering = mie_extinction * sd.MIE_SINGLE_SCATTERING_ALBEDO

    # エアロゾル（Mie 理論、design.md §5.9）
    fine_rgb = am.mode_optics(am.FINE)
    coarse_rgb = am.mode_optics(am.COARSE)
    mie_wl = np.arange(360.0, 830.0 + MIE_WAVELENGTH_STEP_NM / 2, MIE_WAVELENGTH_STEP_NM)
    fine_spec = spectral_mode(am.FINE, mie_wl)
    coarse_spec = spectral_mode(am.COARSE, mie_wl)
    base_column = am.column_km(am.exponential_profile(am.BASE_MIE_SCALE_HEIGHT))
    fine_sigma550 = am.FINE_FRACTION_OF_AOD * am.AOD550_REALISTIC / base_column
    coarse_sigma550 = (1.0 - am.FINE_FRACTION_OF_AOD) * am.AOD550_REALISTIC / base_column

    sets = [
        {
            "name": "Hillaire2020",
            "working_space": "Rec709",
            "method": "Hillaire 2020 / Bruneton 2017 values sampled at 680, 550, 440 nm, used as Rec.709 RGB",
            "rayleigh_scattering_per_km": [5.802e-3, 13.558e-3, 33.1e-3],
            "ozone_absorption_per_km": [0.650e-3, 1.881e-3, 0.085e-3],
            "sun_color": [1.0, 1.0, 1.0],
            "aerosol_method": "Mie theory at 680, 550, 440 nm used as R, G, B",
            "aerosol_optics": {"fine": optics_summary(fine_rgb), "coarse": optics_summary(coarse_rgb)},
            "aerosol_presets": aerosol_presets(np.array([5.802e-3, 13.558e-3, 33.1e-3]),
                                               np.array([0.650e-3, 1.881e-3, 0.085e-3]), fine_rgb, coarse_rgb),
        }
    ]
    for space, colour_name in COLOUR_SPACES.items():
        cs = colour.RGB_COLOURSPACES[colour_name]
        rgb_bar = xyz_bar @ np.asarray(cs.matrix_XYZ_to_RGB).T  # 作業色空間の等色関数 (N, 3)
        weights = sun[:, None] * rgb_bar
        rayleigh_fit, rayleigh_rms = fit_channel_coefficients(rayleigh, weights)
        ozone_fit, ozone_rms = fit_channel_coefficients(ozone, weights)
        sun_rgb = np.asarray(cs.matrix_XYZ_to_RGB) @ sun_xyz
        sun_color = sun_rgb / (sun_xyz[1])  # Y = 1 に正規化（RGB の輝度の重みで dot すると 1）
        fine_c, fine_rms = channel_mode_optics(fine_spec, wl, weights, fine_sigma550)
        coarse_c, coarse_rms = channel_mode_optics(coarse_spec, wl, weights, coarse_sigma550)
        sets.append(
            {
                "name": f"SpectralFit{space.removeprefix('Rec')}",
                "working_space": space,
                "method": "per-channel log-transmittance least-squares fit of solar-weighted spectral transmittance",
                "rayleigh_scattering_per_km": rayleigh_fit.tolist(),
                "ozone_absorption_per_km": ozone_fit.tolist(),
                "sun_color": sun_color.tolist(),
                "fit_rms_log_transmittance": {"rayleigh": rayleigh_rms.tolist(), "ozone": ozone_rms.tolist()},
                "rayleigh_effective_wavelength_nm": [effective_rayleigh_wavelength_nm(s) for s in rayleigh_fit],
                "aerosol_method": "Mie theory 360-830 nm (10 nm, linearly interpolated to 1 nm): per-mode extinction by "
                                  "the same log-transmittance fit, single scattering albedo and phase function as "
                                  "solar x channel weighted means (negative weights clamped to 0)",
                "aerosol_optics": {"fine": optics_summary(fine_c) | {"fit_rms_log_transmittance": fine_rms.tolist()},
                                   "coarse": optics_summary(coarse_c)
                                   | {"fit_rms_log_transmittance": coarse_rms.tolist()}},
                "aerosol_presets": aerosol_presets(rayleigh_fit, ozone_fit, fine_c, coarse_c),
            }
        )
    for s in sets:
        s["mie_scattering_per_km"] = mie_scattering
        s["mie_absorption_per_km"] = mie_extinction - mie_scattering

    output = {
        "generator": "reference/derive_coefficients.py",
        "units": {
            "rayleigh_scattering_per_km": "km^-1 at sea level",
            "ozone_absorption_per_km": "km^-1 at max ozone density (25 km)",
            "mie_scattering_per_km": "km^-1 at sea level",
            "mie_absorption_per_km": "km^-1 at sea level",
            "sun_color": "working-space RGB with luminance Y = 1 (multiply by illuminance in lux)",
            "aerosol_presets": "Mie components (design.md 5.9): scattering/absorption km^-1 where the profile is 1, "
                               "profile rho(h) = exp(-max(h - boundary_layer_top_km, 0) / scale_height_km), "
                               "phase sr^-1 (table: theta = pi * u^2)",
        },
        "conditions": {
            "cmfs": "CIE 1931 2 Degree Standard Observer",
            "colour_version": colour.__version__,
            "wavelength_range_nm": [360, 830],
            "wavelength_step_nm": WAVELENGTH_STEP_NM,
            "fit_distance_km": [float(FIT_DISTANCE_KM[0]), float(FIT_DISTANCE_KM[-1])],
            "mie_fit_distance_km": [float(MIE_FIT_DISTANCE_KM[0]), float(MIE_FIT_DISTANCE_KM[-1])],
            "mie_wavelength_step_nm": MIE_WAVELENGTH_STEP_NM,
            "aerosol_assumptions": {
                "fine": {"volume_median_radius_um": am.FINE.volume_median_radius_um, "sigma_ln": am.FINE.sigma_ln,
                         "refractive_index": str(am.FINE.refractive_index)},
                "coarse": {"volume_median_radius_um": am.COARSE.volume_median_radius_um,
                           "sigma_ln": am.COARSE.sigma_ln, "refractive_index": str(am.COARSE.refractive_index)},
                "aod550": am.AOD550_REALISTIC, "fine_fraction_of_aod550": am.FINE_FRACTION_OF_AOD,
            },
            "spectral_source": "Bruneton 2017 demo.cc (ASTM G-173 ETR, O3 233K), BSD-3",
        },
        "solar_illuminance_lux_from_spectrum": sun_illuminance_lux,
        "sets": sets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(dumps_with_compact_tables(output) + "\n", encoding="utf-8", newline="\n")  # 改行は LF

    print(f"solar illuminance from spectrum: {sun_illuminance_lux:.0f} lx")
    for s in sets:
        r = s["rayleigh_scattering_per_km"]
        o = s["ozone_absorption_per_km"]
        c = s["sun_color"]
        print(f"{s['name']:<16} rayleigh [1e-3/km] {r[0]*1e3:7.3f} {r[1]*1e3:7.3f} {r[2]*1e3:7.3f}"
              f" | ozone [1e-3/km] {o[0]*1e3:6.3f} {o[1]*1e3:6.3f} {o[2]*1e3:6.3f}"
              f" | sun {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
        if "rayleigh_effective_wavelength_nm" in s:
            print(f"{'':<16} rayleigh effective wavelength [nm] "
                  + " ".join(f"{w:.1f}" for w in s["rayleigh_effective_wavelength_nm"])
                  + f" | fit rms(log T) {max(s['fit_rms_log_transmittance']['rayleigh']):.2e}")
        for mode, o in s["aerosol_optics"].items():
            print(f"{'':<16} {mode:<6} sigma_t/sigma_t550 " + " ".join(f"{x:.3f}" for x in o["c_ext_over_550_rgb"])
                  + " | ssa " + " ".join(f"{x:.3f}" for x in o["single_scattering_albedo_rgb"])
                  + " | g " + " ".join(f"{x:.3f}" for x in o["asymmetry_g_rgb"]))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
