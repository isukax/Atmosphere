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
"""

from __future__ import annotations

import json
from pathlib import Path

import colour
import numpy as np

import spectral_data as sd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "atmosphere_coefficients.json"

WAVELENGTH_STEP_NM = 1.0
FIT_DISTANCE_KM = np.linspace(1.0, 300.0, 300)
MIN_TRANSMITTANCE_FOR_FIT = 1e-4
LUMINOUS_EFFICACY_LM_PER_W = 683.0

COLOUR_SPACES = {
    "Rec709": "ITU-R BT.709",
    "Rec2020": "ITU-R BT.2020",
}


def cmfs_xyz(wavelengths_nm: np.ndarray) -> np.ndarray:
    """CIE 1931 2° 等色関数 x̄ȳz̄（形状 (N, 3)）。"""

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy()
    shape = colour.SpectralShape(wavelengths_nm[0], wavelengths_nm[-1], WAVELENGTH_STEP_NM)
    return cmfs.align(shape).values


def fit_channel_coefficients(sigma_per_km: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """チャンネルごとの等価係数 σ_c [km⁻¹] と、log 透過率の RMS 誤差を返す。

    sigma_per_km: 波長ごとの係数 (N,)、weights: E_sun·c̄（形状 (N, 3)）。
    """

    transmitted = np.exp(-np.outer(FIT_DISTANCE_KM, sigma_per_km))  # (D, N)
    t_channel = (transmitted @ weights) / weights.sum(axis=0)        # (D, 3)
    sigma_fit = np.zeros(3)
    rms = np.zeros(3)
    for c in range(3):
        valid = t_channel[:, c] > MIN_TRANSMITTANCE_FOR_FIT
        d = FIT_DISTANCE_KM[valid]
        y = -np.log(t_channel[valid, c])
        # 原点を通る直線 y = σ·d の最小二乗解
        # [POINT] σ < 0（負の吸収 = 光の増幅）は物理的に不正なので 0 に制限する（非負制約付き最小二乗）。
        #         Rec.709 の B 等色関数は負の部分がオゾンの吸収帯（約 600 nm）にかかり、制約なしでは
        #         B チャンネルのオゾン係数が −2×10⁻⁵ km⁻¹ になる。
        sigma_fit[c] = max(0.0, float(np.dot(d, y) / np.dot(d, d)))
        rms[c] = float(np.sqrt(np.mean((y - sigma_fit[c] * d) ** 2)))
    return sigma_fit, rms


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

    sets = [
        {
            "name": "Hillaire2020",
            "working_space": "Rec709",
            "method": "Hillaire 2020 / Bruneton 2017 values sampled at 680, 550, 440 nm, used as Rec.709 RGB",
            "rayleigh_scattering_per_km": [5.802e-3, 13.558e-3, 33.1e-3],
            "ozone_absorption_per_km": [0.650e-3, 1.881e-3, 0.085e-3],
            "sun_color": [1.0, 1.0, 1.0],
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
        },
        "conditions": {
            "cmfs": "CIE 1931 2 Degree Standard Observer",
            "colour_version": colour.__version__,
            "wavelength_range_nm": [360, 830],
            "wavelength_step_nm": WAVELENGTH_STEP_NM,
            "fit_distance_km": [float(FIT_DISTANCE_KM[0]), float(FIT_DISTANCE_KM[-1])],
            "spectral_source": "Bruneton 2017 demo.cc (ASTM G-173 ETR, O3 233K), BSD-3",
        },
        "solar_illuminance_lux_from_spectrum": sun_illuminance_lux,
        "sets": sets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")  # 改行は LF

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
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
