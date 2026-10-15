"""大気の分光データ（係数セットの導出に使う）。

出典: E. Bruneton, precomputed_atmospheric_scattering, atmosphere/demo/demo.cc
      https://github.com/ebruneton/precomputed_atmospheric_scattering

  Copyright (c) 2017 Eric Bruneton
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

単位はすべてコメントに記す。波長 λ の表は 360〜830 nm、10 nm 間隔で、各値は区間 [λ, λ+10) nm の平均。

[POINT] 表の値は「その波長での値」ではなく「10 nm 区間の平均」なので、区間内で一定の階段関数として扱う
        （bin_value）。Bruneton の実装は値を区間先頭の波長のサンプルとして線形補間しており、これだと
        スペクトルが実質 5 nm 短波長側へずれる。太陽照度では約 0.5% の差になる（docs/design.md §14）。
"""

from __future__ import annotations

import numpy as np

# 波長 [nm]
WAVELENGTHS_NM = np.arange(360, 831, 10, dtype=np.float64)

# 大気圏外の太陽分光放射照度 [W·m⁻²·nm⁻¹]（ASTM G-173 ETR 列の 10 nm 区間平均）
SOLAR_IRRADIANCE_W_PER_M2_NM = np.array([
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
    1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
    1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
    1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
    1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
    1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992,
])

# オゾン吸収断面積 [m²]（233 K、Univ. Bremen の参照スペクトルの 10 nm 区間平均）
OZONE_CROSS_SECTION_M2 = np.array([
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
    2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27,
])

# ドブソン単位 [分子·m⁻²]
DOBSON_UNIT_PER_M2 = 2.687e20
# オゾンの最大数密度 [m⁻³]。300 DU を、テント型密度分布の積分（15 km）で割った値。
MAX_OZONE_NUMBER_DENSITY_PER_M3 = 300.0 * DOBSON_UNIT_PER_M2 / 15000.0

# Rayleigh 散乱係数の係数 [m⁻¹]: σ_R(λ) = RAYLEIGH_COEFF · (λ / 1 µm)⁻⁴（海面高度）
RAYLEIGH_COEFF_PER_M = 1.24062e-6
# Mie（Ångström の式 σ = β/H · λ^-α、α = 0 なので波長に依存しない）
MIE_ANGSTROM_BETA = 5.328e-3
MIE_ANGSTROM_ALPHA = 0.0
MIE_SCALE_HEIGHT_M = 1200.0
MIE_SINGLE_SCATTERING_ALBEDO = 0.9


def rayleigh_scattering_per_km(wavelength_nm: np.ndarray) -> np.ndarray:
    """海面高度での Rayleigh 散乱係数 [km⁻¹]。"""

    return RAYLEIGH_COEFF_PER_M * np.power(wavelength_nm * 1e-3, -4.0) * 1000.0


def bin_value(table: np.ndarray, wavelength_nm: np.ndarray) -> np.ndarray:
    """10 nm 区間平均の表を階段関数として評価する。λ ∈ [360 + 10i, 370 + 10i) → table[i]（範囲外は端の区間）。"""

    index = np.floor((np.asarray(wavelength_nm, dtype=np.float64) - WAVELENGTHS_NM[0]) / 10.0).astype(int)
    return table[np.clip(index, 0, table.size - 1)]


def ozone_absorption_per_km(wavelength_nm: np.ndarray) -> np.ndarray:
    """オゾン最大密度（高度 25 km）での吸収係数 [km⁻¹]。"""

    return MAX_OZONE_NUMBER_DENSITY_PER_M3 * bin_value(OZONE_CROSS_SECTION_M2, wavelength_nm) * 1000.0


def solar_irradiance(wavelength_nm: np.ndarray) -> np.ndarray:
    """大気圏外の太陽分光放射照度 [W·m⁻²·nm⁻¹]。"""

    return bin_value(SOLAR_IRRADIANCE_W_PER_M2_NM, wavelength_nm)


def mie_extinction_per_km() -> float:
    """海面高度での Mie 消散係数 [km⁻¹]（波長に依存しない）。"""

    return MIE_ANGSTROM_BETA / MIE_SCALE_HEIGHT_M * 1000.0
