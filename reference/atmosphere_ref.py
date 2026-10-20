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
    # エアロゾルのプリセット（design.md §5.9）: Mie の鉛直分布 exp(−max(h − h_top, 0)/H) + a·exp(−h/H_FT) と、
    # 位相関数の表
    mie_boundary_layer_top_km: float = 0.0
    mie_phase_table: np.ndarray | None = None  # (U, C) θ = π·u²（u 等間隔）。None なら Cornette-Shanks
    mie_free_trop_density: float = 0.0
    mie_free_trop_scale_height_km: float = 1.0
    # 2 つ目の Mie の成分（Phase 4.9。None なら 1 成分）。意味は成分 0 の同名のフィールドと同じ
    mie1_scattering_per_km: np.ndarray | None = None
    mie1_extinction_per_km: np.ndarray | None = None
    mie1_scale_height_km: float = 1.0
    mie1_boundary_layer_top_km: float = 0.0
    mie1_free_trop_density: float = 0.0
    mie1_free_trop_scale_height_km: float = 1.0
    mie1_phase_g: float = 0.8
    mie1_phase_table: np.ndarray | None = None

    @classmethod
    def from_meta(cls, meta: dict) -> "Atmosphere":
        """アプリが書いたサイドカーの "atmosphere"（AtmosphereParams の JSON）から作る。
        位相関数の表はサイドカーに入れていないので、係数セットとプリセットの名前で data/atmosphere_coefficients.json
        から読む（アプリと同じ値）。"""

        a = meta["atmosphere"]
        mie_scattering = np.asarray(a["mie_scattering_per_km"], dtype=np.float64)
        # [POINT] aerosol_preset は g や AOD を変えると " (modified)" が付く。表は元の名前で引く
        #         （σs・σa はサイドカーの値＝AOD の倍率を掛けた後の値を使うので、表以外は名前を使わない）
        preset_name = a.get("aerosol_preset_name", a.get("aerosol_preset"))
        table = None
        if a.get("mie_phase_model") == "table":
            table = load_phase_table(a["coefficient_set"], preset_name)
        mie1 = {}
        if int(a.get("mie_component_count", 1)) == 2:
            m1 = a["mie1"]
            s1 = np.asarray(m1["scattering_per_km"], dtype=np.float64)
            mie1 = dict(
                mie1_scattering_per_km=s1,
                mie1_extinction_per_km=s1 + np.asarray(m1["absorption_per_km"], dtype=np.float64),
                mie1_scale_height_km=float(m1["scale_height_km"]),
                mie1_boundary_layer_top_km=float(m1["boundary_layer_top_km"]),
                mie1_free_trop_density=float(m1["free_troposphere_density"]),
                mie1_free_trop_scale_height_km=float(m1["free_troposphere_scale_height_km"]),
                mie1_phase_g=float(m1["phase_g"]),
                mie1_phase_table=(load_phase_table(a["coefficient_set"], preset_name, 1)
                                  if m1["phase_model"] == "table" else None),
            )
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
            mie_boundary_layer_top_km=float(a.get("mie_boundary_layer_top_km", 0.0)),
            mie_phase_table=table,
            mie_free_trop_density=float(a.get("mie_free_troposphere_density", 0.0)),
            mie_free_trop_scale_height_km=float(a.get("mie_free_troposphere_scale_height_km", 1.0)),
            **mie1,
        )

    @classmethod
    def spectral(cls, meta: dict, wavelengths_nm: np.ndarray) -> "Atmosphere":
        """幾何・スケールハイト・位相関数はサイドカーの値、係数は分光データ（チャンネル = 波長）で作る（V9 用）。"""

        import spectral_data as sd

        a = meta["atmosphere"]
        wl = np.asarray(wavelengths_nm, dtype=np.float64)
        mie_extinction = np.full(wl.shape, sd.mie_extinction_per_km())
        return cls(
            bottom_radius_km=float(a["bottom_radius_km"]),
            top_radius_km=float(a["top_radius_km"]),
            rayleigh_scattering_per_km=sd.rayleigh_scattering_per_km(wl),
            rayleigh_scale_height_km=float(a["rayleigh_scale_height_km"]),
            mie_scattering_per_km=mie_extinction * sd.MIE_SINGLE_SCATTERING_ALBEDO,
            mie_extinction_per_km=mie_extinction,
            mie_scale_height_km=float(a["mie_scale_height_km"]),
            mie_phase_g=float(a["mie_phase_g"]),
            ozone_absorption_per_km=sd.ozone_absorption_per_km(wl),
            ozone_center_km=float(a["ozone_center_km"]),
            ozone_half_width_km=float(a["ozone_half_width_km"]),
        )

    # ------------------------------------------------------------------ 媒質
    @property
    def has_mie1(self) -> bool:
        return self.mie1_scattering_per_km is not None

    def densities(self, altitude_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rayleigh・Mie（成分 0）・オゾンの密度。成分 1 は mie1_density。"""

        h = np.asarray(altitude_km, dtype=np.float64)
        rayleigh = np.exp(-h / self.rayleigh_scale_height_km)
        mie = mie_profile(h, self.mie_boundary_layer_top_km, self.mie_scale_height_km, self.mie_free_trop_density,
                          self.mie_free_trop_scale_height_km)
        ozone = np.maximum(0.0, 1.0 - np.abs(h - self.ozone_center_km) / self.ozone_half_width_km)
        return rayleigh, mie, ozone

    def mie1_density(self, altitude_km: np.ndarray) -> np.ndarray:
        return mie_profile(np.asarray(altitude_km, dtype=np.float64), self.mie1_boundary_layer_top_km,
                           self.mie1_scale_height_km, self.mie1_free_trop_density, self.mie1_free_trop_scale_height_km)

    def mie_phase(self, cos_theta: np.ndarray, component: int = 0) -> np.ndarray:
        """Mie の位相関数 [sr⁻¹]、形状 (N, C)（Cornette-Shanks なら (N, 1)）。
        表は θ = π·u² で刻んだ点を u = √(θ/π) で線形補間する（GPU の MiePhase と同じ）。"""

        cos_theta = np.asarray(cos_theta, dtype=np.float64)
        table, g = (self.mie_phase_table, self.mie_phase_g) if component == 0 else (self.mie1_phase_table,
                                                                                   self.mie1_phase_g)
        if table is None:
            return cornette_shanks_phase(g, cos_theta)[:, None]
        table = np.asarray(table, dtype=np.float64)
        u = np.sqrt(np.arccos(np.clip(cos_theta, -1.0, 1.0)) / np.pi)
        grid = np.linspace(0.0, 1.0, table.shape[0])
        return np.stack([np.interp(u, grid, table[:, c]) for c in range(table.shape[1])], axis=-1)

    def extinction(self, altitude_km: np.ndarray) -> np.ndarray:
        """σt(h) [km⁻¹]、形状 (..., 3)。"""

        r, m, o = self.densities(altitude_km)
        ext = (r[..., None] * self.rayleigh_scattering_per_km + m[..., None] * self.mie_extinction_per_km
               + o[..., None] * self.ozone_absorption_per_km)
        if self.has_mie1:
            ext = ext + self.mie1_density(altitude_km)[..., None] * self.mie1_extinction_per_km
        return ext

    # ------------------------------------------------------------------ 幾何
    def distance_to_top(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        disc = r * r * (mu * mu - 1.0) + self.top_radius_km ** 2
        return np.maximum(0.0, -r * mu + np.sqrt(np.maximum(disc, 0.0)))

    def ray_intersects_ground(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return (mu < 0.0) & (r * r * (mu * mu - 1.0) + self.bottom_radius_km ** 2 >= 0.0)

    def kink_breakpoints(self, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """光線（半径 r、天頂角の余弦 mu）がオゾン分布の折れ目の高度（と Mie の境界層の上端）を横切る距離 t、
        形状 (M, 6 or 8)。横切らない分は NaN。半径 ρ の球面との交点は t = −r·mu ± √(r²mu² − r² + ρ²)。"""

        heights = [self.ozone_center_km - self.ozone_half_width_km, self.ozone_center_km,
                   self.ozone_center_km + self.ozone_half_width_km]
        if self.mie_boundary_layer_top_km > 0.0:
            heights.append(self.mie_boundary_layer_top_km)  # 境界層の上端で Mie の密度の微分が不連続
        if self.has_mie1 and self.mie1_boundary_layer_top_km > 0.0:
            heights.append(self.mie1_boundary_layer_top_km)
        radii = self.bottom_radius_km + np.array(heights)
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

    # ------------------------------------------------------------------ 視線に沿った散乱の積分
    @property
    def channel_count(self) -> int:
        """係数のチャンネル数（RGB なら 3、分光なら波長の数）。"""

        return int(np.asarray(self.rayleigh_scattering_per_km).size)

    def single_scattering_per_lux(self, camera_km: np.ndarray, directions: np.ndarray, sun_dir: np.ndarray,
                                  outer_panels: int = 32, inner_panels: int = 8, order: int = 4,
                                  chunk: int = 8) -> np.ndarray:
        """単位照度あたりの単散乱輝度 [sr⁻¹]、形状 (N, C)。太陽は点光源、惑星の影あり、地表反射なし。"""

        return self.scattering_integrals(camera_km, directions, sun_dir, outer_panels=outer_panels,
                                         inner_panels=inner_panels, order=order, chunk=chunk)[0]

    def multiple_scattering_psi(self, r_km: float, mu_sun: float, ground_albedo, outer_panels: int = 32,
                                inner_panels: int = 8, order: int = 4) -> np.ndarray:
        """Multi-scattering LUT の値 Ψ_ms [sr⁻¹]（シェーダーの MultiScatteringLut.hlsl と同じ 64 方向・同じ式）。"""

        k = np.arange(64)
        a = (k % 8 + 0.5) / 8.0
        b = (k // 8 + 0.5) / 8.0
        theta = 2.0 * np.pi * a
        phi = np.arccos(1.0 - 2.0 * b)
        dirs = np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], axis=1)
        origin = np.array([0.0, r_km, 0.0])
        sun = np.array([np.sqrt(max(0.0, 1.0 - mu_sun * mu_sun)), mu_sun, 0.0])
        lum, ms1 = self.scattering_integrals(origin, dirs, sun, isotropic=True, ground_albedo=ground_albedo,
                                             outer_panels=outer_panels, inner_panels=inner_panels, order=order)
        second_order = lum.mean(axis=0)   # Σ L·(4π/N)·(1/(4π))
        fms = ms1.mean(axis=0)
        return second_order / (1.0 - fms)  # Ψ_ms = L₂ / (1 − f_ms)

    def multiple_scattering_sh(self, r_km: float, mu_sun: float, ground_albedo, moments: np.ndarray,
                               outer_panels: int = 32, inner_panels: int = 8, order: int = 4) -> np.ndarray:
        """多重散乱の SH 近似の係数 c_b（6, C）（design.md §5.11。シェーダーの MultiScatteringLut.hlsl の MS_SH の版と
        同じ方向・同じ式）。moments: Mie の成分ごとの χ_0〜χ_2、形状 (成分, 3, C)。Rayleigh は (1, 0, 0.1)。

          c₁ = Σ w·Y_b·L₁（L₁ = 実際の位相関数の単散乱 + 地表での直達光の反射）
          F_ab = Σ w·Y_a·Y_b·Ã_{l(b)} + (ρ/π)·G_a·e_b（m ごとのブロック）、c = (I − F)⁻¹·c₁"""

        dirs, weights = sh_sun_grid(mu_sun)
        origin = np.array([0.0, r_km, 0.0])
        sun = np.array([np.sqrt(max(0.0, 1.0 - mu_sun * mu_sun)), mu_sun, 0.0])
        lum, _, extra = self.scattering_integrals(origin, dirs, sun, ground_albedo=ground_albedo,
                                                  outer_panels=outer_panels, inner_panels=inner_panels, order=order,
                                                  components=True)
        # SH の枠: z = +Y、x = +X、y = −Z（基底は y² だけ）
        basis = sh_basis6(dirs[:, 0], dirs[:, 1], dirs[:, 2] ** 2)                 # (N, 6)
        n_ch = lum.shape[1]
        moments = np.asarray(moments, dtype=np.float64).reshape(-1, 3, n_ch)
        rayleigh_chi = np.array([1.0, 0.0, 0.1])
        a_k = extra["as1"]                                                          # (N, K, C)
        band_as1 = rayleigh_chi[None, :, None] * a_k[:, 0, None, :]                # (N, 3, C)
        for k in range(1, a_k.shape[1]):
            band_as1 = band_as1 + moments[k - 1][None] * a_k[:, k, None, :]
        c1 = np.einsum("n,nb,nc->bc", weights, basis, lum)
        tg = np.where(extra["hit_ground"][:, None], extra["ground_transmittance"], 0.0)
        albedo = np.broadcast_to(np.asarray(ground_albedo, dtype=np.float64), (n_ch,))
        coeffs = np.zeros((6, n_ch))
        for block in SH_BLOCKS:
            idx = np.array(block)
            bands = SH_BANDS[idx]
            f = np.einsum("n,na,nb,nbc->cab", weights, basis[:, idx], basis[:, idx], band_as1[:, bands, :])
            if 0 in block:
                g = np.einsum("n,na,nc->ca", weights, basis[:, idx], tg)               # (C, 3)
                f = f + (albedo / np.pi)[:, None, None] * g[:, :, None] * SH_HEMISPHERE_COSINE[idx][None, None, :]
            a = np.eye(idx.size)[None] - f
            coeffs[idx] = np.linalg.solve(a, c1[idx].T[..., None])[..., 0].T
        return coeffs

    def legendre_moments(self, component: int = 0, count: int = 400001) -> np.ndarray:
        """Mie の成分の位相関数の Legendre のモーメント χ_0〜χ_2、形状 (3, C)。u（θ = π·u²）で台形則（GPU と同じ
        補間の位相関数を、C++ の PhaseMoments とは別の積分で求める）。"""

        u = np.linspace(0.0, 1.0, count)
        mu = np.cos(np.pi * u * u)
        dmu_du = np.sin(np.pi * u * u) * 2.0 * np.pi * u
        p = self.mie_phase(mu, component)                                           # (U, C or 1)
        pl = np.stack([np.ones_like(mu), mu, 0.5 * (3.0 * mu * mu - 1.0)])
        return 2.0 * np.pi * np.trapezoid(pl[:, :, None] * p[None] * dmu_du[None, :, None], u, axis=1)

    def scattering_integrals(self, camera_km: np.ndarray, directions: np.ndarray, sun_dir: np.ndarray,
                             isotropic: bool = False, ground_albedo=None, outer_panels: int = 32,
                             inner_panels: int = 8, order: int = 4, chunk: int = 8, components: bool = False):
        """視線ごとの積分 (L, F)、形状 (N, C)。
          L = ∫ T_view·Σ σs·p·T_sun dt（+ 地表の反射）… 単位照度あたりの輝度 [sr⁻¹]
          F = ∫ T_view·σs dt                         … f_ms の被積分量（Multi-scattering LUT 用）
        isotropic: 位相関数を 1/(4π) にする。ground_albedo: 地表のランバート反射を加える（None なら加えない）。
        components: True なら 3 つ目の戻り値に、成分ごとの ∫ T_view·σs,k dt（"as1"、(N, K, C)。K = Rayleigh・Mie・
        Mie の成分 1）、地表に当たったか（"hit_ground"）、終端までの透過率（"ground_transmittance"）を返す。"""

        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        camera_km = np.asarray(camera_km, dtype=np.float64)
        sun_dir = np.asarray(sun_dir, dtype=np.float64)
        n_ch = self.channel_count
        n = directions.shape[0]
        lum = np.zeros((n, n_ch))
        ms1 = np.zeros((n, n_ch))
        k = 3 if self.has_mie1 else 2
        extra = {"as1": np.zeros((n, k, n_ch)), "hit_ground": np.zeros(n, dtype=bool),
                 "ground_transmittance": np.zeros((n, n_ch))}
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            out = self._ray_chunk(camera_km, directions[sl], sun_dir, outer_panels, inner_panels, order, isotropic,
                                  ground_albedo, components)
            lum[sl], ms1[sl] = out[0], out[1]
            if components:
                extra["as1"][sl], extra["hit_ground"][sl], extra["ground_transmittance"][sl] = out[2]
        return (lum, ms1, extra) if components else (lum, ms1)

    def _ray_chunk(self, c, dirs, sun, outer_panels, inner_panels, order, isotropic, ground_albedo,
                   components=False):
        n = dirs.shape[0]
        n_ch = self.channel_count
        t_start, t_end = np.zeros(n), np.zeros(n)
        hit_ground = np.zeros(n, dtype=bool)
        for i, d in enumerate(dirs):
            top = ray_sphere(c, d, self.top_radius_km)
            ground = ray_sphere(c, d, self.bottom_radius_km)
            t_start[i] = max(top[0], 0.0)
            t_end[i] = top[1]
            if ground is not None and 0.0 < ground[0] < t_end[i]:
                t_end[i] = ground[0]
                hit_ground[i] = True

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

        def extinction_at(tt):  # tt: (n, M) の視線上の距離 → (n, M, C)
            p = c + tt[..., None] * dirs[:, None, :]
            return self.extinction(np.linalg.norm(p, axis=-1) - self.bottom_radius_km)

        # カメラ → 求積点の光学的厚さ: 区間全体の積分を累積 + 区間内の部分積分（区間先頭 → 点）
        ext_nodes = extinction_at(t).reshape(n, outer_panels, order, n_ch)
        panel_depth = np.einsum("npk,npkc->npc", (half[:, :, None] * wx), ext_nodes)
        cumulative = np.cumsum(panel_depth, axis=1)
        depth_before = np.concatenate([np.zeros((n, 1, n_ch)), cumulative[:, :-1]], axis=1)
        t_nodes = t.reshape(n, outer_panels, order)
        sub_half = 0.5 * (t_nodes - edges[:, :-1, None])                   # (n, P, order)
        sub_mid = 0.5 * (t_nodes + edges[:, :-1, None])
        sub_t = sub_mid[..., None] + sub_half[..., None] * x               # (n, P, order, order)
        sub_ext = extinction_at(sub_t.reshape(n, -1)).reshape(n, outer_panels, order, order, n_ch)
        partial = np.einsum("npk,npkj,npkjc->npkc", sub_half, np.broadcast_to(wx, sub_half.shape + (order,)), sub_ext)
        view_t = np.exp(-(depth_before[:, :, None, :] + partial).reshape(n, -1, n_ch))

        # 求積点での散乱と太陽方向の透過率
        p = c + t[..., None] * dirs[:, None, :]
        r = np.linalg.norm(p, axis=-1)
        rayleigh_d, mie_d, _ = self.densities(r - self.bottom_radius_km)
        cos_theta = dirs @ sun
        if isotropic:
            phase_r = np.full((n, 1), 1.0 / (4.0 * np.pi))
            phase_m = np.full((n, 1, 1), 1.0 / (4.0 * np.pi))
        else:
            phase_r = rayleigh_phase(cos_theta)[:, None]
            phase_m = self.mie_phase(cos_theta)[:, None, :]  # (n, 1, C or 1)
        sigma_r = rayleigh_d[..., None] * self.rayleigh_scattering_per_km
        sigma_m = mie_d[..., None] * self.mie_scattering_per_km
        source = sigma_r * phase_r[..., None] + sigma_m * phase_m
        sigma_s = sigma_r + sigma_m
        if self.has_mie1:
            # [POINT] 成分ごとに位相関数を掛ける（鉛直分布が違うので混ぜた位相関数にはまとめられない）
            sigma_m1 = self.mie1_density(r - self.bottom_radius_km)[..., None] * self.mie1_scattering_per_km
            phase_m1 = phase_m if isotropic else self.mie_phase(cos_theta, 1)[:, None, :]
            source = source + sigma_m1 * phase_m1
            sigma_s = sigma_s + sigma_m1
        sun_t = self.sun_transmittance(p.reshape(-1, 3), sun, inner_panels, order).reshape(r.shape + (n_ch,))
        lum = np.einsum("nm,nmc->nc", w, view_t * source * sun_t)
        ms1 = np.einsum("nm,nmc->nc", w, view_t * sigma_s)
        as1 = None
        if components:
            sigmas = [sigma_r, sigma_m] + ([sigma_m1] if self.has_mie1 else [])
            as1 = np.stack([np.einsum("nm,nmc->nc", w, view_t * s_k) for s_k in sigmas], axis=1)

        if ground_albedo is not None and np.any(hit_ground):
            # 地表のランバート反射: ρ/π · T_sun · cos(太陽天頂角)、カメラ → 地表の透過率を掛ける
            g = hit_ground
            ground_pos = c + t_end[g, None] * dirs[g]
            normal = ground_pos / np.linalg.norm(ground_pos, axis=1, keepdims=True)
            cos_sun = np.maximum(normal @ sun, 0.0)
            ground_t = self.sun_transmittance(ground_pos, sun, inner_panels, order)
            total_view_t = np.exp(-cumulative[g, -1, :])
            lum[g] += total_view_t * ground_t * cos_sun[:, None] * np.asarray(ground_albedo) / np.pi
        if components:
            return lum, ms1, (as1, hit_ground, np.exp(-cumulative[:, -1, :]))
        return lum, ms1

    def sun_transmittance(self, points_km: np.ndarray, sun: np.ndarray, panels: int, order: int) -> np.ndarray:
        """点（形状 (M, 3)）から太陽方向の透過率 (M, C)。惑星の影なら 0。"""

        r = np.linalg.norm(points_km, axis=1)
        mu = np.clip((points_km @ sun) / r, -1.0, 1.0)
        # 影の点は計算しない（光線が惑星の内部を通り、負の高度で密度の exp が溢れるため。シェーダーも同じ順序）
        lit = ~self.ray_intersects_ground(r, mu)
        t = np.zeros((r.size, self.channel_count))
        if np.any(lit):
            t[lit] = self.transmittance_to_top(r[lit], mu[lit], panels, order)
        return t


# ---------------------------------------------------------------------- 多重散乱の SH 近似（design.md §5.11）
# 並び b = 0〜5: (l, m) = (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)。Condon–Shortley の符号なし（シェーダーと同じ）
SH_BANDS = np.array([0, 1, 1, 2, 2, 2])
SH_BLOCKS = ((0, 1, 3), (2, 4), (5,))  # m = 0 / 1 / 2 のブロック（F はブロック対角）
# e_b = ∫_{z>0} Y_b·z dω（地表の照度。m > 0 は 0）
SH_HEMISPHERE_COSINE = np.array([0.886226925452758, 1.0233267079464885, 0.0, 0.4954159122921, 0.0, 0.0])


def sh_basis6(x: np.ndarray, z: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """局所の枠での方向の成分 x, z と y² から、6 つの基底の値、形状 (..., 6)"""

    return np.stack([np.full(np.shape(x), 0.28209479177387814), 0.4886025119029199 * z, 0.4886025119029199 * x,
                     0.31539156525252005 * (3.0 * z * z - 1.0), 1.0925484305920792 * x * z,
                     0.5462742152960396 * (x * x - y2)], axis=-1)


def sh_sun_grid(mu_sun: float, n_theta: int = 32, n_phi: int = 32):
    """太陽中心の方向の格子（θ = π·u²、立体角の重み、和 = 4π）。シェーダーの SH の Multi-scattering LUT と同じ並び
    （格子の枠: z' = 太陽、x' = (−μs, sinθs, 0)、y' = +Z）。(方向 (N, 3), 重み (N,))"""

    s = np.sqrt(max(0.0, 1.0 - mu_sun * mu_sun))
    sun = np.array([s, mu_sun, 0.0])
    gx, gy = np.array([-mu_sun, s, 0.0]), np.array([0.0, 0.0, 1.0])
    ring = np.arange(n_theta)
    u0, u1, uc = ring / n_theta, (ring + 1) / n_theta, (ring + 0.5) / n_theta
    theta = np.pi * uc * uc
    w_ring = (np.cos(np.pi * u0 * u0) - np.cos(np.pi * u1 * u1)) * (2.0 * np.pi / n_phi)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    local = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1, 3)
    dirs = local[:, :1] * gx + local[:, 1:2] * gy + local[:, 2:] * sun
    return dirs, np.repeat(w_ring, n_phi)


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


def mie_profile(h: np.ndarray, top_km: float, scale_height_km: float, free_trop_density: float,
                free_trop_scale_height_km: float) -> np.ndarray:
    """Mie の鉛直分布 exp(−max(h − h_top, 0)/H) + a·exp(−h/H_FT)（GPU の SampleMedium と同じ式）"""

    if top_km > 0.0:
        rho = np.exp(-np.maximum(h - top_km, 0.0) / scale_height_km)
    else:
        rho = np.exp(-h / scale_height_km)
    if free_trop_density > 0.0:
        rho = rho + free_trop_density * np.exp(-h / free_trop_scale_height_km)
    return rho


def load_phase_table(coefficient_set: str, preset: str, component: int = 0) -> np.ndarray:
    """data/atmosphere_coefficients.json のプリセットの位相関数の表（component 番目の成分）、形状 (U, 3)"""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "atmosphere_coefficients.json"
    coeff = json.loads(path.read_text(encoding="utf-8"))
    s = next(x for x in coeff["sets"] if x["name"] == coefficient_set)
    p = next(x for x in s["aerosol_presets"] if x["name"] == preset)
    return np.asarray(p["components"][component]["phase"]["values_rgb"], dtype=np.float64)


def sun_direction(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    th, ph = np.radians(zenith_deg), np.radians(azimuth_deg)
    return np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])
