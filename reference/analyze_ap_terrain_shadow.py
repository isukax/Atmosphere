"""調査: 太陽が地平線近くのとき、空気遠近法の Mie の光輪が地形を「貫通」して見える（design.md §14 Phase 4 の調査）。

    reference/.venv/Scripts/python.exe reference/analyze_ap_terrain_shadow.py

validate_phase4.py が書いた dumps/phase4/aerial_perspective の sunset_front（太陽天頂角 88°、カメラ正面）を使う。
地形のピクセルごとに、視線（カメラ → 地形の点）の上の空気が地形の影に入っているかを調べ、単散乱の内散乱の
うち影に入る割合を求める。Phase 4 の実装（と論文・参照実装の既定）は、空気の点への太陽光を惑星の影と
Transmittance LUT だけで決め、地形による遮蔽（ボリュームシャドウ）を入れていない（Phase 4.5 で対応、§5.5）。

  - 視線は深度ダンプとカメラ（Phase4Tests の testCamera）から復元する。
  - 遮蔽物は地形だけ（terrain_ref.HeightGrid）。テスト球の影は入れない。地形の外（±20 km より外）は遮蔽なし。
  - 単散乱のみ。多重散乱は影の中にも入るので、ここでは影の効果に含めない（下限の見積もり）。
  - 統計は地形のピクセルだけで取る（テスト球を除く）。比較画像の補正は球のピクセルにもかける。
出力: dumps/phase4/terrain_shadow/report.json と、比較画像 docs/images/phase4_ap_terrain_shadow.png（数分かかる）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atmosphere_ref import Atmosphere, sun_direction  # noqa: E402
from exrio import read_exr  # noqa: E402
from pngio import tonemap_srgb, write_png  # noqa: E402
from terrain_ref import HeightGrid, aerial_single_scattering, reconstruct  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DUMPS = ROOT / "dumps" / "phase4" / "aerial_perspective"
OUT = ROOT / "dumps" / "phase4" / "terrain_shadow"


def main() -> int:
    case = "sunset_front"
    volume, meta = read_exr(DUMPS / f"{case}_volume.exr")
    geometry, _ = read_exr(DUMPS / f"{case}_geometry.exr")
    depth_img, depth_meta = read_exr(DUMPS / f"{case}_depth.exr")
    atmo = Atmosphere.from_meta(meta)
    a = meta["atmosphere"]
    sun = sun_direction(a["sun_zenith_deg"], a["sun_azimuth_deg"])
    sun_color = np.asarray(a.get("sun_color", [1.0, 1.0, 1.0]), dtype=np.float64)
    h, w = depth_img.shape[:2]
    cam = np.array([0.0, meta["camera_height_m"], 0.0])

    stride = 2  # 320×180 のピクセルで調べる
    mask = geometry[..., 3] > 0.5
    ys, xs = np.nonzero(mask[::stride, ::stride])
    ys, xs = ys * stride, xs * stride
    direction, distance_m, hit = reconstruct(depth_img, depth_meta["near_m"], cam, 0.0, meta["camera_pitch_deg"],
                                             meta["vertical_fov_deg"], ys, xs)
    grid = HeightGrid()
    on_terrain = np.abs(hit[:, 1] - grid(hit[:, 0], hit[:, 2])) < 30.0  # テスト球のピクセルを除く
    total, shadowed, fraction = aerial_single_scattering(atmo, cam, direction, distance_m, sun, grid, None)
    single_total = total * sun_color
    single_shadow = shadowed * sun_color
    pixel = volume[ys, xs, :3].astype(np.float64)
    surface = geometry[ys, xs, :3].astype(np.float64)

    # 太陽の方位から ±15° の地形のピクセル（光輪がかかって見える範囲）
    horizontal = direction[:, [0, 2]] / np.linalg.norm(direction[:, [0, 2]], axis=1, keepdims=True)
    sun_h = sun[[0, 2]] / np.linalg.norm(sun[[0, 2]])
    azimuth_diff = np.degrees(np.arccos(np.clip(horizontal @ sun_h, -1, 1)))
    near_sun = on_terrain & (azimuth_diff < 15.0)
    sel = on_terrain
    report = {
        "case": case,
        "sun": {"zenith_deg": a["sun_zenith_deg"], "azimuth_deg": a["sun_azimuth_deg"], "mie_g": a["mie_phase_g"]},
        "pixels": {"terrain": int(sel.sum()), "terrain_within_15deg_of_sun_azimuth": int(near_sun.sum())},
        "fraction_of_view_ray_in_terrain_shadow": {
            "terrain_mean": float(fraction[sel].mean()), "near_sun_mean": float(fraction[near_sun].mean())},
        "single_scattering_shadowed_fraction_rgb": {
            "terrain": (shadowed[sel].sum(0) / total[sel].sum(0)).tolist(),
            "near_sun": (shadowed[near_sun].sum(0) / total[near_sun].sum(0)).tolist()},
        "share_of_final_pixel_rgb": {
            "aerial_perspective_single_scattering_near_sun": (single_total[near_sun].sum(0) / pixel[near_sun].sum(0)).tolist(),
            "shadowed_single_scattering_near_sun": (single_shadow[near_sun].sum(0) / pixel[near_sun].sum(0)).tolist(),
            "surface_without_aerial_perspective_near_sun": (surface[near_sun].sum(0) / pixel[near_sun].sum(0)).tolist()},
        "notes": "single scattering only; occluder = terrain only (no spheres); flat world",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))

    # 比較画像: 左 = 今の合成、中 = 影に入る単散乱を差し引いたもの（地形・球のピクセル。球の影は入れない）、
    #           右 = 視線上の点のうち地形の影に入る割合（白 = 1）
    corrected = volume[..., :3].astype(np.float64).copy()
    shadow_map = np.zeros((h, w, 3))
    for y0, x0, s, f in zip(ys, xs, single_shadow, fraction):
        corrected[y0:y0 + stride, x0:x0 + stride] -= s
        shadow_map[y0:y0 + stride, x0:x0 + stride] = f
    corrected = np.maximum(corrected, 0.0)
    scale = 0.18 / max(np.median(volume[..., :3].mean(-1)), 1e-12)
    gap = np.ones((h, 4, 3))
    write_png(ROOT / "docs" / "images" / "phase4_ap_terrain_shadow.png",
              np.concatenate([tonemap_srgb(volume[..., :3] * scale), gap, tonemap_srgb(corrected * scale), gap,
                              shadow_map], axis=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
