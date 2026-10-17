# Atmosphere

リアルタイム大気散乱の検証アプリ（D3D12 単独アプリ）。将来 Shoga ランタイムへ組み込む前提。

- 調査: [docs/research.md](docs/research.md)
- 設計（単位系・決定事項・検証基準・フェーズ）: [docs/design.md](docs/design.md)

## ビルド

Visual Studio 2026、vcpkg（`VCPKG_ROOT` または `%USERPROFILE%\vcpkg`）、Windows SDK（DXC 1.8 以上）が必要。

```bash
cmake --preset vs2026-x64
cmake --build --preset debug
```

## 実行

```bash
build/vs2026-x64/Debug/atmosphere.exe
```

`--debug` で D3D12 デバッグレイヤーを有効にする。出力モード（Auto / SDR / HDR10）と GT7 のピーク輝度は UI で切り替えられる。

対話モードの操作: 右ドラッグで視点、WASD / QE で移動（Shift で 20 倍）。
UI の Scene で描画方式を切り替える: テストパターン / LUT（Hillaire 2020、実用パス）/ レイマーチ（リファレンス）/ パストレース（真値、フレームごとに累積）。
空の表示では作業色空間は係数セット（Rec.709 / Rec.2020）に従う。
LUT / レイマーチでは地形とテスト球を重ねて描き、空気遠近法をかける（Scene の「Terrain & spheres」で切替。パストレースでは描かない）。
太陽の影（シャドウマップ、2 段のカスケード）は Scene の「Sun shadow」で切り替える（地形・球の面 / 空気遠近法の単散乱）。空のピクセルの光芒は扱わない（docs/design.md §5.5）。空気遠近法の影の方式は「AP shadow method」で切り替える（0: ボリューム（点）/ 1: ボリューム（ぼかし）/ 5: ピクセルごとに影の区間を差し引く。docs/design.md §5.6）。
UI の GPU time にパスごとの GPU 時間を表示する。

## 検証

```bash
python -m venv reference/.venv
reference/.venv/Scripts/python.exe -m pip install -r reference/requirements.txt
reference/.venv/Scripts/python.exe reference/validate_phase1.py
reference/.venv/Scripts/python.exe reference/validate_phase2.py
reference/.venv/Scripts/python.exe reference/validate_phase3.py
cmake --build --preset release
reference/.venv/Scripts/python.exe reference/validate_phase4.py
reference/.venv/Scripts/python.exe reference/validate_phase4_5.py
```

- Phase 1: V5 GT7 / V6 PQ / 色空間行列 → `dumps/phase1/report.json`
- Phase 2: V1 Transmittance LUT / V2 単散乱 → `dumps/phase2/report.json`（約 1.5 分）
- Phase 3: Multi-scattering LUT / V3 Sky-View LUT / V4 パストレース / V9 分光比較 → `dumps/phase3/report.json`（約 3.5 分）
- Phase 4: V3c 空気遠近法 / V3d 空からの照度 → `dumps/phase4/report.json`、V8 GPU 時間 → `dumps/phase4/perf/perf.json`（V8 は Release ビルドがあるときだけ実行）
- Phase 4.5: V3e 地表の影 / V3f 空気遠近法の影 / V3g 空気遠近法の影の方式の比較 → `dumps/phase4_5/report.json`（約 3 分）
- 調査（Phase 4）: 太陽が低いときの光輪の貫通 → `reference/analyze_ap_terrain_shadow.py`（`dumps/phase4/terrain_shadow/report.json`）

大気係数セット（`data/atmosphere_coefficients.json`）は分光データから生成する:

```bash
reference/.venv/Scripts/python.exe reference/derive_coefficients.py
```
