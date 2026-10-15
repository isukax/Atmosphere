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

## 検証

```bash
python -m venv reference/.venv
reference/.venv/Scripts/python.exe -m pip install -r reference/requirements.txt
reference/.venv/Scripts/python.exe reference/validate_phase1.py
reference/.venv/Scripts/python.exe reference/validate_phase2.py
reference/.venv/Scripts/python.exe reference/validate_phase3.py
```

- Phase 1: V5 GT7 / V6 PQ / 色空間行列 → `dumps/phase1/report.json`
- Phase 2: V1 Transmittance LUT / V2 単散乱 → `dumps/phase2/report.json`（約 1.5 分）
- Phase 3: Multi-scattering LUT / V3 Sky-View LUT / V4 パストレース / V9 分光比較 → `dumps/phase3/report.json`（約 3.5 分）

大気係数セット（`data/atmosphere_coefficients.json`）は分光データから生成する:

```bash
reference/.venv/Scripts/python.exe reference/derive_coefficients.py
```
