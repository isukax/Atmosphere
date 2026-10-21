# Atmosphere compact

Hillaire2020 / C+A+B に固定した D3D12 大気モジュールと、単独で動くサンプルです。このフォルダだけを別の場所へコピーしてビルドできます。

固定構成は Sun disk 有効、SH L2 + ground、AP 透過率 RGB Mean、AP Shadow Per Pixel。Mie は1成分のテーブル位相関数です。空の背景の光芒、他のプリセット・方式選択、リファレンス描画は含みません。

レンダリングの順序、各 LUT の役割、散乱・影・合成の計算方法は [レンダリングと技術](RENDERING.md) を参照してください。

## ビルド

Windows、Visual Studio 2026、CMake 3.25 以上、Windows SDK の DXC、vcpkg が必要です。GPU は D3D12、SM 6.6、Resource Binding Tier 3 に対応している必要があります。

このフォルダで実行します。

```powershell
cmake --preset vs2026-x64
cmake --build --preset release
build/vs2026-x64/Release/atmosphere_compact.exe
```

`VCPKG_ROOT`、次に `%USERPROFILE%/vcpkg` の順でツールチェーンを探します。DXC を自動検出できない環境では `-DDXC_EXECUTABLE=...` を指定してください。

大気ライブラリだけをビルドする場合：

```powershell
cmake -S . -B build/core -DCOMPACT_BUILD_SAMPLE=OFF
cmake --build build/core --config Release
```

`atmosphere_compact_core` と `atmosphere_compact_gfx` は GLM と Windows の D3D12 / DXGI だけに依存します。ImGui はサンプルだけ、JSON は任意の比較ツールだけに必要です。通常ビルドで Python、EXR、元の検証プロジェクトを必要としません。

既存の vcpkg インストールを使う場合は、新しいビルドディレクトリに `-DVCPKG_MANIFEST_MODE=OFF -DVCPKG_INSTALLED_DIR=<installed>` を指定できます。

## サンプル

右ドラッグで視点、WASD / QE で移動、Shift で20倍速。表示は GT7、Auto / SDR / HDR10 を使い、モニター移動や OS の HDR 状態変更に追従します。

UI の配置は実行中だけ保持します。検証版の `imgui.ini` と共有しません。GT7 の権利表記は `THIRD_PARTY_NOTICES.md` に同梱しています。

- AOD550：0.001〜1.0 の対数スライダー、初期値0.1。Reset で0.1。
- 太陽方向・照度・視半径、露出、地表アルベド RGB を変更できます。照度の初期値は同梱の分光由来データです。
- `Planet ground albedo` は大気の地表反射モデル、`Terrain material albedo` はサンプルの地形材質です。
- Scene の `Terrain` / `Spheres` で、それぞれの表示と影を個別に切り替えます。初期状態は両方有効です。両方を無効にすると空だけを表示します。
- LUT サイズ、積分ステップ数、Per Pixel の探索点数と二分探索回数は Quality で編集し、Apply quality で反映します。
- Transmittance / Sky-View のサイズは既定の半分・既定・2倍、多重散乱タイルと AP は16・32・64。AP は立方体です。
- 既定の積分数は Transmittance 500、多重散乱40、Sky-View 30、AP 最遠スライス64、影の探索32、二分探索6。
- `--debug` は D3D12 デバッグレイヤー、`--hidden --frames 3` は非表示で指定フレームだけ実行します。`--debug --smoke` は8フレームで地形・球の表示4通り、LUT サイズ変更、ウィンドウリサイズも実行します。

## 組み込み

`src/atmosphere`、`shaders/atmosphere` と、必要な `gfx` / `shaders/common` が大気側です。`app`、`render`、`scene`、`display`、`ui` はサンプル側です。CMake の `add_subdirectory` で追加するか、この境界に沿って移植先の GPU 基盤へ置き換えます。汎用 RHI は追加していません。

公開 API は `compact::atmosphere::AtmosphereRenderer` です。

```cpp
compact::atmosphere::AtmosphereRenderer atmosphere(device, pipelines);
compact::atmosphere::AtmosphereSettings settings;
settings.aod550 = 0.1f;
atmosphere.setSettings(settings);

// Before beginFrame: changing sizes waits for GPU idle and recycles resources.
atmosphere.setQuality(quality);

compact::atmosphere::AtmosphereFrameInput frame;
frame.cameraPosM = cameraPositionM;
frame.invViewProjRot = inverseProjectionTimesRotationView;
frame.exposureScale = 1.0f / (1.2f * std::exp2(ev100));

atmosphere.updateLuts(cmd, upload, frame);
// Host renders its shadow maps and supplies frame.shadow.
atmosphere.updateAerialPerspective(cmd, upload, frame);
atmosphere.updateSkyIrradiance(cmd, upload, frame);
// Host draws opaque geometry using transmittanceLut() and skyIrradiance().
frame.depthSrv = depthSrv;
atmosphere.composite(cmd, upload, frame, sceneColorUav, width, height);
```

### 入出力の規約

- 右手系、+Y 上、ワールド距離 m。原点は半径6360 km の惑星表面、惑星座標は `worldToPlanetKm()` で変換します。大気上端は6460 km。サンプルの対応視点は大気内です。
- `sunDirectionWs` は太陽へ向かう方向。通常は単位ベクトルを渡します。明らかに単位長でない入力は設定時に正規化し、浮動小数点の丸めの範囲で単位長の値はそのまま保持します。
- 太陽照度は大気上端で太陽方向に垂直な面の lx、太陽サイズは視半径 rad。サンプル UI は度数を表示します。視半径は API で0.0008〜0.1 rad、UI は0.05〜5度です。
- 作業色空間は linear Rec.709。シーンカラーは露出済みの値 `fb = L[cd/m²] * exposureScale` を使います。合成は `geometryFb * mean(T_rgb) + inscatter * solarRgbLux * exposureScale`。表示変換はサンプル側です。
- カメラの逆行列は `inverse(projection * rotationOnlyView)`。HLSL へ行優先で転送します。深度は Reverse-Z、無限遠ファー、クリア値0。深度 SRV が無効なら全面を空として描きます。
- 天空照度は9個の float4、RGB は単位太陽照度あたりの余弦畳み込み済み実 SH 係数です。多重散乱の内部アトラスは鏡映対称の6係数で、天空照度とは異なります。サンプルの `Mesh.hlsl` が消費側の例です。
- 外部の深度・影テクスチャはシェーダーから読める状態、合成先は UAV 状態にしてください。合成先は同じピクセルを読み書きします。内部リソースの状態遷移はモジュールが管理します。
- シャドウマップは R32_FLOAT SRV、光源に近いほど深度が大きく、クリア値0。変換行列はカメラ相対 m → UV / 深度です。広域マップは必須入力ではなく、無効値なら影なし。近景マップは任意です。範囲外は日なたとして扱います。
- AP は Rayleigh 単散乱＋平均透過率、位相関数なし Mie、多重散乱の3枚。Per Pixel は累積単散乱から影の区間を差し引き、多重散乱には影を掛けません。
- AP の公称最大距離はサイズによらず128 km。2次配置、近距離フェード、最遠スライスへ比例する積分数を使います。
- `setQuality()` はフレームの記録開始前に呼んでください。サイズ変更時は GPU 完了を待ってリソースを再生成します。取得した LUT の参照・SRV をサイズ変更後まで保持せず、描画時に取得し直してください。
- AOD / Transmittance の積分数変更は透過率と多重散乱、地表アルベド / 多重散乱積分数変更は多重散乱を無効化します。太陽方向・照度・露出・ディスクサイズは媒質 LUT を再生成しません。Sky-View / AP / 天空照度は毎フレーム更新します。
- `Device` と `PipelineLibrary` は Renderer より長く生存させ、Renderer の破棄前にも GPU 完了を待ってください。コマンドリストとアップロード領域は呼び出し側が所有します。

## 固定データの再生成

`data/hillaire_cab.json` は元の係数データから Hillaire2020 / C+A+B だけを抽出したスナップショットです。通常実行では読みません。

```powershell
python tools/generate_data.py
```

Python の標準ライブラリだけで `FixedData.h` を生成します。テーブルは元の2049点、角度配置 θ = πu² を維持し、float 入力・double 累積・区間ごとの64点中点則で SH モーメントを求めます。χ₀ の正規化誤差を1に補正しません。生成ヘッダーに入力の SHA256 を記録します。

元の検証版の係数データ・生成処理・描画処理は変更していません。コピーしたコードはこのフォルダ内で独立して保守します。

## 検証

比較ツールは元の検証プロジェクトが存在する場合だけ有効化します。`COMPACT_REFERENCE_ROOT` と `COMPACT_REFERENCE_SHADERS` で場所を指定できます。既存版のソースを変更せず、別の比較実行ファイルへコンパイルします。

```powershell
cmake -S . -B build/compare -DCOMPACT_BUILD_COMPARE=ON
cmake --build build/compare --config Release
build/compare/Release/compact_dump.exe --debug --quality --out dumps/compare/compact
build/compare/Release/compact_reference_dump.exe --debug --out dumps/compare/reference
python tests/compare.py dumps/compare/reference dumps/compare/compact --report dumps/compare/report.json
```

Python の比較には NumPy を使います。比較ツール・読み戻し・レポート出力は大気ライブラリに含まれません。実測結果とコード量は `VALIDATION.md` に記録します。
