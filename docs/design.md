# 大気散乱 検証アプリ 設計書

- 作成日: 2026-10-15
- 前提文書: [research.md](research.md)
- 状態: 確定（§12）。Phase 1 から実装

---

## 1. 決定事項

| # | 項目 | 決定 | 理由 |
|---|---|---|---|
| D1 | 単位系 | **物理単位**（SI + 測光量）。光は lx / cd·m⁻²、距離は m（大気内部計算のみ km） | ユーザー指定。Shoga 統合時にライト・露出・IBL と一貫させるため |
| D2 | アプリ形態 | **単独の D3D12 アプリ**（Shoga の RHI/RenderGraph は使わない） | ユーザー指定。大気モジュールだけは Shoga へ移植しやすい境界で切る（§9） |
| D3 | 大気手法 | Hillaire 2020（Transmittance / Multi-scattering / Sky-View / Aerial Perspective の 4 LUT） | research.md §2 |
| D4 | 比較用リファレンス | アプリ内 GPU 総当たりレイマーチ（LUT なし）と GPU パストレーサー + Python による倍精度数値積分 | LUT による誤差と多重散乱近似の誤差を分けて測るため |
| D5 | トーンマップ | **GT7 Tone Mapping**（Polyphony Digital 公開サンプル、MIT。ICtCp UCS・既定パラメータ） | ユーザー指定 |
| D6 | 出力 | **SDR**（8bit sRGB）と **HDR10**（10bit PQ / Rec.2020）。表示先ディスプレイの状態から自動選択し、手動切替も可能 | ユーザー指定 |
| D7 | ディスプレイ輝度変化への追従 | 出力先ディスプレイの変化（移動 / OS の HDR 切替 / 設定変更）を検出し、GT7 カーブのピーク輝度とスワップチェーン形式を再構成する。ピークはユーザーが上書きできる | ユーザー指定。OS 報告値（`MaxLuminance`）は信頼できない場合がある |
| D8 | 露出 | 手動 EV100。`fb = L · 1/(1.2·2^EV100)`（GraphicsTest/HDR テストベッドと同一の規約） | プロジェクト間で数値を比較できるようにするため |
| D9 | 作業色空間 | **linear Rec.709 / linear Rec.2020 を実行時に切替可能**（既定 Rec.709）。散乱係数は色空間ごとに分光データから導出したセットを使う（§2.6） | Rec.709 は Hillaire の係数で参照実装と直接比較するため。Rec.2020 は GT7 の入力色空間と一致し、広色域での空の色を検証するため |
| D10 | 座標系 | 右手系、+Y 上、ワールド単位 m。深度は **Reverse-Z + 無限遠ファープレーン**（D32_FLOAT） | 数十 km の遠景で深度精度を確保するため |
| D11 | シェーダー | HLSL SM 6.6、DXC でビルド時にコンパイル。**Bindless**（`ResourceDescriptorHeap[]`） | Shoga の Bindless 前提と揃えるため |
| D12 | 言語 / ビルド | C++20、CMake + vcpkg マニフェスト、改行 LF | GraphicsTest の既存プロジェクトと揃えるため |
| D13 | 要所のコメント | `// [POINT]` マーカーで「何を・なぜ・単位」を書く（§10） | ユーザー指定 |

---

## 2. 単位系

### 2.1 物理量と単位

| 量 | 記号 | 単位 | 備考 |
|---|---|---|---|
| ワールド座標・カメラ位置・オブジェクト寸法 | x | **m** | API・シーン・頂点はすべて m |
| 大気計算内部の距離・高度・惑星半径 | r, h, t | **km** | シェーダー内の大気関数のみ。m→km の変換は 1 箇所（`AtmosphereConstants` 構築時）で行う |
| 散乱・吸収・消散係数 | σs, σa, σt | **km⁻¹** | 例: Rayleigh σs = (5.802, 13.558, 33.1)×10⁻³ km⁻¹ |
| スケールハイト | H | km | Rayleigh 8 km、Mie 1.2 km |
| 光学的厚さ / 透過率 | τ / T | 無次元 | T = exp(-τ) |
| 位相関数 | p | **sr⁻¹** | ∫p dω = 1 |
| 太陽照度（大気上端、太陽方向に垂直な面） | E_sun | **lx**（= lm·m⁻²） | 既定 128,000 lx。文献値に幅あり（約 127–134 klx）、UI で変更可 |
| 輝度（空・地表・画面上の値） | L | **cd·m⁻²**（= nit） | RGB の各値は「Y が輝度になる RGB」（Rec.709 の Y 係数で cd/m²） |
| 単位照度あたり輝度（LUT の格納値） | L/E | **sr⁻¹**（= cd·m⁻²/lx） | §2.3 参照 |
| 地表アルベド | ρ | 無次元 | ランバート: L = ρ/π · E |
| 太陽視半径 | α_sun | rad | 既定 0.004675 rad（参照実装と同値） |
| 露出 | EV100 | EV（ISO 100 基準） | 晴天日中 ≈ 15 |
| GT フレームバッファ値 | fb | 無次元（**1.0 = 100 cd/m² の表示輝度**） | GT7 サンプルの規約（`REFERENCE_LUMINANCE = 100`） |
| ディスプレイ輝度 | — | nit | SDR は GT7 規約でペーパーホワイト 250 nit を想定 |

注: RGB の 3 値は「作業色空間の RGB」。Hillaire の既定係数は波長 680 / 550 / 440 nm でサンプルした値を Rec.709 の RGB として扱う近似で、厳密な分光計算ではない（§2.6）。

### 2.2 変換式（実装のすべての単位変換はこの表のどれかに当たる）

```
x_km        = x_m · 0.001
P_planet_km = x_world_m · 0.001 + (0, R_ground_km + h0_km, 0)      // ワールド原点は惑星表面 +h0 の真上（北極点）
L_sky       = LUT(L/E)[sr⁻¹] · E_sun[lx]                           → cd/m²
L_sun_disk  = E_sun / Ω_sun · T(view→TOA),  Ω_sun = 2π(1 − cos α_sun) ≈ 6.87×10⁻⁵ sr
              → 約 1.86×10⁹ cd/m²（大気上端）
L_ground    = ρ/π · (E_sun · T_sun · max(N·L, 0) + E_sky)           → cd/m²
k_exposure  = 1 / (1.2 · 2^EV100)                                    [1/(cd/m²)]
fb          = L · k_exposure                                         （シーンバッファの格納値、§2.4）
display_nit = GT7(fb) · 100                                          （HDR。SDR は GT7 側で [0,1] に正規化）
```

露出の目安（D8 の規約による）:
- EV100 = 15 では L_sat = 1.2·2¹⁵ ≈ 39,300 cd/m² が fb 1.0（= 100 nit 表示）になる。
- 反射式露出計（K = 12.5）で測った平均輝度 L_avg = 0.125·2^EV100 は fb ≈ 0.104 になる。暗すぎる場合は露出補正（EV バイアス）で調整する。この規約の見直しは §12 Q5。

### 2.3 LUT は「単位照度あたり」で持つ

Sky-View LUT と Aerial Perspective の内散乱は **E_sun = 1 lx として計算**する（格納値の単位 sr⁻¹）。合成時に E_sun と k_exposure を掛ける。

- FP16 LUT の値域に収まる（日中の天空で 10⁻²〜10⁰ 程度。太陽周辺の Mie ピークも 65504 を大きく下回る）。
- LUT が露出にも太陽照度にも依存しないので、露出を変えても LUT の再計算が不要になる。
- 制約: 単一光源（太陽）前提。月などを追加する場合は光源ごとの LUT か、照度込みの格納に変える（範囲外）。

### 2.4 シーンバッファとプリエクスポージャ

- シーンカラー RT（R16G16B16A16_FLOAT）には **露出済みの fb 値** を書く。物理輝度（太陽ディスクは 10⁹ cd/m²）をそのまま書くと FP16 の上限 65504 を超えるため。
- 手動露出なので、プリエクスポージャ = 当該フレームの露出（ディスプレイ段での再スケールは 1.0）。将来自動露出を入れる場合は「前フレームの露出でプリエクスポーズし、ディスプレイ段で `exposure / preExposure` を掛ける」一般形にする。
- 太陽ディスクは露出後も FP16 上限を超えうる（EV100 < 約 14.5 で超える）。書き込み前に `min(fb, 65000)` で**明示的にクランプ**する（inf/NaN の伝播を防ぐ）。

### 2.5 命名規約（単位をコードで読めるようにする）

| 接尾辞 | 単位 | 例 |
|---|---|---|
| `M` | m | `cameraPosM`, `terrainSizeM` |
| `Km` | km | `bottomRadiusKm`, `viewHeightKm` |
| `PerKm` | km⁻¹ | `rayleighScatteringPerKm` |
| `Lux` | lx | `sunIlluminanceLux` |
| `Nits` | cd/m² | `skyLuminanceNits`, `displayPeakNits` |
| `PerLux` | sr⁻¹（cd/m²/lx） | `skyViewLutPerLux` |
| `Fb` | GT fb（1 = 100 nit） | `sceneColorFb` |
| `Ev100` | EV | `exposureEv100` |
| `Rad` / `Deg` | 角度 | `sunZenithRad`（UI は Deg） |

C++ 側は型エイリアスではなく接尾辞で表す（型を増やさない）。Python・ダンプのメタデータでも同じ名前を使う。

### 2.6 作業色空間の切替（Rec.709 / Rec.2020）

**散乱係数は 3×3 行列では変換できない。** 透過率 `exp(-σ·t)` は σ について非線形で、係数は色ではなく波長ごとの物理量なので、色の変換行列を掛けても正しくならない。色空間ごとに分光データから係数セットを導出する。

係数セット（`reference/derive_coefficients.py` で生成し、`data/atmosphere_coefficients.json` に出力してコミットする）:

| セット名 | 作業色空間 | Rayleigh / オゾン係数 | 太陽の色 | 用途 |
|---|---|---|---|---|
| `Hillaire2020` | Rec.709 | 論文値（680/550/440 nm のサンプル） | 白（D65、Y = 1） | 論文・参照実装との比較（V3/V4） |
| `SpectralFit709` | Rec.709 | 分光フィット（下記） | 大気上端の太陽スペクトル → RGB（Y = 1） | 色空間比較 |
| `SpectralFit2020` | Rec.2020 | 分光フィット（下記） | 同上 | 色空間比較 |

- 分光データ: Rayleigh は `σ(λ) = 1.24062×10⁻⁶ m⁻¹ · (λ/1µm)⁻⁴`、オゾン吸収断面積と大気上端の太陽分光放射照度は Bruneton 2017 実装（BSD-3、出典を明記）の表、等色関数は CIE 1931 2°（colour-science）。Mie は波長に依存しない（灰色）ので全セットで共通値。
- フィット方法: 作業色空間の各チャンネル c について、太陽スペクトルを光学的距離 d だけ通したときの分光透過光を RGB に射影し、`T_c(d) = exp(-σ_c·d)` が最小二乗（log 空間、d は天頂〜地平線方向の範囲）で一致する σ_c を求める。
- 参考: Rec.2020 の原色はスペクトル軌跡上の単色光（630 / 532 / 467 nm）なので、「原色波長でのサンプル値」がフィット値に近くなる見込み。Rec.709 の原色は単色光ではない。この差を V9 で測る。
- 作業色空間に依存するもの（切替時にすべて更新し、大気 LUT は dirty にする）:

| 対象 | Rec.709 | Rec.2020 |
|---|---|---|
| 大気係数・太陽の色 | 上表のセット | 上表のセット |
| 輝度の重み（Y、nit 表示、AP の平均） | (0.2126, 0.7152, 0.0722) | (0.2627, 0.6780, 0.0593) |
| 作業 → Rec.2020（GT7 入力） | BT.2087 の 3×3 行列 | 単位行列 |
| アルベド | 灰色のみ使うため共通。色付きアルベドを UI で入力する場合は Rec.709 で入力して変換 | |

- SDR 出力時は GT7 の出力（Rec.2020）を Rec.709 にクリップするため、Rec.2020 作業時の高彩度（夕焼け等）は SDR ではクリップされる。これは表示側の制約で、作業色空間の欠陥ではない。
- サイドカー JSON には作業色空間と係数セット名を必ず記録する。

---

## 3. 全体構成

```
Atmosphere/
├─ docs/                     research.md, design.md
├─ CMakeLists.txt / CMakePresets.json / vcpkg.json
├─ src/
│  ├─ main.cpp               起動引数（対話 / バッチ）
│  ├─ app/                   Window（Win32）, App（フレームループ, 入力, カメラ）
│  ├─ gfx/                   D3D12 の薄い層（Shoga へは移植しない）
│  │  ├─ Device              デバイス, キュー, フェンス, フレーム同時実行数 2
│  │  ├─ BindlessHeap        シェーダー可視 CBV/SRV/UAV ヒープ + インデックス割当
│  │  ├─ Resources           Texture / Buffer 生成, アップロード, 状態遷移
│  │  ├─ Shaders             DXIL の読み込み, ルートシグネチャ（Bindless 用 1 つ）, PSO
│  │  ├─ GpuTimer            タイムスタンプクエリ（パス別 ms）
│  │  └─ Readback            RT → CPU（バッチダンプ用）
│  ├─ display/               出力（§7）
│  │  ├─ DisplayMonitor      出力先ディスプレイの検出・変化通知
│  │  ├─ Swapchain           SDR / HDR10 の作成・再作成, ColorSpace, HDR メタデータ
│  │  └─ DisplayPass         GT7 → エンコード → UI 合成
│  ├─ atmosphere/            ★ Shoga へ移植する単位（§9）
│  │  ├─ AtmosphereParams    物理パラメータ（単位付き）, 地球プリセット
│  │  ├─ AtmosphereConstants GPU 定数への変換（m→km 変換はここだけ）
│  │  └─ AtmosphereRenderer  LUT 4 種の生成 + 空/空気遠近法の合成 + リファレンス描画
│  ├─ scene/                 Terrain（手続き生成）, TestObjects（球）, Camera, Sun
│  ├─ render/                SceneRenderer（地形のフォワード描画）, Exposure
│  ├─ ui/                    ImGui（パラメータ, 計測値, 単位表示, ピクセルピッカー）
│  └─ batch/                 シナリオ実行 → ダンプ（§8）
├─ shaders/
│  ├─ common/                Units.hlsli, Bindless.hlsli, ColorSpace.hlsli, Noise.hlsli
│  ├─ atmosphere/            AtmosphereCommon.hlsli（媒質, 交差, LUT パラメタ化）, Phase.hlsli,
│  │                         TransmittanceLut.cs, MultiScatteringLut.cs, SkyViewLut.cs,
│  │                         AerialPerspective.cs, SkyComposite.ps, ReferenceRaymarch.cs, PathTrace.cs
│  ├─ scene/                 Terrain.vs/ps, Object.vs/ps
│  └─ display/               GT7ToneMap.hlsli, PQ.hlsli, SRGB.hlsli, Display.ps, UIComposite
├─ data/                     atmosphere_coefficients.json（Python 生成・コミット）, scenarios/*.json
└─ reference/                Python（numpy, colour-science）: atmosphere_ref.py, derive_coefficients.py,
                             gt7_tone_mapping.py, compare.py, tests/
```

---

## 4. フレームの流れ

```
CPU: 入力・UI → パラメータ更新 → 変更検出（大気パラメータ変更なら LUT dirty）
     DisplayMonitor をポーリング → 変化があればスワップチェーン / GT7 定数を再構成

[Compute] 1. TransmittanceLut      … 大気パラメータ変更時のみ
[Compute] 2. MultiScatteringLut    … 大気パラメータ変更時のみ（1 に依存）
[Compute] 3. SkyViewLut            … 毎フレーム（カメラ高度・太陽方向に依存）
[Compute] 4. AerialPerspective     … 毎フレーム（カメラ・太陽に依存）
[Graphics] 5. Scene                … 地形・球を SceneColor(fb) / Depth に描画
                                      太陽光は「シェーディング点での透過率」で減衰（per-pixel）
[Graphics] 6. SkyComposite         … 深度 = 0（Reverse-Z の無限遠）: Sky-View LUT + 太陽ディスク
                                      それ以外: color·T_ap + inscatter_ap
[Graphics] 7. Display              … fb → Rec.2020 → GT7 → (PQ | sRGB+ディザ) → バックバッファ
[Graphics] 8. UI                   … ImGui を UI テクスチャへ → Display 内で合成
```

リファレンスモード（UI で切替）では 1–4 と 6 の代わりに:
- **Raymarch**: ピクセルごとに LUT を使わず多数ステップで積分（同じ多重散乱近似。LUT 化による誤差だけを見る）
- **PathTrace**: デルタトラッキングで多重散乱を厳密に計算し、フレーム間で累積（近似そのものの誤差を見る）

さらに差分表示（相対誤差のヒートマップ / EV 差）で A/B 比較する。

---

## 5. 大気モジュール詳細

### 5.1 パラメータ（`AtmosphereParams`、既定値 = 地球）

| パラメータ | 既定値 | 単位 |
|---|---|---|
| bottomRadiusKm / topRadiusKm | 6360 / 6460 | km |
| rayleighScatteringPerKm | (5.802, 13.558, 33.1)×10⁻³ | km⁻¹ |
| rayleighScaleHeightKm | 8.0 | km |
| mieScatteringPerKm / mieAbsorptionPerKm | 3.996×10⁻³ / 0.444×10⁻³ | km⁻¹ |
| mieScaleHeightKm / miePhaseG | 1.2 / 0.8 | km / — |
| ozoneAbsorptionPerKm | (0.650, 1.881, 0.085)×10⁻³ | km⁻¹ |
| ozoneCenterKm / ozoneHalfWidthKm | 25 / 15 | km |
| groundAlbedo | 0.3 | — |
| sunIlluminanceLux / sunAngularRadiusRad | 128,000 / 0.004675 | lx / rad |
| sunZenithDeg / sunAzimuthDeg | UI | deg |
| originAltitudeM (h0) | 0 | m |

Mie 吸収は参照実装の値（消散 4.44、散乱 3.996）を正とする（research.md §3.1）。

### 5.2 LUT 仕様

| LUT | 形式 | 既定サイズ | 格納値（単位） | パラメタ化 | 更新 |
|---|---|---|---|---|---|
| Transmittance | RGBA16F（RGB 使用） | 256×64 | T（無次元） | Bruneton 2017: u = 地平線までの距離比、v = 高度（地平線付近に集中） | 大気変更時 |
| MultiScattering | RGBA16F | 32×32 | Ψ_ms（sr⁻¹、単位照度あたり） | u = cos(太陽天頂角)、v = 高度 | 大気変更時 |
| SkyView | RGBA16F | 192×108 | L/E（sr⁻¹） | u = 太陽との相対方位、v = 緯度（`sqrt` で地平線に集中） | 毎フレーム |
| AerialPerspective | RGBA16F 3D | 32×32×32 | RGB = 内散乱 L/E（sr⁻¹）、A = 透過率の RGB 平均 | 画面 UV × 深度（線形、1 km/スライス、0–32 km） | 毎フレーム |

- ステップ数（論文 PC 値）: Transmittance 40、MultiScattering 20（64 方向）、SkyView 30、AP 30。UI で変更でき、誤差-コスト曲線を測る。
- サブテクセル UV 補正（`fromUnitToSubUvs`）を全 LUT に適用する（端のテクセル中心をパラメータ端に合わせる）。
- 32 km を超える距離は最終スライスにクランプする（地形範囲が 40 km 四方なので、検証範囲ではほぼ影響しない。超える部分は差分表示で確認）。

### 5.3 合成時の要点

- 太陽ディスクは Sky-View LUT に含めず解析的に描画する（`cosθ > cos α_sun`）。周縁減光はしない。
- 空気遠近法の最初のスライス手前は内散乱を距離で線形フェードする（スライス境界の段差防止）。
- 地表の太陽光は `T(シェーディング点 → 太陽)` を Transmittance LUT から per-pixel で引く（カメラ位置 1 回ではなく。地形の高低差で色が変わることを確認するため）。惑星の影（太陽が地平線下）も同じ LUT 引きで 0 になる。
- 天空からの拡散照度 E_sky（地形の環境光）は Phase 4 で、Sky-View LUT を余弦重み積分した L2 球面調和（9 係数）として計算する。それまでは E_sky = 0（論文 Fig.3 と同条件）。
- カメラ高度は大気上端未満に制限する（宇宙視点は範囲外、§11）。

### 5.4 リファレンス

| 種類 | 何を検証するか | 実装 |
|---|---|---|
| Python 数値積分（倍精度） | 透過率・単散乱の正解値 | `reference/atmosphere_ref.py`。GPU と同じ式・パラメータ。格子点の値を JSON で出力 |
| GPU Raymarch | LUT 化（解像度・パラメタ化・補間）の誤差 | 多重散乱近似は LUT と同一、ステップ数 256 以上 |
| GPU PathTrace | 多重散乱近似（等方仮定）の誤差 | デルタトラッキング + 比率トラッキング、次数 N 指定、フレーム累積 |

---

## 6. シーン

- 地形: 40 km × 40 km の格子（512² 頂点、約 78 m 間隔）。fBm で最大標高 約 3,000 m。アルベドは定数（既定 0.18）。法線は高さから計算。
- テスト球: アルベド 0.18 / 0.8 の球をカメラ近傍に配置（露出・トーンマップの確認用。アルベド既知）。
- カメラ: フライカメラ（m/s 単位の移動速度）。UI に高度（m）、EV100、太陽天頂角を常時表示。
- 影: 地形のセルフシャドウ・ボリュームシャドウ（光芒）は初期範囲外（§11）。

---

## 7. 表示パイプライン（SDR / HDR、GT7）

### 7.1 ディスプレイ処理の流れ

```
SceneColor (fb, linear 作業色空間)
  │ × (exposure / preExposure)            … 手動露出では 1.0
  ▼
作業色空間 → linear Rec.2020（3×3 行列。Rec.2020 作業時は単位行列）
  ▼
GT7 Tone Mapping（mode = SDR | HDR(peakNits)）
  ├─ SDR: 出力 [0,1]（1.0 = SDR 白 = ペーパーホワイト 250 nit 想定）
  │        → Rec.2020 → Rec.709 → 負値・1 超を clamp → sRGB OETF
  │        → ブルーノイズディザ（±0.5 LSB）→ R8G8B8A8_UNORM
  └─ HDR: 出力 fb（[0, peakNits/100]）→ ×100 = nit
           → PQ（ST 2084, nit/10000）→ R10G10B10A2_UNORM（Rec.2020）
  ▼
UI 合成（ImGui は R8G8B8A8 の UI テクスチャに描き、ここで合成）
  - SDR: sRGB 空間でアルファ合成
  - HDR: UI を linear 化 × uiWhiteNits（既定 200 nit）→ Rec.2020 → シーンの nit と線形合成 → PQ
```

### 7.2 GT7 Tone Mapping の実装方針

- 正本: Polyphony Digital の公開サンプル（SIGGRAPH 2025 PBS Course、MIT ライセンス）。HLSL は GraphicsTest/HDR/testbed の `tonemap.hlsli`（`toneMapGT7`）から移植し、ライセンス表記を付ける。
- パラメータ（サンプル既定値）: alpha 0.25、midPoint 0.538、linearSection 0.444、toeStrength 1.280、blendRatio 0.6、fadeStart 0.98、fadeEnd 1.16、UCS = ICtCp。
- モード:
  - SDR: target = 250 nit（`GRAN_TURISMO_SDR_PAPER_WHITE`）、出力に 1/2.5 を掛けて [0,1] にする。
  - HDR: target = ディスプレイのピーク輝度（nit）。**下限 250**（サンプルの注記: カーブパラメータが 250 nit 前提で決められているため）、上限 10,000。
- カーブ定数（kA, kB, kC, k）と target の UCS 輝度（`framebufferLuminanceTargetUcs`）は、**ピーク輝度が変わったときだけ CPU で再計算**して定数バッファで渡す（ピクセルごとに `rgbToICtCp(target)` を計算しない）。
- 正しさの検証は、GraphicsTest/LUTGenGT7 の `gt7_tone_mapping.py`（元の C++ と照合済み）と格子点で比較して行う（§8）。

### 7.3 ディスプレイの検出と変化への追従（D7）

検出のきっかけ:
1. `WM_MOVE` / `WM_SIZE` / `WM_DISPLAYCHANGE` → `IDXGISwapChain::GetContainingOutput` で出力先が変わったか
2. 毎フレーム `IDXGIFactory::IsCurrent()` が false → ファクトリを作り直して再列挙（OS の HDR 切替・ディスプレイ構成変更で false になる）

状態の取得（`IDXGIOutput6::GetDesc1`）:
- `ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020` → OS の HDR が有効
- `MaxLuminance` / `MaxFullFrameLuminance` / `MinLuminance`（nit）

再構成ルール:

| 変化 | 処理 |
|---|---|
| OS HDR 有効 ↔ 無効 | 自動モードなら出力モードを切替（スワップチェーンの形式と ColorSpace を作り直す） |
| ピーク輝度の変化 | GT7 の target を更新（スワップチェーンの再作成は不要） |
| 出力先ディスプレイの変化 | 上記 2 つを両方評価 |

- ユーザー上書き: UI で「出力モード（Auto / SDR / HDR10）」と「ピーク輝度（Auto = OS 報告値 / 手動 nit）」を選べる。ゲーム機の HDR キャリブレーション（HGIG）に相当する手動値を優先できるようにする。
- HDR10 メタデータ（`SetHDRMetaData`）: MaxMasteringLuminance = 使用中のピーク、MaxCLL = ピーク。GT7 の出力はピークを超えないため。
- SDR 出力時は ColorSpace を `RGB_FULL_G22_NONE_P709` にする。OS HDR 有効で SDR を強制した場合の明るさは Windows の「SDR コンテンツの明るさ」に従う（アプリでは制御しない）。

---

## 8. 検証計画とバッチモード

### 8.1 バッチモード

```
atmosphere.exe --batch scenarios/noon.json --out dumps/noon/
```
- シナリオ JSON: カメラ（位置 m、向き）、太陽角（deg）、大気パラメータ、露出、描画モード（LUT / Raymarch / PathTrace(spp)）、出力モード。
- ダンプ: float32 オフスクリーン RT から読み戻し、EXR + サイドカー JSON（GraphicsTest/HDR と同じスキーマ思想）。サイドカーには**格納値の単位**（`sr^-1 (per lux)` / `cd/m2` / `fb (1=100cd/m2)` / `nits`）と色空間を必ず書く。
- 比較は Python（`reference/compare.py`）で行い、C++ 側にテストフレームワークは持たない。

### 8.2 合格基準

| ID | 比較 | 基準 |
|---|---|---|
| V1 | Transmittance LUT（GPU）vs Python 倍精度 | 格子点で相対誤差 < 1×10⁻³ |
| V2 | 単散乱 Raymarch（GPU）vs Python | 代表方向で相対誤差 < 1×10⁻³ |
| V3 | LUT 描画 vs GPU Raymarch（同じ近似） | RGB 別 RMSE を記録。地平線付近の 1D プロファイルを保存 |
| V4 | LUT 描画 vs PathTrace | 論文と同じシナリオ（昼・夕方・地球の影）で RGB 別 RMSE が論文と同程度（10⁻³ オーダー、単位照度あたりで正規化） |
| V5 | GT7 HLSL vs `gt7_tone_mapping.py` | SDR / HDR 450・1000・2000 の格子点（33³）で知覚色差 ΔE_ITP（BT.2124）< 0.5。チャンネル別の絶対誤差は参考値（下記注） |
| V6 | PQ エンコード vs Python | 10bit コード差 ≤ 1 LSB |
| V7 | ディスプレイ変化 | OS HDR 切替・ウィンドウの別ディスプレイへの移動・ピーク手動変更で、再起動なしに正しいモード/ピークへ追従（目視 + ログ） |
| V8 | 性能 | パス別 GPU 時間（1080p / 4K）。論文値（GTX 1080 で LUT 合計 約 0.17 ms）と比較 |
| V9 | 作業色空間（§2.6）: 各係数セットの GPU 描画（透過率 + 単散乱）vs Python 分光リファレンス（5 nm 刻み → XYZ → 作業色空間） | 色差 ΔE_ITP と輝度の相対誤差を、昼・夕方・地平線で記録。合否ではなく比較データとして残す |

注: AMD RDNA3 では FP16 への書き込みが切り捨て丸めになる（GraphicsTest/HDR の実測）。数値比較用のダンプは必ず float32 の RT から読み戻す。

注（V5 の基準を ΔE_ITP にした理由、2026-10-15 実測）: 当初の基準「チャンネル別 abs < 1×10⁻³ fb」では HDR が不合格（最大 1.0×10⁻² fb @ HDR2000）になった。原因は、高彩度色で ICtCp → RGB の逆変換をするときに、ほぼ 0 のチャンネルで桁落ちが起きること。float32 と float64 の Python 参照どうしでも PQ 10bit で最大 70 コード（0.01 nit 程度）ずれており、GPU 固有の誤りではない。画素全体の知覚色差では、GPU vs float64 参照が最大 0.064、float32 参照 vs float64 参照が最大 0.072 で同程度だった。

---

## 9. Shoga への移植境界

`src/atmosphere/` と `shaders/atmosphere/` だけを移植対象にする。依存は以下に限る。

| 大気モジュールが要求するもの | 検証アプリ | Shoga での対応先 |
|---|---|---|
| コンピュート / グラフィックスのディスパッチ, リソース生成 | `gfx/`（薄い D3D12 ラッパー） | RHI のコマンド IR |
| パス間の依存と状態遷移 | 手書き（§4 の固定順） | RenderGraph のパス宣言 |
| Bindless インデックス | `BindlessHeap` | Shoga の Bindless ヒープ |
| 太陽の方向・照度 | `scene/Sun` | `ExtractedDirectionalLight`（照度 lx の追加が必要） |
| 深度・シーンカラー | アプリの RT | SceneTextures |
| 露出 | `render/Exposure` | ViewPipeline の Exposure（物理単位化が必要） |

`AtmosphereRenderer` の公開 API 案（最小）:
```cpp
// [POINT] 大気モジュールの入出力はすべて単位付きの名前にする（2.5 の命名規約）。
struct AtmosphereFrameInput {
    glm::vec3 cameraPosM;          // ワールド座標 [m]
    glm::mat4 viewProj;            // Reverse-Z
    glm::vec3 sunDirectionWs;      // 太陽「へ向かう」方向（正規化）
    float     sunIlluminanceLux;   // 大気上端での照度 [lx]
    float     exposureScale;       // 1/(1.2·2^EV100) [1/(cd/m²)]
    uint32_t  depthSrvIndex;       // Bindless
};
class AtmosphereRenderer {
public:
    void setParams(const AtmosphereParams&);           // 変更検出 → LUT dirty
    void updateLuts(GpuContext&, const AtmosphereFrameInput&);   // §4 の 1–4
    void composite(GpuContext&, const AtmosphereFrameInput&, uint32_t sceneColorUav); // §4 の 6
    AtmosphereLutIndices lutIndices() const;           // 地形シェーダーが透過率 LUT を引くため
};
```

---

## 10. コメント規約（要所のポイント）

- マーカー `// [POINT]` を付け、「何をしているか / なぜそうするか / 単位」を書く。言語は日本語（理解目的。§12 Q3）。
- 必ず `[POINT]` を書く箇所:
  1. 単位変換（m→km、L/E × E_sun、露出、fb↔nit、nit→PQ）
  2. LUT パラメタ化（Transmittance の地平線距離写像、Sky-View の非線形緯度、サブテクセル UV 補正）
  3. 多重散乱近似（L₂ と f_ms の積分、`1/(1−f_ms)`、等方仮定の限界）
  4. 数値的な注意（レイ-球交差の桁落ち、地表オフセット、FP16 クランプ、NaN 対策）
  5. Reverse-Z と無限遠の扱い（空ピクセルの判定 depth == 0）
  6. GT7（SDR 補正 1/2.5、ピーク下限 250、カーブ定数の CPU 事前計算）
  7. ディスプレイ変化の検出と再構成
  8. cbuffer の `packoffset` と C++ 側 `static_assert(offsetof)`（GraphicsTest/HDR の教訓）
- 例:

```hlsl
// [POINT] Sky-View LUT は「太陽照度 1 lx あたりの輝度」[sr^-1] を格納している。
//         ここで E_sun [lx] を掛けて cd/m^2 にし、さらに露出 [1/(cd/m^2)] を掛けて
//         GT フレームバッファ値 (1.0 = 100 cd/m^2) にする。
//         LUT を照度・露出から独立させることで、露出変更で LUT を作り直さずに済み、
//         FP16 の値域 (max 65504) にも収まる。
float3 skyFb = SampleSkyViewLut(uv) * (gSunIlluminanceLux * gExposureScale);
```

---

## 11. 範囲外（今回はやらない）

宇宙視点（大気上端より上）、夜空・月・星、雲、ハイトフォグ / ボリュメトリックフォグ、ボリュームシャドウ（光芒）、地形のセルフシャドウ、自動露出、ホワイトバランス、ブルーム / グレア、スペクトルレンダリング、モバイル品質検証。

---

## 12. 確認済み事項（2026-10-15 回答）

| # | 内容 | 回答 |
|---|---|---|
| Q1 | 外部ライブラリ | 使用可。C++: imgui（dx12 + win32）、tinyexr、glm、nlohmann-json（シナリオ・サイドカー JSON）を vcpkg で使う。DXC は Windows SDK 同梱版（1.8、SM 6.6 対応）を使い、無い場合は vcpkg の directx-dxc に切り替える。d3d12.h は Windows SDK のものを使う（directx-headers は不要になった）。Python: numpy、colour-science、OpenEXR（`reference/.venv`） |
| Q2 | GT7 の HLSL | GraphicsTest/HDR からコピーして移植（出典・MIT 表記を付ける）。Python 参照は GraphicsTest/LUTGenGT7 の `gt7_tone_mapping.py` をコピー |
| Q3 | `[POINT]` コメントの言語 | 日本語 |
| Q4 | 進め方 | Phase 1 から順に、フェーズごとに検証してから次へ進む |
| Q5 | 露出の規約 | D8 のとおり（`1/(1.2·2^EV100)`） |
| Q6 | git | `git init` する（コミットは指示があるまでしない） |
| Q7 | 作業色空間 | Rec.2020 への切替に対応する（§2.6） |

---

## 13. 実装フェーズ

| Phase | 内容 | 完了条件 |
|---|---|---|
| 1 | CMake/vcpkg、Device、Bindless、Swapchain（SDR/HDR10）、DisplayMonitor、GT7 + PQ/sRGB、ImGui、テスト画像（輝度ランプ） | V5, V6, V7（2026-10-15: 実装済み。V5・V6 合格、V7 は OS の HDR 切替を含む手動確認待ち） |
| 2 | 大気パラメータ・単位系、係数セット導出（§2.6）、Transmittance LUT、GPU Raymarch リファレンス（単散乱）、Python リファレンス、バッチダンプ | V1, V2 |
| 3 | MultiScattering / Sky-View LUT、太陽ディスク、PathTrace、作業色空間の切替 | V3, V4（空のみ）, V9 |
| 4 | 地形・テスト球、AP ボリューム、地表の透過率減衰、天空拡散照度（SH） | V3, V4（地形あり）, V8 |
| 5 | Shoga 移植設計のレビュー（§9 の境界の見直し） | 設計レビュー |

---

## 参考

- GT7 Tone Mapping: K. Yasutomi, K. Suzuki, H. Uchimura, "Driving Toward Reality: Physically Based Tone Mapping and Perceptual Fidelity in Gran Turismo 7", SIGGRAPH 2025 PBS Course. https://blog.selfshadow.com/publications/s2025-shading-course/
- GraphicsTest/LUTGenGT7（Python 移植・元 C++ と照合済み）、GraphicsTest/HDR/testbed（HLSL 実装・スワップチェーン・PQ）
- 大気: research.md の参考文献
