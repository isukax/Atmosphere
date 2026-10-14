# 大気散乱 リアルタイムレンダリング 調査レポート

- 作成日: 2026-10-15
- 目的: 将来 Shoga ランタイムに組み込むことを前提とした、大気散乱（空・空気遠近法）の検証実装に向けた事前調査
- 統合先の前提（Shoga 現状の確認結果）
  - D3D12 / HLSL、Bindless 前提の RHI、RenderGraph、Forward / Deferred の両パス、ForwardTranslucent パスあり
  - `RenderFeatureRegistry` による拡張 Feature 登録が可能（`ViewPipelineDesc::extensionFeatures`）
  - 空・IBL は未実装。環境光は定数 `ambientColor`（`LightingContext.hlsli`, `DeferredLightingShared.hlsli`）
  - `ExtractedDirectionalLight` は `directionWs / intensity / color` のみで、単位は未定義（デフォルト 1.0）
  - 露出は手動スカラー（`ViewPipelineExposureDesc`）+ `ExposureCompute` スロットあり
  - `ShadowDepth` パス種別は存在するが、ディレクショナルライト用シャドウマップの実装状況は未確認

---

## 1. 手法の系譜

| 系統 | 代表 | 概要 | 動的な時刻変化 | 宇宙視点 | 空気遠近法 | 多重散乱 | ゲーム用途での評価 |
|---|---|---|---|---|---|---|---|
| 解析モデル | Preetham 1999, Hosek-Wilkie 2012, Prague (Wilkie et al. 2021) | 実測/シミュレーション結果にフィットした式で天空輝度を返す | ○ | × | ×（別途必要） | フィットに内包 | 地上視点の空ドーム専用。高度変更・大気パラメータ変更不可。IBL ベイク用途向き |
| 実時間レイマーチ | Nishita 1993, O'Neil 2005 (GPU Gems 2) | ピクセルごとに視線・光線を数値積分 | ○ | ○ | ○ | ×（単散乱のみ） | 安価だが多重散乱なしで夕景・地平線が暗く不自然 |
| 4D/3D 事前計算 LUT | Bruneton & Neyret 2008, Elek 2009, Yusov 2013, Bruneton 2017 (改良版実装) | 散乱を (r, μ, μs, ν) の 4D テーブルに次数ごとに反復事前計算 | △（再計算は重い／時分割） | ○ | ○ | ○（反復回数分） | 品質は高いが大気パラメータ変更時の再計算コストが大きい（論文中、Bruneton 2017 実装で全 LUT 更新 250ms@GTX1080）。地平線付近の LUT アーティファクト、小惑星で精度低下 |
| 低次元 LUT + 多重散乱近似 | **Hillaire 2020 (EGSR, UE4/5 SkyAtmosphere)** | 透過率 2D・多重散乱 2D・Sky-View 2D・空気遠近法 3D の小さな LUT を毎フレーム生成 | ◎（毎フレーム更新可能） | ○（画面空間レイマーチに切替） | ○（フラスタム 3D LUT） | ○（無限次数を等方近似） | 現在の事実上の標準。UE4/5、Bevy 0.16+ などが採用 |
| パストレース | Hillaire 2020 中のリファレンス, 各種オフライン | デルタトラッキング等で厳密解 | － | ○ | ○ | ○（厳密） | 実時間用ではない。**検証用の正解生成に使う** |

補足:
- Frostbite (Hillaire 2016) は Bruneton 系 4D LUT を時分割更新し、空気遠近法をフラスタムボリュームで適用。2020 手法の前身。
- Ghost of Tsushima (Patry, SIGGRAPH 2021 Advances) は Bruneton/Elek/Yusov 系の 3D LUT を使い、毎フレーム太陽・月角度で 2D にリサンプリング。霞はフロクセル (128x64x64, 10cm–100km, 指数スライス) で別途計算。Rayleigh 誤差を最小化する独自 LMS 色空間を使用。
- Hillaire 2020 のフォローアップとして LUT パラメタ化や多重散乱近似そのものを改良した 2024–2026 年の論文は今回の調査では見つからなかった。周辺の発展は以下:
  - Schneegans et al. 2024: Mie 理論で成分ごとの位相関数・係数を事前計算（Bruneton モデル拡張、火星等）。Hillaire モデルへの統合は「今後の課題」と記載。
  - Schneegans et al. 2025 (CGF): 日食の物理ベース実時間レンダリング（屈折含む）。Bruneton モデルを拡張。Hillaire モデルは宇宙視点向けに最適化されていない点を指摘。
  - Bevy 0.16 (2025) で Hillaire 2020 を実装、その後 LUT モードと Raymarched モード（画面空間で直接積分、鋭いボリュームシャドウ向け）の切替、0.18 で大気によるシーンライティング（遮蔽・PBR）に拡張。

---

## 2. 推奨: Hillaire 2020 をベースにする

理由:
1. **全 LUT を毎フレーム再生成できるコスト**（GTX 1080 で LUT 生成合計 約 0.17ms）。時刻・天候・大気パラメータを実行時に自由に変えられる。ゲームランタイムでの要件（TOD、天候、アーティスト調整）に合う。
2. **多重散乱が非反復**。濃い大気（アート調整で密度を上げた場合）でも発散せず、エネルギー保存的。Bruneton は濃い大気で反復数が爆発し、FP32 でも数値的に破綻する例が論文で示されている。
3. **地上〜宇宙まで単一モデル**。宇宙視点は画面空間レイマーチへ切替。
4. **スケーラブル**（モバイル iPhone 6s で約 1ms）。LUT 解像度とステップ数で品質を調整できる。
5. **参照実装が MIT ライセンス**（sebh/UnrealEngineSkyAtmosphere, D3D11/HLSL）。同リポジトリに GPU パストレーサーと Bruneton 実装も含まれ、比較検証の土台がそのまま使える。
6. 業界採用実績（UE4.24+/UE5, Fortnite, Bevy）と、比較対象（UE5 の SkyAtmosphere 実装・パラメータ）が豊富。

Bruneton 2017 は「比較用リファレンス」として位置付ける（BSD-3、スペクトル→輝度変換の扱いが丁寧）。

---

## 3. Hillaire 2020 の詳細

### 3.1 大気モデル（地球、RGB）

| 成分 | 散乱係数 (1/km) | 吸収係数 (1/km) | 密度分布 | 位相関数 |
|---|---|---|---|---|
| Rayleigh | (5.802, 13.558, 33.1)e-3 | 0 | exp(-h / 8km) | 3/(16π)·(1+cos²θ) |
| Mie | 3.996e-3 | 0.444e-3（消散 4.44e-3、単散乱アルベド 0.9） | exp(-h / 1.2km) | Cornette-Shanks, g = 0.8（HG でも可） |
| オゾン | 0 | (0.650, 1.881, 0.085)e-3 | 25km 中心・幅 30km のテント: max(0, 1 - \|h-25\|/15) | － |

- 地表半径 6360km、大気上端 6460km（100km 層）
- 地表は拡散反射、論文本文はアルベド 0.3（参照実装の既定値は 0）
- 太陽視半径: 参照実装 0.004675 rad
- 注意: 論文 Table 1 では Mie 吸収が 4.40 と読めるが、参照実装は消散 4.44・散乱 3.996（= 吸収 0.444）。**実装値を正とする**
- オゾンは散乱せず吸収のみだが、夕暮れ時の天頂の青さに必須（Kutz 2013）

### 3.2 LUT 構成と論文記載の性能

| LUT | 次元/パラメタ | PC 解像度 (GTX1080) | ステップ | 時間 | モバイル (iPhone 6s) |
|---|---|---|---|---|---|
| Transmittance | 2D: 高度 × 視線天頂角（Bruneton 2017 のパラメタ化） | 256×64 | 40 | 0.01ms | 256×64 / 40 / 0.53ms |
| Multi-scattering | 2D: 高度 × 太陽天頂角 | 32×32 | 20（方向数 64） | 0.07ms | 32×32 / 20 / 0.12ms |
| Sky-View | 2D: 経度（太陽相対方位）× 緯度（非線形） | 200×100（参照実装 192×108） | 30 | 0.05ms | 96×50 / 8 / 0.27ms |
| Aerial Perspective | 3D: 画面 32×32 × 深度 32 スライス（0–32km） | 32³ | 30 | 0.04ms | 32²×16 / 8 / 0.11ms |

- 画面への合成（空 + 空気遠近法）0.14ms、合計 0.31ms @1280×720
- 宇宙視点は Sky-View LUT を使わず画面空間レイマーチ（合計 0.5ms）
- Fortnite の iPhone 6s で空全体 約 1ms
- ボリュームシャドウ有効時（CSM + 32 サンプル/ピクセルのレイマーチ + ジッタ + TAA）で 1.0ms

### 3.3 各 LUT の要点

**Transmittance LUT**: 太陽方向へのセカンダリレイマーチを置き換える。Bruneton と同じパラメタ化（地平線付近にテクセルを集中）。大気パラメータが変わった時のみ更新でよい。

**Multi-scattering LUT（本手法の核心）**:
- 2 次以上の散乱は等方的と仮定（位相関数を無視）。
- 各点で 64 方向にレイマーチし、2 次散乱輝度 L₂ と、周囲から等方的に戻ってくるエネルギーの割合 f_ms を積分。
- 無限次数の和を等比級数で近似: F_ms = 1 / (1 - f_ms)、Ψ_ms = L₂ · F_ms。
- 単散乱積分に `σs(x) · Ψ_ms` を加算するだけで多重散乱を反映（式 11: Lscat = σs (T·S·p + Ψ_ms) E）。
- 限界: Mie の g が大きいほど誤差増（RMSE: g=0 で 0.0058、g=0.8 で 0.039）。非常に濃い大気では色相がずれる。

**Sky-View LUT**:
- カメラ位置の局所 up を基準にした経緯度テクスチャ。地平線が常に水平線。
- 緯度を `v = 0.5 + 0.5·sign(l)·sqrt(|l| / (π/2))` で非線形化し地平線付近を高密度化。
- 太陽ディスクは低解像度・非線形マッピングのため LUT に含めず、後段で合成。
- 参照実装は経度を「太陽との相対方位」で持つため光源 1 つ前提。月など複数光源の散乱を入れるなら絶対方位化などの変更が必要。

**Aerial Perspective (AP) ボリューム**:
- カメラフラスタムに沿った 3D テクスチャ。RGB = 内散乱、A = 透過率（RGB 透過率の平均＝グレースケール）。
- 不透明物体へはライティング後のフルスクリーンパスで適用（Sky-View 合成と同時）。
- 半透明（フォワード）は論文では頂点単位。デスクトップならピクセル単位で 3D テクスチャをサンプルしても安価。
- 透過率がモノクロである点は近似。遠景の色付き減衰が必要なら RGB 透過率を別ボリュームで持つ（コスト 2 倍）か、Dual-source blending で RGB 透過率を合成する方法がある（UE には有色透過率オプションあり）。

### 3.4 論文で明示された限界と今後の課題
- 等方多重散乱近似による誤差（強い前方散乱、高密度大気）
- 空間的に変化する大気条件（局所的な霧・雲との統合）は対象外
- RGB レンダリング（スペクトルは future work）
- ボリュームシャドウ（山・雲の光芒）は LUT では表現不可 → レイマーチ + シャドウマップが必要

---

## 4. 実装時の落とし穴・設計上の論点

### 4.1 単位・露出・浮動小数点範囲（最重要）
- 物理単位では太陽照度は地表で約 10⁵ lux、太陽ディスク輝度は 10⁹ cd/m² オーダー。**FP16 の最大値 65504 を容易に超える**。
- 対策は二択:
  1. **物理単位 + プリエクスポージャ**（Frostbite / UE 方式）: シェーダ内で露出値を事前乗算してから FP16 RT に書く。自動露出（EV100）と一体で設計する必要がある。
  2. **正規化単位**（太陽照度 = 1 などとし、輝度スケールを露出で吸収）: 参照実装のデモはこちら。検証は楽だが、人工光源や発光体との物理的な比率が崩れる。
- Shoga は現状ライト強度の単位が未定義なので、**大気を入れる前にライト単位系（lux / cd / nit）と露出系の方針を決める必要がある**。これは大気以外（IBL、発光、ブルーム）にも波及する決定事項。

### 4.2 座標と精度
- 惑星半径 6360km を float32 で扱うと ULP ≈ 0.5m。km 単位で計算しても相対精度は同じ。
- 惑星中心原点で計算するのは LUT 生成内部のみに留め、レンダリング側は「カメラ高度」「惑星中心からのカメラ相対ベクトル」で渡す。
- 地平線付近のレイ-球交差は判別式の桁落ちが起きやすい。数値的に安定な交差式を使い、地表直下に潜らないよう小さなオフセット（参照実装は `PLANET_RADIUS_OFFSET = 0.01km`）を入れる。
- ワールド座標系（Shoga は Y-up と思われる。要確認）と惑星座標系の変換、ワールド単位（m）↔ 大気計算単位（km）のスケールを一箇所に集約する。
- 大規模ワールドや惑星規模を将来想定するなら、平面ワールドの「惑星上の原点位置」を指定するパラメータを用意する（UE の Transform Mode に相当）。

### 4.3 地平線・低太陽高度
- Transmittance LUT の地平線付近は境界でアーティファクトが出やすい（three-geospatial や Bruneton 系実装でも完全除去できていないと明記）。サブテクセル UV 補正（`fromUnitToSubUvs`）が必要。
- 日没前後は Sky-View LUT の解像度不足（地球の影、太陽周り）が出る。Sky-View LUT の解像度とステップ数、または画面空間レイマーチへのフォールバックを比較検証項目にする。
- 夜間（太陽が地平線下）は月光源の追加、星空、Purkinje シフトなどが別途必要（範囲外としてよい）。

### 4.4 太陽ディスク
- Sky-View LUT とは別に、視線方向と太陽方向の角度で解析的に描画し、透過率 LUT を掛ける。
- 周縁減光（limb darkening）の有無。
- ブルーム閾値との相性（物理単位なら極端に明るい）。

### 4.5 時間的安定性
- LUT 生成は低ステップでもほぼノイズなしだが、ボリュームシャドウのレイマーチを入れる場合はブルーノイズジッタ + TAA/再投影が前提。
- Shoga は MotionVectors パスがあるので、TAA または専用の時間フィルタと組み合わせ可能。

### 4.6 スペクトル vs RGB
- 標準は 3 波長相当の RGB。夕景の色相・彩度に誤差が出る。
- Bruneton 2017 は複数波長で計算して輝度/RGB へ変換する方式を持つ。Ghost of Tsushima は Rayleigh 透過率誤差を最小化する LMS 系の色空間を採用。
- 検証実装では RGB で開始し、スペクトルリファレンス（オフライン）との差を測ってから必要性を判断するのが妥当。

### 4.7 ワイドカラーガマット
- 係数は sRGB/Rec.709 原色前提の値。Shoga/HDR 出力がより広い作業色空間を使うなら係数の再導出が必要（GraphicsTest/WideColorGamutLighting との整合を確認）。

---

## 5. エンジン統合で必要な接点（Shoga 視点）

| 接点 | 内容 | Shoga の現状 / 必要な変更 |
|---|---|---|
| 大気パラメータ | 惑星半径、各成分係数・密度プロファイル、地表アルベド、太陽視半径 | Scene 側にコンポーネント（例: `SkyAtmosphereComponent`）と Extract 構造体を追加 |
| 太陽ライト | ディレクショナルライトを大気の光源として扱い、**透過率で減衰させた色をサーフェスライティングに使う**（カメラ位置で 1 回、またはピクセル単位） | `ExtractedDirectionalLight` に「大気光源」フラグ・照度単位を追加 |
| 空の描画 | Sky-View LUT + 太陽ディスクを背景として合成 | 背景（深度 = far）領域を描くパスが必要。Deferred は Lighting 後、Forward は Opaque 後 |
| 空気遠近法（不透明） | AP ボリュームを深度から引いて合成 | ライティング後・トーンマップ前のフルスクリーンパス |
| 空気遠近法（半透明） | ForwardTranslucent のピクセルシェーダで AP ボリュームを参照 | Bindless なので AP ボリュームのインデックスを view 定数に入れるだけで済む |
| 環境光（スカイライト） | 空からの拡散照度。定数 ambient の置換 | 簡易: 多重散乱 LUT / Sky-View LUT から SH or 照度を算出。本格: 空キューブマップのリアルタイムキャプチャ＋プリフィルタ（GraphicsTest/IBL の成果と接続） |
| LUT の寿命 | Transmittance / Multi-scattering はパラメータ変更時のみ、Sky-View / AP はビューごと毎フレーム | RenderGraph の永続リソース（外部インポート）と、ビューごと一時リソースの区別。エディタ+ゲームの複数ビュー対応 |
| ボリュームシャドウ（光芒） | ディレクショナルシャドウマップ参照のレイマーチ | シャドウマップ実装状況に依存。後段フェーズ |
| ハイトフォグ / ボリュメトリックフォグ / 雲 | 大気透過率 LUT・AP と同じ照明モデルを共有 | 将来拡張。フロクセル基盤を作るなら AP ボリュームと統合可能 |
| 露出 | 物理単位ならプリエクスポージャと自動露出 | 4.1 の決定次第 |

RenderGraph 上のパス順（Deferred の場合の案）:
```
[Compute] TransmittanceLUT        (パラメータ変更時)
[Compute] MultiScatteringLUT      (パラメータ変更時)
[Compute] SkyViewLUT              (ビュー毎)
[Compute] AerialPerspectiveVolume (ビュー毎)
DepthPrepass → GBuffer → DeferredLighting
[Graphics] Sky + Sun disk + AerialPerspective 合成   (SceneColor へ)
ForwardTranslucent (AP をピクセルで適用)
Exposure → ToneMapping
```
LUT 生成 4 パスは依存関係が浅く、非同期コンピュートに載せやすい。

---

## 6. 他エンジン・実装の比較メモ

| 実装 | 方式 | 特徴 / 参考点 |
|---|---|---|
| UE4.24+/UE5 SkyAtmosphere | Hillaire 2020 | 原典。AP 開始距離・距離スケール、ハイトフォグへのライティング、半透明への AP（モバイルは頂点単位）、Fast Sky / AP の CVar で品質切替 |
| sebh/UnrealEngineSkyAtmosphere (MIT) | Hillaire 2020 + Bruneton + パストレーサー | D3D11/HLSL。検証・比較用のベース。最終更新 2022 |
| ebruneton/precomputed_atmospheric_scattering (BSD-3) | Bruneton 2017 | スペクトル→輝度変換、厳密な LUT パラメタ化、ドキュメント豊富。2025 年にも更新あり |
| Bevy (0.16+) | Hillaire 2020 | LUT / Raymarched モード切替、シーン単位↔メートル変換係数、0.18 で大気によるシーンライティング。比較的新しい実装として設計の参考になる |
| takram three-geospatial (MIT) | Bruneton 2017 | Web 向け。ECEF 対応・Stars・AerialPerspective 合成。光芒は未実装、地平線アーティファクトは完全には消せないと明記 |
| Ghost of Tsushima | Bruneton 系 3D LUT + フロクセル霞 | 3D→2D 毎フレームリサンプル、雲/霞/パーティクルの照明統合、Purkinje、照度ベース露出 |
| Frostbite (2016) | Bruneton 系 4D LUT 時分割 | 物理単位・露出系との統合、雲との合成 |

未確認事項（推測で書かない）: Unity HDRP Physically Based Sky の現行実装詳細、UE5 の現行 LUT 既定解像度。必要なら追加調査する。

---

## 7. 検証方法の提案

1. **リファレンス生成**: GPU パストレーサー（デルタトラッキング + 比率トラッキング、散乱次数 N 指定）を検証アプリ内に実装、または参照実装のものを移植。Python/NumPy による単散乱・透過率のオフライン積分も CPU ゴールデンとして用意（GraphicsTest/HDR の `pyref` / `golden` と同じ運用）。
2. **比較シナリオ**（論文と同条件 + 追加）
   - 昼（太陽高度 30–60°）、夕暮れ（0–5°）、太陽が地平線下（地球の影）、上空 10–80km、宇宙視点
   - 濃い大気（Rayleigh/Mie ×5〜×50）、Mie g = 0 / 0.8 / 0.95
   - 火星風・小惑星（パラメタ化の頑健性）
3. **指標**: RGB 別 RMSE（論文値 昼 ~1e-3 オーダーと比較）、相対誤差マップ、地平線付近の 1D プロファイル、LUT 解像度・ステップ数を振ったときの誤差-コスト曲線
4. **性能**: パス別 GPU 時間（PIX / Tracy のマーカー）、解像度別、Sky-View LUT vs 画面空間レイマーチ
5. **安定性**: 太陽高度を連続変化させた際のちらつき・バンディング、カメラ高度遷移（地上→宇宙）のポッピング

---

## 8. 段階的な検証実装計画（案）

| フェーズ | 内容 | 完了条件 |
|---|---|---|
| P0 | 単位系・露出方針、座標系、検証アプリの土台（D3D12/HLSL、Shoga RHI を使うか単独か）を決定 | 決定事項の文書化 |
| P1 | Transmittance + 単散乱の画面空間レイマーチ（リファレンス兼用）、パストレーサー | パストレーサーとの単散乱一致 |
| P2 | Multi-scattering LUT、Sky-View LUT、太陽ディスク | 論文 Fig.9 相当シナリオで RMSE が論文と同オーダー |
| P3 | Aerial Perspective ボリューム、不透明/半透明への適用、太陽光の透過率減衰をサーフェスライティングに反映 | 地形・オブジェクトありのシーンで破綻なし、性能計測 |
| P4 | スカイライト（環境光）生成、IBL 連携 | 定数 ambient を置換 |
| P5（任意） | ボリュームシャドウ（光芒）、宇宙視点、月/夜、スペクトル比較 | － |
| P6 | Shoga への RenderFeature としての統合設計 | 設計レビュー |

---

## 9. 決める必要がある事項（次ステップ前に確認したい）

1. **ライトの単位系と露出**: 物理単位（lux, EV100, プリエクスポージャ）で行くか、正規化単位で行くか。
2. **検証アプリの土台**: Shoga の RHI/RenderGraph の上に作るか、GraphicsTest の他プロジェクトと同様に単独の D3D12 アプリにするか。
3. **対象スコープ**: 地上〜高高度のみか、宇宙視点・惑星表現まで必要か。夜空・月は必要か。
4. **プラットフォーム**: Shoga はクロスプラットフォーム前提だが、モバイルクラスの GPU まで品質スケーリングを検証するか。
5. **雲・フォグとの統合**を将来計画に含めるか（AP ボリュームをフロクセル基盤として設計するかに影響）。
6. **外部コード利用**: 参照実装（MIT / BSD-3）からの移植を許容するか、論文から独自実装するか。

---

## 参考文献・リソース

- S. Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", EGSR 2020 / CGF 39(4). https://sebh.github.io/publications/egsr2020.pdf
- 参照実装 (MIT): https://github.com/sebh/UnrealEngineSkyAtmosphere
- E. Bruneton, F. Neyret, "Precomputed Atmospheric Scattering", EGSR 2008. https://diglib.eg.org/handle/10.2312/387
- E. Bruneton, 改良版実装 (BSD-3): https://github.com/ebruneton/precomputed_atmospheric_scattering
- E. Bruneton, "A Qualitative and Quantitative Evaluation of 8 Clear Sky Models", TVCG 2017
- S. Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016 PBS Course. https://www.ea.com/news/physically-based-sky-atmosphere-and-cloud-rendering
- S. Hillaire, SIGGRAPH 2020 PBS Course slides. https://blog.selfshadow.com/publications/s2020-shading-course/hillaire/s2020_pbs_hillaire_slides.pdf
- J. Patry, "Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping in Ghost of Tsushima", SIGGRAPH 2021 Advances. https://www.advances.realtimerendering.com/s2021/jpatry_advances2021/index.html
- Wilkie et al., "A Fitted Radiance and Attenuation Model for Realistic Atmospheres" (Prague sky model), SIGGRAPH 2021. https://cgg.mff.cuni.cz/publications/skymodel-2021/
- Hosek, Wilkie, "An Analytic Model for Full Spectral Sky-Dome Radiance", SIGGRAPH 2012. https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint.pdf
- Schneegans et al., "Physically Based Real-Time Rendering of Atmospheres using Mie Theory", 2024. https://www.researchgate.net/publication/380239530
- Schneegans et al., "Physically Based Real-Time Rendering of Eclipses", CGF 2025. https://onlinelibrary.wiley.com/doi/10.1111/cgf.70017
- Bevy atmosphere example: https://bevy.org/examples/3d-rendering/atmosphere/ / Bevy 0.18 release notes: https://bevy.org/news/bevy-0-18/
- takram three-geospatial (MIT): https://github.com/takram-design-engineering/three-geospatial
- UE SkyAtmosphere reference: https://dev.epicgames.com/documentation/unreal-engine/sky-atmosphere-component-properties-in-unreal-engine
- M. Sakmary, "Real-time Rendering of Atmosphere and Clouds in Vulkan", CESCG 2023. https://cescg.org/wp-content/uploads/2023/04/Sakmary-Real-time-Rendering-of-Atmosphere-and-Clouds-in-Vulkan.pdf
