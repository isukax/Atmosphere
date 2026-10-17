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

## 10. 追加調査: 空気遠近法のボリュームと影（光芒）のブロック状の見た目（2026-10-16）

背景: Phase 4.5 で空気遠近法のボリューム（32×32×32）の単散乱に太陽のシャドウマップを掛けた。カメラを地表近くに置き、太陽が低いと、球や山の影がボリュームの格子（画角 60° で 1 ボクセル ~1.9°）の大きさのブロックとして見える。他の実装がこれをどう扱っているかを調べた。

### 10.1 何が起きているか（本実装）
- Hillaire 2020 の LUT 方式は「空気遠近法は画面上でも奥行きでも滑らか」という前提に立つ。講演スライドも、高周波を作るのは地平線とボリュームの影だけだと述べている（*"Only the horizon and volumetric shadows create high frequencies"*）。影を入れると、この前提が崩れる。
- 本実装では次の 3 つが重なっている。
  1. ボリュームの画面方向の解像度（32×32）。
  2. ボリュームが引くのは全体のカスケードだけ（1 テクセル 約 10 m）。近くの半径 2 m の球の影は、このテクセルより細い。
  3. 1 回の比較で、低域通過フィルターをかけていない。

### 10.2 他の実装の対処
| 実装 | 方式 | 影の扱い | 出典 |
|---|---|---|---|
| UE Sky Atmosphere | 空気遠近法を froxel の LUT に入れる（Hillaire 2020 そのもの） | ディレクショナルライトの「Cast Shadow on Atmosphere」と `r.SkyAtmosphere.SampleLightShadowmap` で、大気のボリュームの影を作る。LUT の解像度は `r.SkyAtmosphere.AerialPerspectiveLUT.Width` / `.DepthResolution` で上げられる（3D テクスチャなのでメモリが増える）。映像向けには `r.SkyAtmosphere.FastSkyLUT=0` と `r.SkyAtmosphere.AerialPerspectiveLUT.FastApplyOnOpaque=0` で LUT の最適化を切り、画面のレイマーチにする（コスト増。高周波の模様が TAA で消えないことがある、とドキュメントに注記）。空気遠近法の開始距離 `aerial_perspective_start_depth` は既定 0.1 km（目的は性能） | UE ドキュメント（Sky Atmosphere、Python API） |
| UE Volumetric Fog | カメラの視錐台に沿った低解像度の froxel（高さフォグ・ボリュメトリックフォグ用） | ディレクショナルライト 1 灯を CSM（または静的な影）で遮る。froxel の粗さは、フレームごとに違うボクセル内のジッターと、強い時間方向の再投影で隠す。代償は動く光の残像 | UE ドキュメント（Volumetric Fog） |
| Assassin's Creed IV（Wronski 2014） | froxel 160×90×64（または 128） | シャドウマップ（4 カスケード 1k×1k）をそのまま引くと、細かさがボリュームのナイキスト周波数を超えて、エイリアシングとちらつきが出た。指数シャドウマップ（ESM）にして 256² に縮小し、ぼかしてから引く（低域通過フィルター）。光漏れは媒質の中では目立たない | AC4 GDC 2014 スライド |
| Frostbite（Hillaire 2015） | 段階的な（カスケード）消散のボリューム | 任意の光源から影を落とせる、単純なボリュームのシャドウマップ | SIGGRAPH 2015 Advances の要旨 |
| Bevy | LUT モード（3D の内散乱 LUT）と、レイマーチのモード | LUT モードは影の解像度が LUT の画面方向の解像度に制限されると、設計文書が明記。鋭いボリュームの影が要るときは、画面のレイマーチのモード（Sky-View LUT・空気遠近法の LUT を使わない）。設計文書（未実装の提案を含む）には、ジッター・TAA、時間方向の再投影の案もある | Bevy の AtmosphereMode のドキュメント、設計文書（HackMD） |
| Bruneton 2017 | LUT（4D の散乱） | ピクセルごとに、視線上で影に入る長さ `shadow_length` を求め、その区間の散乱を差し引く（GetSkyRadianceToPoint）。デモは 1 個の球の影の円錐との交差を解析的に求めるだけ | Bruneton の参照実装のドキュメント |
| Intel（Yusov 2013） | エピポーラ線上のサンプリング + 1D min/max 二分木 | 影の境界に合わせて画面のサンプルを置き、シャドウマップの行ごとの min/max 木で視線の可視区間を速く求める（Chen et al. 2011 の 1D min-max mipmap の系譜）。鋭い光芒を高い解像度で出せるが、実装は複雑 | GDC 2013、GPU Pro 5 |
| Unity HDRP | Physically Based Sky + ボリュメトリックフォグ | 光芒はボリュメトリックフォグ（froxel）側で、光源ごとの「Shadow Dimmer」で影の強さを調整 | Unity のブログ、チュートリアル |
| The Last of Us Part II（Kovalovs 2020） | ビュー空間の froxel | 要旨は「グリッドのフォグのアーティファクトを減らす」工夫を述べる。詳細は未確認 | SIGGRAPH 2020 Talks の要旨 |

### 10.3 まとめ（対処の型）
1. **影を froxel の解像度まで低域通過フィルターする**（AC4: ESM の縮小 + ぼかし）。ブロックとちらつきは消えるが、細い影は出ない（光芒がやわらかくなる）。安い。
2. **ボリュームの解像度を上げる**（UE の cvar、AC4 の 160×90×64）。コストとメモリはボクセル数に比例する。32³ → 160×90×64 は 28 倍。
3. **ジッター + 時間方向の再投影**（UE Volumetric Fog、Bevy の提案）。TAA と履歴バッファが前提で、残像との引き換え。
4. **不透明物にはピクセルごとにレイマーチする**（UE の FastApplyOnOpaque=0、Bevy のレイマーチのモード）。正確で鋭いが、コストは画素数に比例する。ステップ数を減らすとノイズが出て、TAA に頼る。
5. **ピクセルごとに影に入る区間だけを求めて差し引く**（Bruneton の shadow_length、Yusov / Chen の min/max 木はその高速化）。散乱そのものは LUT のまま、可視性だけを画面の解像度で求める。

主要な実装（UE）は、1（シャドウマップの解像度に依存）と 2 の調整、近景の 3（ボリュメトリックフォグ）、映像向けの 4 を併用していて、LUT 版で鋭い光芒を出すことはあきらめている。

### 10.4 本実装への当てはめ（案、未決定）
- **すぐできる**: 空気遠近法で引く影に、ボクセルの大きさに合わせた低域通過フィルターをかける（1。PCF の広さをスライスの奥行きと画角から決めるか、ESM / VSM のような、ぼかせる形式で持つ）。カメラの近くは近景のカスケードを使う。
- **鋭さを保つ**: 5 の型。空気遠近法のボリュームは影なしの単散乱（視線上の累積値）を持ったまま、ピクセルごとに視線上の点（例: 16〜32 点）でシャドウマップだけを引く。影に入る区間の単散乱を、累積値の差（ボリュームを奥行きで引いた差）で差し引く。大気の計算はボリュームのままで、ピクセルごとに増えるのはシャドウマップを引く回数だけ。そのためには、ボリュームで単散乱を多重散乱から分けて持つ必要がある（Mie は Phase 3 で分離済み。Rayleigh は多重散乱と混ざっている）。
- 2 / 3 / 4 は、それぞれ Shoga 側の TAA・性能の予算と関わるので、移植の設計（Phase 5）で扱う。

## 11. 追加調査: 『グランツーリスモ７』の天球レンダリング（2026-10-17）

目的: Polyphony Digital の講演から、本実装（Hillaire 2020 ベース、Phase 4.5 まで実装済み）に取り入れられる点を探す。

### 11.1 資料

| 講演 | 内容 | 出典 |
|---|---|---|
| CEDEC+KYUSHU 2022「『グランツーリスモ７』における天球レンダリング」（鈴木 健太郎、安冨 健一郎） | 日本語、173 枚。GDC 版とほぼ同じ構成。オゾン・地表アルベド・検証の記述が少し多い | Polyphony Digital の発表資料ページの PDF |
| GDC 2023「Realistic Real-time Sky Dome Rendering in Gran Turismo 7」（同） | 英語、268 枚（発表者ノート付き）。Sky-View LUT のマッピングの擬似コード、フォグ・雲の詳細 | 同上 |
| SIGGRAPH 2023「Rendering Technologies behind Gran Turismo 7: VR, Ray Tracing and Sky Simulation」（鈴木） | 空の部分は上の 2 つの要約。新しい情報はない | 同上 |

以下はスライドと発表者ノートの記述による。スライドにない内容は「推測」と書く。

### 11.2 GT7 の方式

**全体の構成**: 正確な大気モデルを分光パストレーサーでオフラインで解き、結果を時刻 × 天候の LUT に焼いてランタイムで引く。ランタイムで散乱は解かない。

**オフライン（Skysim）**

| 項目 | 内容 |
|---|---|
| レンダラー | 大気専用のボリュームパストレーサー（C++ 約 2 万行、開発 1 年程度）。**分光**（波長ごとに計算し、CIE 2006 2° の等色関数で XYZ → Rec.2020）。Ratio tracking（Novák 2014）、位相関数と波長（分光視感効率）の重点サンプリング、太陽近傍の適応的なサンプル増加、デノイザ、社内の空き PC での分散レンダリング |
| 空気 | Rayleigh の位相関数（近似で誤差なし）。鉛直分布は U.S. Standard 1976。NRLMSIS と比べ、差の最も大きい条件で 3 割ほど違ったが、見た目は地平線をよく見比べて分かる程度だった。指数で近似すると H = 7.994 km（線形のプロットでは少しずれる） |
| エアロゾル | 層の和。各層は「複素屈折率（波長依存）+ 対数正規の粒径分布（平均半径・幾何標準偏差）+ 鉛直密度の表」。散乱係数・吸収係数・位相関数は **BHMIE（Bohren & Huffman の Mie 計算）で粒子の物性から導出**する（直接与えない）。屈折率は RefractiveIndex.INFO・ARIA・OPAC、粒径分布は AERONET と Omar 2005 |
| オゾン | Chappuis 帯（可視光の吸収）がブルーアワーの深い青に効く、と強調。参考文献は Krueger & Minzner 1976（中緯度のオゾン分布）と Bremen 大学の O3 吸収断面積 |
| 太陽 | ASTM G173-03 の大気圏外スペクトルを 10 nm で平均し、太陽の立体角 6.807×10⁻⁵ sr で割って放射輝度にする。散乱計算では平行光 |
| 地表 | アルベドも分光（森・砂漠・海・雪の測定値）。照り返しは「意外と重要」 |

**調査で分かったと述べていること**:
1. エアロゾルの粒径分布は二峰性（fine-mode と coarse-mode）。Le Mans の写真で、太陽の周りの広く弱い散乱は fine-mode、太陽のすぐ近くの鋭い散乱は coarse-mode と読み取れる。
2. Henyey-Greenstein では、どちらのモードの位相関数にも合わない。coarse-mode の前方では 16 倍以上ずれる（対数のプロットで）。
3. **「Mie 散乱は波長に依存しない」は正しくない。** fine-mode の粒径パラメータは Rayleigh と Mie の中間の領域にあり、散乱はかなり青い（夕焼けを赤くする側に働く）。位相関数は Mie のまま。アーティストが「フォグに青が、夕焼けに赤が足りない」と言う理由の 1 つ、と推測している。
4. エアロゾルの鉛直分布は指数関数からかけ離れていて、粒径によっても違う（Sasano 1996、Brattich 2019）。
5. 自己評価: Rayleigh は正確。Mie（エアロゾル）は問題が多い。**どの精度の改善がどれだけ効いたかは詳しく検証していない。** 計算コストは極めて大きい。
6. 副次的な効果: coarse-mode による太陽周りの散乱で、車体に映る太陽が自然に大きく柔らかくなった（GGX の粗さやポストエフェクトでぼかすより良い）。

**ランタイム**

| 項目 | 内容 |
|---|---|
| 空の LUT | 天候 1 状態につき 3D テクスチャ 1 枚: 時刻（太陽高度）64 スライス × 方向 32×128（晴れ）/ 16×64（曇り）。R11G11B10F（BC6H は色の精度が足りない）。8 枚で 3.5 MiB。**高度は LUT のパラメータから外す**（GT のコースの高度差は 300 m 程度） |
| 時刻方向のスライス | 太陽が地平線の上に 48 枚、下に 16 枚（−10° まで、それ以下は一定値へ補間）。上側は「大気上端までの距離」を真上〜水平で線形に刻んで太陽天頂角に戻す（地平線付近に集まる）。下側は天頂角 90〜100° を 2 次で刻む |
| 方向のマッピング | 太陽を極にした極座標。太陽から地平線までの角度で緯度を正規化して、**地平線が LUT 上で直線**になるようにする（隣のスライスと補間でき、任意の太陽高度に戻せる）。さらに v を非線形に変換して、太陽の周りと地平線の上に texel を集める。太陽が数度より低いときは通常の経緯度マッピングに切り替える（極座標が破綻するため） |
| 天候 | 「基底空」8 枚の線形補間。晴れは clear / hazed の 2 枚を haziness で、曇りは「高空の巨大な雲」をエアロゾル層として大気モデルに入れた 6 枚を cloudiness で。地上から見る限り、違うエアロゾル条件でも似た空になるので、晴れは 1 軸で十分と判断（実験による） |
| ほかの LUT | フォグ 2 種、透過率、太陽光の強度、空の平均の環境光 |
| グローバルフォグ（空気遠近法） | 空と同じ大気モデル（無限遠で空と一致）。内散乱 LUT（方向 32×32 × 時刻 64）と、**減衰の LUT を別に** 64×64（u = (距離/大気上端までの距離)^(1/3)、v = 視線角^(1/2)）。深度方向は LUT に持たず、解析式 LUT(dir)·(1 − e^(−a·θ^p·x)) / (1 − e^(−a·θ^p))（a, p はリファレンスにフィット）で近似。空の LUT を使い回すと近景の誤差が大きかったので、雲より下の低高度の大気だけで計算した別の LUT にした |
| 雲の影の光芒 | 太陽の単散乱を除いたフォグ LUT（Shadowed Fog LUT）を用意し、ワールドを覆う 3D 格子で求めた雲の遮蔽で通常のフォグ LUT と補間する |
| 雲 | 積雲のみ、Schneider 2015/2017 系のノイズ + レイマーチ。原点中心の上半球の「Cloud G-buffer」（PS5 で 1024²、RGBA16F: 背景の透過率・雲までの距離・太陽の内散乱・環境光）に書き、カメラに依存しないので本画面・バックミラー・キューブマップで共用。散乱の計算は非同期コンピュートで 32〜64 フレームに分割（G-buffer 1 枚に約 6 秒）、照明は毎フレーム遅延シェーディング。多重散乱は Wrenninge 2013 の近似。遠くの雲の影は太陽方向の透過率ボリュームで。増分は PS5 で +0.18 ms |
| 検証 | 社内の全周 HDR 写真（Scapes、数千枚）と見比べ、テクニカルアーティストが基底空の大気状態と補間の重みを合わせた（「物理ベースのルックデブ」） |

### 11.3 本実装との違い

| 項目 | 本実装 | GT7 |
|---|---|---|
| 目的 | 大気パラメータ・カメラ高度を実行時に変えられる（D3） | 地上 300 m 以内で、写真に合う空（焼き込み） |
| 解き方 | RGB 3 チャンネル、毎フレームの LUT。多重散乱は等方近似 | 分光パストレース（オフライン） |
| Mie の位相関数 | Cornette-Shanks、g = 0.8 の 1 つ | 粒径分布から Mie 理論で計算、モードごと |
| Mie の波長依存 | なし（灰色、§2.6 の全セット共通） | あり（fine-mode は青い） |
| 鉛直分布 | Rayleigh・Mie は指数、オゾンはテント | 表（U.S. Standard 1976、測定値） |
| 地表 | 灰色のアルベド | 分光のアルベド |
| 空気遠近法の透過率 | RGB の平均（論文・参照実装と同じ） | 減衰を別の LUT に持つ（RGB） |
| 検証 | 同じ大気モデルを別の解法で解いた結果との比較（LUT / レイマーチ / パストレース / Python） | 実写の HDR 写真 |

**最大の違い**: 本実装の検証（V1〜V9）は、すべて「解法の誤差」を測っている。大気モデルそのものが現実の空とどれだけ違うかは測っていない。GT7 の結論は逆で、見た目を決めるのは解法より大気モデル（特にエアロゾル）の誤差だった。

### 11.4 改善案

| # | 案 | 効く所 | コスト | 本実装の準備状況 |
|---|---|---|---|---|
| G | 空気遠近法の透過率を RGB にする | V3c の最大の誤差（遠景の色） | 小 | 合成はコンピュートで RGB を掛けられる |
| A | Mie の位相関数を表にする（Mie 理論 + 粒径分布） | 太陽の周りの光輪・霞の見た目 | 小〜中 | **位相関数はすでに LUT の外で掛けている**（§12 Q9） |
| B | Mie の波長依存 | 遠景の青み、夕焼けの赤 | 小（データのみ） | 係数はすでに float3 |
| E | 分光パストレース | 上の効果を測るための正解 | 中 | GPU パストレーサーと分光データあり |
| D | 鉛直分布を表にする | エアロゾルの層構造 | 中（毎ステップのテクスチャ読み） | なし |
| C | エアロゾルのプリセットと haziness の 1 軸 | Shoga のアーティスト向けの操作 | 小（A・B・D の上に） | なし |
| F | 実測の空との比較（V10） | 大気モデルの妥当性 | 中（データの入手と整理） | なし |
| H | 多重散乱の近似の改善（実験） | 夕方の R の不足（V4） | 小〜中 | V4 の仕組みあり |
| I | 空のピクセルの光芒 | 稜線の上の空が明るすぎる問題（§5.5） | 中 | 方式 5 を広げる。Q14 で範囲外にしたので要判断 |

**G: 空気遠近法の透過率を RGB にする**
- 設計と検証の計画は design.md §5.7（2026-10-17）。結果: Transmittance LUT の比は地平線付近で誤差が大きく不採用。ボリュームに RGB の透過率を持つ方式を既定にした（design.md §14 Phase 4.6）。
- V3c で、誤差の大部分は透過率を RGB 平均にする近似だった（R/B で ±6%、5〜20 km。相対 RMSE 2.5〜4.6%）。GT7 は減衰を内散乱と別の LUT に持っている。
- 本実装ではボリュームを増やさずにできる: 2 点間の透過率は Transmittance LUT の比 T(カメラ → 点) = T(カメラ → 上端) / T(点 → 上端) で求まる（Bruneton 2017 の `GetTransmittance`。地表に向かう光線は向きを逆にして同じ比を取る）。ピクセルごとにテクスチャ 2 回。
- 合成（`SkyFromLut.hlsl` の `ApplyAerialPerspective`）はコンピュートでシーンカラーを読み書きしているので、RGB の透過率をそのまま掛けられる。
- 注意: 地平線付近では比の分母が小さくなり、Bruneton の実装でも誤差が出ると知られている。スライス 0 の手前のフェードとの整合も要る。V3c で確かめる。

**A: Mie の位相関数を表にする**
- GT7 が最も重視した点。本実装では、Phase 3 の対策（g = 0.99 の光輪の崩れ）で Mie の位相関数を LUT から外し、Sky-View・空気遠近法・空からの照度（SH の太陽中心の求積）のすべてで、描画時に正確な角度で `CornetteShanksPhase` を掛けている。この関数を 1D テクスチャ（散乱角 → p、RGB）に替えれば、3 か所すべてに入る。
- 1D テクスチャは θ = π·u² で刻む（`SkyIrradiance.hlsl` の Mie の格子と同じ考え方。前方のピークを細かく）。∫p dω = 1 を Python で確かめる。
- Multi-scattering LUT は等方なので変わらない。パストレーサーは表の累積分布から方向をサンプリングする（V4 用）。
- モードが複数あり鉛直分布が違うと、視線上の位置で位相関数の混ざり方が変わるので、位相関数を 1 つにまとめて外に出せない。モードごとに Mie の LUT を分ける（2 モードなら Sky-View Mie LUT と空気遠近法の Mie ボリュームが 2 枚ずつ）。同じ分布なら 1 枚でよい。
- coarse-mode のピークは g = 0.99（半値全幅 0.88°）より鋭い可能性がある。描画は正確な角度で掛けるので LUT の解像度の問題は出ないが、太陽ディスク（視半径 0.27°）と同程度の幅になると、ディスクの描画・ブルームとの関係を考える必要がある（推測）。V3b を新しい位相関数で測り直す。
- Mie の計算: BHMIE を numpy に移植するか、外部ライブラリ（miepython、PyMieScatt など）を使う。外部ライブラリは確認が要る（§12 Q1 の運用）。

**B: Mie の波長依存**
- `mieScatteringPerKm` / `mieExtinctionPerKm` はすでに float3 なので、シェーダーは変わらない。`derive_coefficients.py` で、Mie 理論の σs(λ)・σa(λ) を作業色空間ごとにフィットする（Rayleigh・オゾンと同じ方法、§2.6）。
- design.md §2.6 の「Mie は波長に依存しない（灰色）ので全セットで共通値」を改める。`Hillaire2020` のセットは論文との比較用にそのまま残す。
- V9 の分光リファレンスにも Mie の波長依存を入れて、夕方・地平線での差を測る。

**E: 分光パストレース**
- V9 が分光で比べているのは透過率 + 単散乱だけ。GT7 は「散乱の多い計算は RGB だと誤差が大きい」としているが、本実装の多重散乱の色の誤差は測っていない。
- GPU パストレーサーに波長のサンプリング（光線ごとに波長を選び、分光視感効率または等色関数で重点サンプリング）を加え、XYZ → 作業色空間で累積する。
- LUT 描画との差を「3 チャンネル化」「多重散乱の近似」「大気モデル（A・B・D）」に分けて測れる。GT7 自身が検証していないと述べた「どの精度がどれだけ効くか」を、本実装の検証の仕組みで測れる。

**D: 鉛直分布を表にする**
- `SampleMedium` の指数とテントを、高度 → 密度の 1D テクスチャ（例: RGBA = 空気、エアロゾル fine、coarse、オゾン）に替える。テクスチャを使わない形として、Bruneton 2017 の `DensityProfileLayer`（指数 + 線形の 2 層）もある。
- 空気は U.S. Standard 1976、オゾンは Krueger & Minzner 1976。GT7 の結論では、空気の分布の違いは見た目にほとんど効かない。効くのはエアロゾルの層構造（境界層の上端で急に薄くなる、など）と推測する（未検証）。
- コスト: 積分の各ステップでテクスチャ 1 回。毎フレームのパス（Sky-View LUT、空気遠近法のボリューム）に効くので V8 で測る。パストレーサーのデルタトラッキングの上限 σ̄ は表の最大値にする。

**C: エアロゾルのプリセットと haziness**
- Omar 2005（AERONET のクラスタ分析で、砂塵・バイオマス燃焼・汚染された海洋などのタイプに分けたもの。タイプの一覧と値は論文で要確認）の、2 モードの粒径分布と屈折率からプリセットを作る。
- Shoga のアーティスト向けに、GT7 の clear ↔ hazed に当たる haziness の 1 軸を用意する。GT7 は焼いた輝度を補間するが、本実装は大気パラメータ（係数・位相関数の表・密度）を補間して毎フレーム解き直すので、物理的に一貫したまま操作できる。
- A・B・D ができてからの話。

**F: 実測の空との比較（新しい検証項目 V10）**
- GT7 が引用している Kider et al. 2014 は、空の HDR 画像と分光放射輝度計の測定を公開している（データの範囲・ライセンスは未確認）。日時と地点が分かるので、同じ日の AERONET のデータと組み合わせられる可能性がある（未確認）。
- 合否ではなく記録。大気モデル（プリセット）が現実の空にどれだけ近いかを測る、本実装で唯一の手段になる。

**H: 多重散乱の近似の改善（実験）**
- V4 で、近似は R の多重散乱を昼 −18%、夕方 −37% 少なく見積もった。GT7 は夕焼けの輝きの大部分を長い光路の多重散乱によるとし、ここを重視している。
- 案 1: Multi-scattering LUT の 2 次散乱 L₂ で、太陽光の 1 回目の散乱に実際の位相関数を使い、2 回目（LUT の点での散乱）だけを等方にする。今は `isotropicPhase = true` で両方を等方にしている。f_ms はそのまま。64 方向では Mie のピーク（g = 0.8 で幅 19°）を粗く拾うので、方向数も増やす。効果は分からないので、V4 の「多重散乱成分の比」で判定する。
- 案 2: パストレースとの比（補正係数）を（高度, 太陽天頂角）の LUT にして、プリセットごとにオフラインで焼く（GT7 の「オフラインの正解を LUT に」の考え方）。大気パラメータを変えると使えないので、Shoga で大気をプリセットに限る場合だけの案。

**I: 空のピクセルの光芒**
- GT7 は「太陽の単散乱を除いたフォグ」と遮蔽の補間で雲の光芒を出している。本実装の方式 5（影に入る区間の単散乱を、ボリュームの累積値の差で差し引く、§5.6）と同じ考え方。
- 方式 5 は、視線の終点をボリュームの最遠（128 km）に置けば、空のピクセルにもそのまま使える（影の判定はシャドウマップの範囲内）。稜線の上の空が明るすぎる問題（§5.5、量っていない）を、ボリュームを増やさずに扱える。
- §12 Q14 で空のピクセルの光芒は範囲外と決めているので、入れるかどうかは要判断。

**小さな点**
- 太陽の立体角: GT7 は 6.807×10⁻⁵ sr（平均視半径 0.2666° = 0.004652 rad に当たる）。本実装は参照実装の 0.004675 rad（6.87×10⁻⁵ sr）なので、太陽ディスクの輝度（照度 / Ω）が 0.9% 低い。照度は D14 で決めているので、影響は Ω の分だけ。参照実装との比較を崩さないなら、比較用と実用で値を分ける。
- 等色関数: GT7 は CIE 2006 2°。Rec.709 / Rec.2020 の原色と白色点は CIE 1931 で定義されているので、本実装は 1931 のままにする（2006 にするなら XYZ → RGB の行列も作り直す必要があり、規格の RGB と一致しなくなる）。
- 既定の作業色空間: GT7 は Rec.2020 で出力し、トーンマップの入力も Rec.2020。本実装の V9 でも `SpectralFit2020` が最も分光計算に近かった（ΔE_ITP 平均 1.4〜3.2）。D9 の既定（Rec.709）を見直す材料になる（要決定）。

### 11.5 採らないもの

- **時刻 × 天候の 3D LUT への焼き込み**: カメラ高度も大気パラメータも実行時に変えられなくなり、D3（Shoga で汎用に使う）と合わない。太陽中心・地平線を直線にするマッピングの主な利点（太陽の周りの解像度）は、本実装では Mie の位相関数を LUT の外に出したことですでに得ている（§14 の V3b）。Sky-View LUT の更新をフレームごとに間引きたい場合（モバイル）に、スライス間の補間の考え方を参考にできる。
- **R11G11B10F の LUT**: 仮数が 5〜6 bit で、本実装の検証の精度（10⁻³）に足りない。Shoga のランタイムでは帯域を減らす候補になる。
- **フォグの深度方向の解析式によるフィット（3D → 2D）**: カメラ高度の固定が前提。本実装の 32³ ボリュームは 0.043 ms で十分安い。
- **雲（Cloud G-buffer など）**: 範囲外（design.md §11）。Shoga に雲を入れるときの参考として記録だけしておく（カメラに依存しない上半球のバッファを本画面・ミラー・キューブマップで共用、散乱と照明を分けて遅延シェーディング、非同期コンピュートで多フレームに分割）。

### 11.6 進める順の案

1. G（小さく、既知の最大の誤差を直す）
2. A + B（GT7 の要点。構造はもう整っている）
3. E（A・B の効果と、3 チャンネル化の誤差を測る正解）
4. D → C → F
5. H は V4 の仕組みで試せる実験。I は Q14 の見直しが要る

決める必要がある事項: Mie 計算に外部ライブラリを使うか（numpy に BHMIE を移植するか）、I をやるか、既定の作業色空間。

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
- UE Volumetric Fog: https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-fog-in-unreal-engine
- UE SkyAtmosphereComponent (Python API): https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/SkyAtmosphereComponent
- UE Sky Atmosphere component: https://dev.epicgames.com/documentation/en-us/unreal-engine/sky-atmosphere-component-in-unreal-engine
- B. Wronski, "Assassin's Creed IV: Black Flag - Road to Next-Gen Graphics", GDC 2014. https://bartwronski.com/wp-content/uploads/2014/03/ac4_gdc.pdf
- S. Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015 Advances. https://advances.realtimerendering.com/s2015/index.html
- E. Bruneton, Precomputed Atmospheric Scattering (2017 implementation): https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html
- E. Yusov, "Practical Implementation of Light Scattering Effects Using Epipolar Sampling and 1D Min/Max Binary Trees", GDC 2013. https://gdcvault.com/play/1018794/Practical-Implementation-of-Light-Scattering
- J. Chen et al., "Real-Time Volumetric Shadows using 1D Min-Max Mipmaps", I3D 2011. https://cs.dartmouth.edu/wjarosz/publications/chen11realtime.pdf
- Bevy unified volumetrics design doc: https://hackmd.io/@bevy/SyaDBESi1l / AtmosphereMode: https://doc.qu1x.dev/bevy_trackball/bevy/pbr/enum.AtmosphereMode.html
- A. Kovalovs, "Volumetric Effects of The Last of Us: Part Two", SIGGRAPH 2020 Talks. https://history.siggraph.org/learning/volumetric-effects-of-the-last-of-us-part-two-by-kovalovs
- K. Suzuki, K. Yasutomi, 『グランツーリスモ７』における天球レンダリング, CEDEC+KYUSHU 2022. https://s3.amazonaws.com/gran-turismo.com/pdi_publications/CEDEC_KYUSHU_2022_gt7_sky_PUBLICVERSION.pdf
- K. Suzuki, K. Yasutomi, "Realistic Real-time Sky Dome Rendering in Gran Turismo 7", GDC 2023. https://s3.amazonaws.com/gran-turismo.com/pdi_publications/GDC2023_GT7_SKY_RENDERING.pdf
- K. Suzuki, "Rendering Technologies behind Gran Turismo 7: VR, Ray Tracing and Sky Simulation", SIGGRAPH 2023. https://s3.amazonaws.com/gran-turismo.com/pdi_publications/SIGGRAPH2023_RenderingTechBehindGT7.pdf
- Polyphony Digital 発表資料一覧: https://www.polyphony.co.jp/publications/
- A. H. Omar et al., "Development of global aerosol models using cluster analysis of AERONET measurements", JGR 110 (2005). https://doi.org/10.1029/2004JD004874
- J. T. Kider et al., "A Framework for the Experimental Comparison of Solar and Skydome Illumination", ACM TOG 33(6) (2014). https://doi.org/10.1145/2661229.2661259
- AERONET: https://aeronet.gsfc.nasa.gov/
- C. Bohren, D. Huffman, "Absorption and Scattering of Light by Small Particles" (BHMIE), Wiley 1983
