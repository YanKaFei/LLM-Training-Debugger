<div align="center">

<img src="../../assets/banner.svg" alt="LLM Training Debugger" width="100%">

<br>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [한국어](README.ko.md) · **日本語** · [Français](README.fr.md)

<br>

[![tests](https://img.shields.io/badge/tests-584_passing-3fb950?style=flat-square)](../../tests)
[![python](https://img.shields.io/badge/python-3.9%2B-58a6ff?style=flat-square)](../../pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-8b949e?style=flat-square)](../../LICENSE)
[![status](https://img.shields.io/badge/status-alpha-d29922?style=flat-square)](#プロジェクトの状態)
[![deepseek-harness](https://img.shields.io/badge/built_with-DeepSeek_Harness-4d6bfe?style=flat-square)](https://github.com/topics/deepseek-harness)

</div>

---

## これは何か

**証拠優先（evidence-first）の PyTorch 学習モニタリングライブラリ**です。学習ループを観察し、
指標の異常をインシデントへと変換します。そして最も重要な点として ——
**証拠が結論を支えないときは、結論を出すことを拒否します**。

<div align="center">
<img src="../../assets/architecture.svg" alt="アーキテクチャ" width="100%">
</div>

多くの学習ツールは「何が起きたか」に答えます。このライブラリは、より狭くより難しい問いを立てます：

> **私は本当に知っているのか。そうでないなら、手元にあるものは正確には何なのか。**

## これは何ではないか

境界を先に述べます。正直な境界のほうが機能一覧より有用だからです：

- **実験トラッカーの代替ではありません。** トラッカーへ出力しますが、それ自体はダッシュボードではありません。
- **根本原因分析ではありません。** 報告するのは*どのステップ、どの指標*かです。どのカーネル、どのテンソル、
  どのコード行かは答えられません。
- **コントローラではありません。** 観察するだけです。勾配をクリップせず、オプティマイザ状態を初期化せず、
  学習を停止しません。
- **大規模での検証はありません。** 本リポジトリのすべての数値は 4〜6 層モデルで測定されたものです。
- **検証済みではありません。** ブラインド・ホールドアウトは存在しません。`validated_capabilities` は設計上空です。

## なぜ存在するのか

ひとつの学習は loss 曲線を生みます。曲線がおかしくなったとき、曲線はどこが問題かを語らず、
再実行しても再現しないことがほとんどです。

このライブラリの立場はこうです：**成果物はグラフではない。成果物はインシデントである** ——
何が観察されたかについての、**境界があり、監査可能で、タイムスタンプを持つ**主張であり、
その根拠を伴い、**何を結論できなかったかを明示**します。

この立場は設計全体に現れています：

| 原則 | コードにおける現れ |
|---|---|
| 観察は学習を決して妨げてはならない | `emit()` は例外を投げない。P0–P3 の有界キューはブロックせず破棄する |
| 機構を主張する前に証拠が観察されねばならない | `OBSERVED` / `INFERRED` / `HYPOTHESIS` / `UNKNOWN`。昇格可能なのは `OBSERVED` のみ |
| 観察の欠如は否定的観察ではない | すべての判定が `data_quality` と `completeness_warning` を持つ |
| でっち上げた関係より棄権を優先する | `INSUFFICIENT_EVIDENCE` はエラーではなく一級の判定 |
| 一部の因果的主張は不可能であるべき | `CAUSED_BY` と `ROOT_CAUSE_OF` は**スキーマに存在せず**、永久にそうであるようテストが強制する |

## 動作要件

| | |
|---|---|
| **Python** | 3.9 – 3.13。すべてのバージョンを CI で実際に検証しています。 |
| **コア依存** | `numpy>=1.23`、`pyarrow>=12`、`duckdb>=0.10,<1.5` |
| **PyTorch アダプタ** | `torch>=2.0` — オプションで `torch` extra から導入します。observer、storage、detectors、episodes、fusion、CLI は torch を import しません。 |
| **開発** | `.[dev]` = `pytest` **と** torch。テストスイートが torch を import するためです。 |

> **`duckdb` の上限は好みではなく回避策です。** duckdb 1.5.6 は `fetch_arrow_table()` で
> `SystemError: <built-in function __import__> returned a result with an exception set` を
> 発生させます。CI の 23 件の失敗はすべてその 1 箇所で起こり、残りの約 560 件のテストは
> 通過し `import pyarrow` も正常でした。したがって原因は pyarrow ではなく duckdb の
> Arrow ブリッジにあります。Python 3.9 では見えません。duckdb 1.5.x が
> **Python >= 3.10** を要求するため、3.9 は 1.4.5 に解決され決して遭遇しないからです。
> つまり **Python 3.10 以降でバージョンを固定していない利用者は全員、壊れたビルドを
> 受け取っていました。**
>
> *latest dependencies (informational)* という CI ジョブは、すべての上限を意図的に無視し、
> 失敗を許容しています。**これが赤い間は上限が必要であり、緑になったら上限を外すべき
> です。** [KNOWN_LIMITATIONS.md](../../KNOWN_LIMITATIONS.md) の 15 番を参照。

## クイックスタート

```bash
git clone https://github.com/YanKaFei/LLM-Training-Debugger.git
cd LLM-Training-Debugger
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

完全に実行可能な例：

```bash
python examples/quickstart.py
```

既存のループに組み込むこともできます：

```python
from training_debugger.adapters.pytorch import PyTorchTrainingAdapter

adapter = PyTorchTrainingAdapter(observer, model, optimizer, model_id="my-model")
adapter.run_start(steps=total_steps)

for step in range(1, total_steps + 1):
    with adapter.step(step) as tracker:
        loss = model(**batch)["loss"]
        tracker.record_loss(loss)
        loss.backward()
        tracker.record_gradients()
        with tracker.phase("optimizer"):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tracker.record_parameters()

adapter.run_end()
```

## 得られるもの

<div align="center">
<img src="../../assets/verdicts.svg" alt="判定" width="100%">
</div>

各インシデントは `reason_code`、`evidence_state`、`promotion_disposition` を持ち、
ストリームが不完全だった場合は `data_quality` 警告を伴います。
**イベントが欠落した実行が、健全な実行として表示されることは決してありません。**

## コマンド

<div align="center">
<img src="../../assets/cli-demo.svg" alt="CLI デモ" width="100%">
</div>

```bash
training-debugger --root <store> runs                  # 実行一覧
training-debugger --root <store> verdicts  <run>       # 棄権を含むすべての判定
training-debugger --root <store> incidents <run>       # 根拠付きのインシデント
training-debugger --root <store> incident  <run> <id>  # インシデント1件の全体
training-debugger --root <store> metrics   <run> [m]   # 指標行の照会
training-debugger --root <store> diagnose  <run>       # 決定論的診断
training-debugger --root <store> compare   <a> <b>     # 2つの実行が分岐する最初のステップ
training-debugger --root <store> scan      <r1> ...    # バッチ：どの実行が外れ値か
```

### `compare` —— 分岐の特定

2つの実行の指標系列を整列し、**初期に両者が一致している区間を帰無モデルとして用い**、
そこを離れて留まる最初のステップを報告します。共通のノイズは差分で相殺されるため、
単一実行の閾値では不可能な特定が可能になります。

注入ステップが既知のペアコーパスでの測定：**9/9 の故障ペアが正確な注入ステップに特定され、
3/3 の正常ペアは正しく棄権しました。**

ウォールクロックとメモリの指標は既定で除外されます —— *タイミングの差はアルゴリズム的分岐ではありません*。

### `scan` —— バッチ内の外れ値を見つける

すべてのペアを一度ずつ比較し、各実行のスコアは何人の仲間と分岐したかです。
**like-for-like の実行が必要です**：シードと構成を固定し、1つの要素だけを変えてください。
シードが混在したバッチでは、実行名をでっち上げる代わりに `NO_SINGLE_OUTLIER` を返し、その理由を説明します。

## 機能

すべての機能は明示的な証拠レベルを持ちます。ここで検証済みとされているものはありません。

| 機能 | 状態 | 証拠 |
|---|---|---|
| 非有限値の観察（NaN / Inf） | `EXPERIMENTAL` | 厳密。閾値不要 |
| 勾配異常検出 | `EXPERIMENTAL` | 検出境界 ×20・×40 → 6/6、×2・×5 → 0/6 |
| 棄権を伴うインシデント融合 | `EXPERIMENTAL` | 実行 precision 0.947 / recall 0.500（42 実行コーパス） |
| 2実行の分岐特定 | `EXPERIMENTAL` | 9/9 厳密、3/3 棄権（ペアコーパス） |
| バッチ外れ値スキャン | `EXPERIMENTAL` | like-for-like バッチで 5/5 の仲間が支持 |
| モジュール別活性化ヘルス | `EXPERIMENTAL` | 死んだモジュールを OBSERVED 1.0 で特定・時刻同定 |
| オプティマイザ呼び出しウィットネス | `EXPERIMENTAL` | 凍結済み。既定で無効 |
| 関係推論（順序のみ） | `UNSUPPORTED` | 既定で無効。ある測定済みの試行が 16 件の関連をでっち上げた |

## 既知の制限

公開しています。**失敗が記録されていないモニタは信頼できない**からです：

- **実行レベルの recall は 0.500。** 開発コーパスの主要故障の半分を取り逃します。
- **健全な実行の誤検知率 7.7%**、自己設定のゲートは 5%。**このゲートは不合格です。**
- **オブザーバのオーバーヘッドは自身の <3% 目標を達成できていません**（4層ベンチマーク作業負荷）。
  コストはステップあたりほぼ*固定*（0.2–0.9 ms）であるため、作業負荷が大きくなるほど割合は崩壊します：
  +11.0%（4層）→ +5.2%（8層）→ +0.4%（12層）。ステップが約 17 ms を超えれば予算内です。
  **これは言い訳せず、失敗として報告します。**
- ブラインド・ホールドアウトは存在せず、すべての数値は開発コーパス由来です。
- 4〜6 層モデルでのみ検証されています。

全リスト：[KNOWN_LIMITATIONS.md](../../KNOWN_LIMITATIONS.md) · 設計記録：
[DECISIONS.md](../DECISIONS.md) · **読みやすい索引：[DECISIONS_INDEX.md](../DECISIONS_INDEX.md)** —
このプロジェクトが訂正・撤回を余儀なくされた全主張の表から始まります · 現在の状態：[DEVELOPMENT_STATE.md](../DEVELOPMENT_STATE.md)

## 数値の再現

```bash
pytest -q                                            # 584 成功、1 スキップ
python tests/experiments/full_pipeline_smoke.py      # 8/8
python examples/quickstart.py
python tests/performance/audit_observer_cost.py      # 信頼区間付きのペア比較オーバーヘッド
```

本 README の主要な数値はすべて `tests/experiments/` または `tests/performance/` のスクリプトが生成し、
生の出力は `outputs/` 以下にコミットされています。

## プロジェクトの状態

**Alpha。** インターフェースは安定しておらず、API は予告なく変更される可能性があります。
学習への計装と監査可能な判定の読み取りには実際に有用ですが、
その上に本番パイプラインを構築する段階ではありません。

## フィードバックと議論

**GitHub がこのプロジェクトを議論する唯一の場所です。** チャットサーバーもメーリングリストも
他のチャネルもありません —— したがってここで提起された内容は誰でも見られ、検索可能なまま残ります。

| | |
|---|---|
| **バグを見つけましたか？** | [Issue を作成](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **質問がある、設計判断について議論したい** | [Discussion を開始](https://github.com/YanKaFei/LLM-Training-Debugger/discussions) |
| **機能が欲しい** | [Issue を作成](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) |
| **ここでの主張に同意できない** | [Issue を作成](https://github.com/YanKaFei/LLM-Training-Debugger/issues/new) —— ぜひお願いします |

**反論を歓迎します。とりわけ数値に対する反論を歓迎します。** 本 README の測定を再現して
異なる結果を得たなら、それは提出する価値のあるバグ報告です。ここでの主張が本リポジトリの
証拠に支えられていないなら、**その主張は誤りであり**、訂正または撤回されるべきです。

報告の際は次を含めてください：実行したコマンド、モデル形状とステップ数、PyTorch のバージョン、完全な出力。

## コントリビューション

[CONTRIBUTING.md](../../CONTRIBUTING.md) を参照してください。要約：**主張を弱くする変更は
強くする変更と同様に歓迎され**、**テストを通すためにテストを弱めることは決して許されません**。

## ライセンス

[Apache-2.0](../../LICENSE)。

<div align="center">
<br>
<sub><a href="https://github.com/topics/deepseek-harness">DeepSeek Harness</a> で構築。</sub>
</div>
