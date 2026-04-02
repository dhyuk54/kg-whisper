# KG-Whisper: キーワードで導く Whisper + AdaKWS

aiOla が発表した以下の 2 本の論文を再現したプロジェクトです:
- **AdaKWS**: [Adaptive Keyword Spotting (arXiv:2309.08561)](https://arxiv.org/abs/2309.08561)
- **KG-Whisper-PT**: [Keyword-Guided Adaptation of ASR (arXiv:2406.02649)](https://arxiv.org/abs/2406.02649)

## 概要

キーワードを活用した音声認識の 2 段階パイプラインです:

1. **AdaKWS** — Whisper Small エンコーダ + Character LSTM + AdaIN によるオープンボキャブラリのキーワード検出
2. **KG-Whisper-PT** — 学習済み prefix ベクトル 12 個（パラメータ数わずか 15K）で Whisper Large-v2 デコーダを誘導するプロンプトチューニング

```
音声 + キーワードリスト → AdaKWS（キーワード検出）→ KG-Whisper-PT（キーワードを活用した文字起こし）
```

## 再現結果

### AdaKWS（VoxPopuli EN テストセット、1842 サンプル）

| 指標 | 再現結果 | 論文の値 |
|------|---------|---------|
| F1（閾値 0.3） | **96.17%** | 96.3% |
| AUC | **98.64%** | — |
| EER | **3.56%** | — |

### KG-Whisper-PT（VoxPopuli EN テストセット、1842 サンプル）

| 手法 | WER |
|------|-----|
| 素の Whisper Large-v2 | 7.50% |
| Baseline（PT のみ、キーワードなし） | 7.15% |
| Combined（AdaKWS + PT） | **7.06%** |
| Oracle（正解キーワードを使用） | 6.91% |

### VoxPopuli 詳細統計（1842 サンプル）

| 分類 | 件数 | 説明 |
|------|------|------|
| 改善 | 110 | Combined の WER が Baseline より低下 |
| 変化なし | 1658 | Combined と Baseline の WER が同一 |
| 　- 両者とも WER = 0% | 707 | すべてのモデルが正解、悪影響なし |
| 　- 両者とも WER > 0% | 951 | キーワードの有無にかかわらず同じ誤り |
| 悪化 | 74 | Combined の WER が Baseline より上昇（負例サンプリングに起因） |

- Baseline WER = 0%: 731 / 1842（39.7%）
- Combined WER = 0%: 754 / 1842（40.9%）— キーワード導入により 23 サンプルが完全正解に

### クロスドメイン: Medical ASR（カスタムキーワードリスト、医療用語 80 語）

| 手法 | WER（上位 20 サンプル） |
|------|----------------------|
| Baseline（キーワードなし） | 50.45% |
| Combined（キーワードリスト使用） | **27.35%** |

- 20 サンプル中 11 件が改善、うち 8 件は WER 0% に到達
- ゼロショットでのクロスドメイン転移: VoxPopuli で学習し、ファインチューニングなしで医療データに適用

### データセット

- **VoxPopuli EN**（[facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)）: 欧州議会における英語スピーチの録音
  - 学習: 約 18K サンプル、テスト: 1842 サンプル
  - AdaKWS と KG-Whisper-PT の学習・評価に使用
- **Medical ASR**（[Hani89/medical_asr_recording_dataset](https://huggingface.co/datasets/Hani89/medical_asr_recording_dataset)）: 患者による症状の口述
  - テスト: 1333 サンプル
  - クロスドメイン評価専用（ゼロショット、ファインチューニングなし）

### 学習設定

- **AdaKWS v3**: 英語、25 エポック、クロスオーディオ負例サンプリング、Whisper Small エンコーダ
- **KG-Whisper-PT v2**: 英語、30K ステップ、バッチ内負例サンプリング、Whisper Large-v2 デコーダ、prefix ベクトル 12 個

## セットアップ

### 動作要件

- Python 3.9 以上
- CUDA 対応 GPU（推奨: VRAM 16GB 以上）、または Apple Silicon Mac（MPS）

### uv によるインストール

```bash
# uv のインストール（未導入の場合）
pip install uv

# リポジトリをクローン
git clone https://github.com/dhyuk54/kg-whisper.git
cd kg-whisper

# 仮想環境を作成して有効化
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows

# プロジェクトをインストール
uv pip install -e .

# 追加の依存パッケージをインストール
uv pip install gradio soundfile datasets nltk scikit-learn
```

### チェックポイントのダウンロード

Google Drive からダウンロード: [チェックポイント](https://drive.google.com/drive/folders/1zqFEOnfrXyP2h9bCijdYqCVvT3VaCZgk?usp=sharing)

以下の構成で配置してください:
```
kg_whisper/outputs/
├── checkpoint_final.pt                          # KG-Whisper-PT (183K)
└── adakws_v3/
    └── adakws_checkpoint_29000.pt               # AdaKWS v3 (261MB)
```

### デモ用音声のダウンロード

Google Drive からダウンロード: [デモ用音声](https://drive.google.com/drive/folders/1MaDFDu-aUwNVy3EuxEDdVL6ShfrXog9D?usp=sharing)

以下の構成で配置してください:
```
demo/
├── audio_best/                    # VoxPopuli 改善上位 20 サンプル
│   ├── best_00.wav ... best_19.wav
│   └── ground_truth.json
├── audio_medical_best/            # Medical 改善上位 20 サンプル
│   ├── medical_best_00.wav ... medical_best_19.wav
│   ├── ground_truth.json
│   └── medical_keywords.txt
└── medical_keywords.txt           # 医療用語 80 語
```

## デモの実行

### CUDA（Linux/Windows）

```bash
python -m demo.app \
    --device cuda \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --whisper_pt_checkpoint kg_whisper/outputs/checkpoint_final.pt
```

### Apple Silicon Mac（M1/M2/M3/M4）

MPS（Metal Performance Shaders）が自動で検出されるため、`--device cuda` の指定は不要です。

```bash
uv run python -m demo.app \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --whisper_pt_checkpoint kg_whisper/outputs/checkpoint_final.pt
```

> **動作確認済み環境**: Mac M4 Max（macOS, Python 3.13, PyTorch 2.11）

ブラウザで http://localhost:7860 を開いてください。

### デモの機能

| タブ | 説明 |
|------|------|
| **Single Sample** | サンプルを選んでパイプライン全体を実行（Baseline / AdaKWS / Combined / Oracle） |
| **Custom Keywords** | 任意の音声をアップロードし、カスタムキーワードリストで認識（実運用モード） |
| **Batch Evaluation** | 全サンプルを一括処理し、結果を CSV でダウンロード |

### デモで利用できるデータセット

ダウンロードしたデモ用音声には以下の **2 つのデータセット**（各 20 サンプル）が含まれており、パイプライン全体の動作確認に十分です:

| データセット | 音声ディレクトリ | 説明 |
|-------------|-----------------|------|
| **Best Samples** | `demo/audio_best/` | VoxPopuli 改善上位 20（最も効果が大きいサンプル） |
| **Medical Best** | `demo/audio_medical_best/` | Medical 改善上位 20（クロスドメイン、最も効果が大きいサンプル） |

> **補足**: コード上は Worst Samples、Medical ASR、VoxPopuli EN のデータセットにも対応していますが、これらの音声ファイルはデモ用ダウンロードには含まれていません。別途準備が必要です。

## わかったこと

1. **AdaKWS の F1 が 96.17%** に到達し、論文の 96.3% とほぼ一致 — キーワード検出の再現に成功
2. **Combined < Baseline**（7.06% < 7.15%）— キーワードの導入で文字起こし精度が向上
3. **クロスドメインでも有効** — VoxPopuli で学習したモデルが、ファインチューニングなしで Medical ASR を改善
4. **カスタムキーワードリスト** — ドメイン固有の 10〜50 語程度を推奨（aiOla 公式ドキュメントより）
5. **音声品質が決め手** — クリアな音声 + キーワードリスト = 最良の結果。音質が悪い場合はどのモデルでも限界がある

## 参考文献

- [AdaKWS: Adaptive Keyword Spotting](https://arxiv.org/abs/2309.08561)
- [KG-Whisper: Keyword-Guided Adaptation of ASR](https://arxiv.org/abs/2406.02649)
- [aiOla Jargonic](https://aiola.ai/jargonic/)
- [OpenAI Whisper](https://github.com/openai/whisper)
