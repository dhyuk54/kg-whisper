# KG-Whisper: キーワードガイド付きWhisper + AdaKWS

aiOlaの2つの論文の再現:
- **AdaKWS**: [Adaptive Keyword Spotting (arXiv:2309.08561)](https://arxiv.org/abs/2309.08561)
- **KG-Whisper-PT**: [Keyword-Guided Adaptation of ASR (arXiv:2406.02649)](https://arxiv.org/abs/2406.02649)

## 概要

キーワードガイド付き音声認識のための2段階パイプライン:

1. **AdaKWS** - Whisper Smallエンコーダ + Character LSTM + AdaINによるオープンボキャブラリーキーワード検出
2. **KG-Whisper-PT** - 12個の学習済みprefixベクトル（15Kパラメータ）でWhisper Large-v2デコーダをガイドするプロンプトチューニング

```
音声 + キーワードリスト → AdaKWS (キーワード検出) → KG-Whisper-PT (ガイド付き文字起こし)
```

## 再現結果

### AdaKWS (VoxPopuli EN テスト, 1842サンプル)

| 指標 | 再現結果 | 論文参照値 |
|------|---------|-----------|
| F1 (閾値 0.3) | **96.17%** | 96.3% |
| AUC | **98.64%** | - |
| EER | **3.56%** | - |

### KG-Whisper-PT (VoxPopuli EN テスト, 1842サンプル)

| 方法 | WER |
|------|-----|
| 純粋なWhisper Large-v2 | 7.50% |
| Baseline (PT、キーワードなし) | 7.15% |
| Combined (AdaKWS + PT) | **7.06%** |
| Oracle (完璧なキーワード) | 6.91% |

### VoxPopuli 詳細統計 (1842サンプル)

| 分類 | 数量 | 説明 |
|------|------|------|
| 改善 | 110 | Combined WER < Baseline WER |
| 同一 | 1658 | Combined WER = Baseline WER |
| - 両方WER = 0% | 707 | 全モデル正確、干渉なし |
| - 両方WER > 0% | 951 | キーワードの有無に関わらず同一のエラー |
| 悪化 | 74 | Combined WER > Baseline WER（ネガティブサンプリングのアーティファクト） |

- Baseline WER = 0%: 731 / 1842 (39.7%)
- Combined WER = 0%: 754 / 1842 (40.9%) — キーワードにより23サンプルが完璧に修正

### クロスドメイン: Medical ASR (カスタムキーワードリスト、80医療用語)

| 方法 | WER (上位20サンプル) |
|------|-------------------|
| Baseline (キーワードなし) | 50.45% |
| Combined (キーワードリスト使用) | **27.35%** |

- 20中11サンプルが改善、8サンプルがWER 0%に修正
- ゼロショットクロスドメイン: VoxPopuliで学習、ファインチューニングなしでMedicalでテスト

### 学習構成

- **AdaKWS v3**: 英語、25エポック、クロスオーディオネガティブサンプリング、Whisper Smallエンコーダ
- **KG-Whisper-PT v2**: 英語、30Kステップ、バッチ内ネガティブサンプリング、Whisper Large-v2デコーダ、12 prefixベクトル

## セットアップ

### 要件

- Python 3.9+
- CUDA GPU (推奨: 8GB+ VRAM)

### uvでインストール

```bash
# uvのインストール（未インストールの場合）
pip install uv

# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/kg-whisper.git
cd kg-whisper

# 仮想環境の作成と依存関係のインストール
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows

# プロジェクトのインストール
uv pip install -e .

# 追加依存関係のインストール
uv pip install gradio soundfile datasets nltk
```

### チェックポイントのダウンロード

Google Driveからダウンロード: [チェックポイント](https://drive.google.com/drive/folders/1MaDFDu-aUwNVy3EuxEDdVL6ShfrXog9D?usp=sharing), [デモ音声](https://drive.google.com/drive/folders/1zqFEOnfrXyP2h9bCijdYqCVvT3VaCZgk?usp=sharing)

以下のように配置:
```
kg_whisper/outputs/
├── checkpoint_final.pt                          # KG-Whisper-PT (183K)
└── adakws_v3/
    └── adakws_checkpoint_29000.pt               # AdaKWS v3 (261MB)
```

## デモの実行

```bash
python -m demo.app \
    --device cuda \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --whisper_pt_checkpoint kg_whisper/outputs/checkpoint_final.pt
```

ブラウザで http://localhost:7860 を開きます。

### デモ機能

| タブ | 説明 |
|------|------|
| **Single Sample** | サンプルを選択し、フルパイプラインを実行 (Baseline / AdaKWS / Combined / Oracle) |
| **Custom Keywords** | 音声アップロード + カスタムキーワードリスト（実運用モード） |
| **Batch Evaluation** | 全サンプル実行、カスタムキーワードリストオプション、CSVダウンロード |

## 主な発見

1. **AdaKWS F1 96.17%**は論文の96.3%と一致 — キーワード検出の再現に成功
2. **Combined < Baseline** (7.06% < 7.15%) — キーワードガイドが文字起こしを改善
3. **クロスドメインが機能** — VoxPopuliで学習、ファインチューニングなしでMedical ASRを改善
4. **カスタムキーワードリスト** — ドメイン固有の10-50用語を推奨（aiOlaドキュメント基準）
5. **音声品質が鍵** — クリアな音声 + キーワードリスト = 最良の結果

## 参考文献

- [AdaKWS: Adaptive Keyword Spotting](https://arxiv.org/abs/2309.08561)
- [KG-Whisper: Keyword-Guided Adaptation of ASR](https://arxiv.org/abs/2406.02649)
- [aiOla Jargonic](https://aiola.ai/jargonic/)
- [OpenAI Whisper](https://github.com/openai/whisper)
