# KG-Whisper: Keyword-Guided Whisper with AdaKWS

Reproduction of two papers from aiOla:
- **AdaKWS**: [Adaptive Keyword Spotting (arXiv:2309.08561)](https://arxiv.org/abs/2309.08561)
- **KG-Whisper-PT**: [Keyword-Guided Adaptation of ASR (arXiv:2406.02649)](https://arxiv.org/abs/2406.02649)

## Overview

A two-stage pipeline for keyword-guided speech recognition:

1. **AdaKWS** - Open-vocabulary keyword spotting using Whisper Small encoder + Character LSTM + AdaIN
2. **KG-Whisper-PT** - Prompt tuning with 12 learned prefix vectors (15K params) to guide Whisper Large-v2 decoder

```
Audio + Keyword List → AdaKWS (detect keywords) → KG-Whisper-PT (guided transcription)
```

## Reproduction Results

### AdaKWS (VoxPopuli EN test, 1842 samples)

| Metric | Our Result | Paper Reference |
|--------|-----------|-----------------|
| F1 (threshold 0.3) | **96.17%** | 96.3% |
| AUC | **98.64%** | - |
| EER | **3.56%** | - |

### KG-Whisper-PT (VoxPopuli EN test, 1842 samples)

| Method | WER |
|--------|-----|
| Pure Whisper Large-v2 | 7.50% |
| Baseline (PT, no keywords) | 7.15% |
| Combined (AdaKWS + PT) | **7.06%** |
| Oracle (perfect keywords) | 6.91% |

### VoxPopuli Detailed Statistics (1842 samples)

| Category | Count | Description |
|----------|-------|-------------|
| Improved | 110 | Combined WER < Baseline WER |
| Same | 1658 | Combined WER = Baseline WER |
| - Both WER = 0% | 707 | All models correct, no interference |
| - Both WER > 0% | 951 | Same errors with or without keywords |
| Worse | 74 | Combined WER > Baseline WER (due to negative sampling artifacts) |

- Baseline WER = 0%: 731 / 1842 (39.7%)
- Combined WER = 0%: 754 / 1842 (40.9%) — 23 more samples corrected to perfect by keywords

### Cross-Domain: Medical ASR (custom keyword list, 80 medical terms)

| Method | WER (20 best samples) |
|--------|----------------------|
| Baseline (no keywords) | 50.45% |
| Combined (with keyword list) | **27.35%** |

- 11/20 samples improved, 8 corrected to 0% WER
- Zero-shot cross-domain: trained on VoxPopuli, tested on Medical without fine-tuning

### Training Configuration

- **AdaKWS v3**: English, 25 epochs, cross-audio negative sampling, Whisper Small encoder
- **KG-Whisper-PT v2**: English, 30K steps, batch-internal negative sampling, Whisper Large-v2 decoder, 12 prefix vectors

## Setup

### Requirements

- Python 3.9+
- CUDA GPU (recommended: 16GB+ VRAM)

### Install with uv

```bash
# Install uv (if not installed)
pip install uv

# Clone the repository
git clone https://github.com/dhyuk54/kg-whisper.git
cd kg-whisper

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install the project
uv pip install -e .

# Install additional dependencies
uv pip install gradio soundfile datasets nltk
```

### Download Checkpoints

Download from Google Drive: [Checkpoints](https://drive.google.com/drive/folders/1MaDFDu-aUwNVy3EuxEDdVL6ShfrXog9D?usp=sharing)

Place the files as follows:
```
kg_whisper/outputs/
├── checkpoint_final.pt                          # KG-Whisper-PT (183K)
└── adakws_v3/
    └── adakws_checkpoint_29000.pt               # AdaKWS v3 (261MB)
```

### Download Demo Audio

Download from Google Drive: [Demo Audio](https://drive.google.com/drive/folders/1zqFEOnfrXyP2h9bCijdYqCVvT3VaCZgk?usp=sharing)

Place the files as follows:
```
demo/
├── audio_best/                    # VoxPopuli best 20 samples
│   ├── best_00.wav ... best_19.wav
│   └── ground_truth.json
├── audio_medical_best/            # Medical best 20 samples
│   ├── medical_best_00.wav ... medical_best_19.wav
│   ├── ground_truth.json
│   └── medical_keywords.txt
└── medical_keywords.txt           # 80 medical terms
```

## Running the Demo

```bash
python -m demo.app \
    --device cuda \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --whisper_pt_checkpoint kg_whisper/outputs/checkpoint_final.pt
```

Open http://localhost:7860 in your browser.

### Demo Features

| Tab | Description |
|-----|-------------|
| **Single Sample** | Select a sample, run full pipeline (Baseline / AdaKWS / Combined / Oracle) |
| **Custom Keywords** | Upload any audio + custom keyword list (real-world mode) |
| **Batch Evaluation** | Run all samples with optional custom keyword list, download CSV |

### Available Datasets in Demo

| Dataset | Description |
|---------|-------------|
| Best Samples | VoxPopuli top 20 (highest improvement) |
| Medical Best | Medical top 20 (cross-domain, highest improvement) |

## Training

### Train KG-Whisper-PT

```bash
python -m kg_whisper.train \
    --device cuda \
    --cache_dir "YOUR_CACHE_DIR" \
    --total_steps 30000 \
    --batch_size 4
```

### Train AdaKWS

```bash
python -m kg_whisper.adakws_train \
    --device cuda \
    --cache_dir "YOUR_CACHE_DIR" \
    --epochs 25
```

## Evaluation

### Combined Evaluation (VoxPopuli)

```bash
python -m kg_whisper.eval_combined \
    --pt_checkpoint kg_whisper/outputs/checkpoint_final.pt \
    --adakws_checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --device cuda --cache_dir "YOUR_CACHE_DIR" --mode all
```

### AdaKWS F1/AUC/EER

```bash
python -m kg_whisper.adakws_eval \
    --checkpoint kg_whisper/outputs/adakws_v3/adakws_checkpoint_29000.pt \
    --device cuda --cache_dir "YOUR_CACHE_DIR"
```

## Project Structure

```
kg_whisper/
├── model.py              # KGWhisperPT model (prefix tuning)
├── config.py             # Configuration
├── data.py               # Dataset and dataloader
├── train.py              # KG-Whisper-PT training
├── adakws_model.py       # AdaKWS model (Whisper encoder + CharLSTM + AdaIN)
├── adakws_data.py        # AdaKWS dataset and negative sampling
├── adakws_train.py       # AdaKWS training (with cross-audio negatives)
├── adakws_eval.py        # AdaKWS evaluation (F1/AUC/EER)
├── eval_combined.py      # Full pipeline evaluation
├── eval_medical.py       # Medical domain evaluation
├── kws_simulator.py      # Keyword sampling simulator
└── outputs/              # Checkpoints (download separately)

demo/
├── app.py                # Gradio demo application
├── find_best_samples.py  # Find best VoxPopuli samples
├── find_worst_samples.py # Find worst VoxPopuli samples
├── find_best_medical_samples.py   # Find best Medical samples
├── find_worst_medical_samples.py  # Find worst Medical samples
├── precompute_keywords.py         # Pre-compute keywords for stable results
├── stats_voxpopuli.py             # VoxPopuli statistics
└── medical_keywords.txt           # Medical keyword list (80 terms)
```

## Key Findings

1. **AdaKWS F1 96.17%** matches paper's 96.3% — keyword detection successfully reproduced
2. **Combined < Baseline** (7.06% < 7.15%) — keyword guidance improves transcription
3. **Cross-domain works** — trained on VoxPopuli, improves Medical ASR without fine-tuning
4. **Custom keyword list** — 10-50 domain-specific terms recommended (per aiOla docs)
5. **Audio quality is key** — clear audio + keyword list = best results; poor audio = no model can help

## References

- [AdaKWS: Adaptive Keyword Spotting](https://arxiv.org/abs/2309.08561)
- [KG-Whisper: Keyword-Guided Adaptation of ASR](https://arxiv.org/abs/2406.02649)
- [aiOla Jargonic](https://aiola.ai/jargonic/)
- [OpenAI Whisper](https://github.com/openai/whisper)

## License

This project is built on top of [OpenAI Whisper](https://github.com/openai/whisper) (MIT License).
