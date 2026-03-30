"""KG-Whisper Demo — Gradio UI.

Follows the paper's evaluation pipeline:
  1. Baseline: KG-Whisper-PT transcription without keywords
  2. AdaKWS: detect keywords from candidate list (positives + negatives)
  3. Combined: KG-Whisper-PT transcription WITH AdaKWS-detected keywords
  4. Oracle: KG-Whisper-PT transcription with perfect keywords

Supports Medical ASR and VoxPopuli EN datasets.

Usage:
    python -m demo.app --device cuda
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gradio as gr

import whisper
import whisper.audio
from whisper.normalizers import EnglishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──
DEMO_DIR = Path(__file__).parent

# Dataset configs
DATASETS = {
    "Best Samples": {
        "audio_dir": DEMO_DIR / "audio_best",
        "ground_truth": DEMO_DIR / "audio_best" / "ground_truth.json",
    },
    "Worst Samples": {
        "audio_dir": DEMO_DIR / "audio_worst",
        "ground_truth": DEMO_DIR / "audio_worst" / "ground_truth.json",
    },
    "Medical ASR": {
        "audio_dir": DEMO_DIR / "audio",
        "ground_truth": DEMO_DIR / "ground_truth.json",
    },
    "Medical Best": {
        "audio_dir": DEMO_DIR / "audio_medical_best",
        "ground_truth": DEMO_DIR / "audio_medical_best" / "ground_truth.json",
    },
    "VoxPopuli EN": {
        "audio_dir": DEMO_DIR / "audio_voxpopuli",
        "ground_truth": DEMO_DIR / "ground_truth_voxpopuli.json",
    },
}

# Global model holders
models = {}


def load_ground_truth(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_models(device: str, adakws_checkpoint: str, whisper_pt_checkpoint: str):
    logger.info("Initializing models on device=%s", device)
    models["device"] = device
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # 1. Load Whisper large-v2 for KG-Whisper-PT
    logger.info("Loading Whisper large-v2...")
    whisper_model = whisper.load_model("large-v2", device=device)
    models["whisper"] = whisper_model

    # 2. KG-Whisper-PT
    logger.info("Loading KG-Whisper-PT prefix from %s", whisper_pt_checkpoint)
    from whisper.tokenizer import get_tokenizer
    from kg_whisper.model import KGWhisperPT
    ckpt = torch.load(whisper_pt_checkpoint, map_location="cpu", weights_only=True)
    kg_meta = ckpt["config"]
    tokenizer = get_tokenizer(
        whisper_model.is_multilingual,
        num_languages=whisper_model.num_languages,
        language=kg_meta.get("language", "en"),
        task="transcribe",
    )
    kg_model = KGWhisperPT(
        whisper_model=whisper_model,
        prefix_length=kg_meta["prefix_length"],
        eot_id=tokenizer.eot,
    ).to(device)
    kg_model.prefix.data.copy_(ckpt["prefix"].to(device))
    kg_model.eval()
    models["kg_whisper"] = kg_model
    models["tokenizer"] = tokenizer

    # 3. AdaKWS
    logger.info("Loading AdaKWS from %s", adakws_checkpoint)
    from kg_whisper.adakws_eval import load_adakws_model
    adakws_model, char_vocab, _ = load_adakws_model(
        Path(adakws_checkpoint), device
    )
    models["adakws"] = adakws_model
    models["char_vocab"] = char_vocab

    # 4. Load ground truths
    for name, cfg in DATASETS.items():
        gt_path = cfg["ground_truth"]
        if gt_path.exists():
            models[f"gt_{name}"] = load_ground_truth(gt_path)
            logger.info("Loaded %s: %d samples", name, len(models[f"gt_{name}"]))

    # 5. Build vocab filter from Whisper tokenizer
    vocab = set()
    for i in range(tokenizer.encoding.n_vocab):
        try:
            t = tokenizer.encoding.decode([i]).strip().lower()
            if t and t.isalpha() and len(t) > 1:
                vocab.add(t)
        except Exception:
            pass
    models["vocab_filter"] = vocab
    logger.info("Vocab filter: %d words", len(vocab))

    # 6. Load custom keyword lists
    kws_file = DEMO_DIR / "medical_keywords.txt"
    if kws_file.exists():
        with open(kws_file, "r", encoding="utf-8") as f:
            models["medical_keywords"] = [line.strip().lower() for line in f if line.strip()]
        logger.info("Loaded %d medical keywords", len(models["medical_keywords"]))

    models["n_mels"] = whisper_model.dims.n_mels
    models["current_dataset"] = "Best Samples"
    logger.info("All models loaded.")


# ── Inference helpers ──

def get_current_gt():
    return models[f"gt_{models['current_dataset']}"]


def get_current_audio_dir():
    return DATASETS[models["current_dataset"]]["audio_dir"]


def prepare_mel(audio_path: str) -> torch.Tensor:
    device = models["device"]
    audio = whisper.audio.load_audio(audio_path)
    audio = whisper.audio.pad_or_trim(torch.from_numpy(audio).float())
    mel = whisper.audio.log_mel_spectrogram(audio, n_mels=models["n_mels"])
    return mel.unsqueeze(0).to(device)


def prepare_mel_adakws(audio_path: str) -> torch.Tensor:
    device = models["device"]
    n_mels = models["adakws"].encoder.conv1.weight.shape[1]
    audio = whisper.audio.load_audio(audio_path)
    audio = whisper.audio.pad_or_trim(torch.from_numpy(audio).float())
    mel = whisper.audio.log_mel_spectrogram(audio, n_mels=n_mels)
    return mel.unsqueeze(0).to(device)


def transcribe_with_keywords(mel: torch.Tensor, keywords: list) -> str:
    """Transcribe with KG-Whisper-PT using given keywords."""
    device = models["device"]
    kg_model = models["kg_whisper"]
    tokenizer = models["tokenizer"]

    from kg_whisper.config import KGWhisperConfig
    config = KGWhisperConfig()

    kws_string = " | ".join(keywords) if keywords else ""
    kws_token_ids = tokenizer.encoding.encode(kws_string) if kws_string else []
    kws_token_ids = kws_token_ids[:config.max_kws_tokens]
    pad_len = config.max_kws_tokens - len(kws_token_ids)
    kws_padded = kws_token_ids + [tokenizer.eot] * pad_len

    prompt_tokens = (
        [tokenizer.sot_prev]
        + kws_padded
        + list(tokenizer.sot_sequence_including_notimestamps)
    )
    prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    gen_ids = kg_model.generate(mel, prompt_tokens, eot_id=tokenizer.eot)
    return tokenizer.decode(gen_ids).strip()


def run_adakws_detect(mel: torch.Tensor, keywords: list, threshold: float = 0.3) -> list:
    """Run AdaKWS and return detected keywords with probabilities."""
    detected = models["adakws"].predict(mel, keywords, models["char_vocab"], threshold=threshold)
    vocab = models.get("vocab_filter")
    if vocab:
        detected = [(kw, prob) for kw, prob in detected if kw.lower() in vocab]
    return detected


def sample_keywords(text: str, n_positive: int = 3, n_negative: int = 17) -> tuple:
    """Sample keywords following paper protocol: 3 positive + 17 negative."""
    from kg_whisper.adakws_data import NegativeSampler

    words = [re.sub(r"[^\w'-]", "", w).strip("'").lower() for w in text.strip().split()]
    words = [w for w in words if w]
    if not words:
        return [], set()

    positive_words_set = set(words)

    # Positive: random words from transcript
    unique_words = list(set(words))
    n_pos = min(n_positive, len(unique_words))
    positives = random.sample(unique_words, n_pos)

    # Negative: using NegativeSampler (char_sub, concat, random), no duplicates
    neg_sampler = NegativeSampler(unique_words)
    negatives = []
    used = set(w.lower() for w in positives)
    attempts = 0
    while len(negatives) < n_negative and attempts < n_negative * 5:
        pos_kw = random.choice(words)
        neg = neg_sampler.sample(positive_words_set, pos_kw)
        if neg.lower() not in used:
            used.add(neg.lower())
            negatives.append(neg)
        attempts += 1

    all_keywords = positives + negatives
    random.shuffle(all_keywords)

    return all_keywords, set(w.lower() for w in positives)


_normalizer = EnglishTextNormalizer()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = _normalizer(reference).split()
    hyp_words = _normalizer(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# ── Dataset switching ──

def switch_dataset(dataset_name: str):
    models["current_dataset"] = dataset_name
    gt = get_current_gt()
    sample_names = sorted(gt.keys())
    return gr.update(choices=sample_names, value=sample_names[0] if sample_names else None)


# ── Gradio callback: single sample ──

def transcribe_sample(sample_name: str):
    if not sample_name:
        return ("",) * 10

    random.seed(42)  # reproducible keyword sampling

    gt = get_current_gt()
    audio_dir = get_current_audio_dir()
    audio_path = str(audio_dir / sample_name)
    reference_text = gt.get(sample_name, {}).get("text", "")

    mel = prepare_mel(audio_path)
    mel_kws = prepare_mel_adakws(audio_path)

    # Step 1: Baseline — transcribe without keywords
    t0 = time.time()
    baseline_text = transcribe_with_keywords(mel, [])
    baseline_time = time.time() - t0
    baseline_wer = compute_wer(reference_text, baseline_text)

    # Step 2: Sample keywords (3 positive + 17 negative) from reference
    all_keywords, positive_set = sample_keywords(reference_text)
    candidates_str = ", ".join([
        f"**{kw}**" if kw.lower() in positive_set else kw
        for kw in all_keywords
    ])

    # Step 3: AdaKWS detection
    t0 = time.time()
    detected = run_adakws_detect(mel_kws, all_keywords, threshold=0.3)
    kws_time = time.time() - t0
    # Deduplicate detected keywords while preserving order
    seen = set()
    detected_keywords = []
    for kw, _ in detected:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            detected_keywords.append(kw)
    detected_str = ", ".join([f"{kw} ({prob:.0%})" for kw, prob in detected]) if detected else "None"

    # Step 4: Combined — transcribe with AdaKWS-detected keywords
    t0 = time.time()
    combined_text = transcribe_with_keywords(mel, detected_keywords)
    combined_time = time.time() - t0
    combined_wer = compute_wer(reference_text, combined_text)

    # Step 5: Oracle — transcribe with perfect keywords
    oracle_keywords = [kw for kw in all_keywords if kw.lower() in positive_set]
    t0 = time.time()
    oracle_text = transcribe_with_keywords(mel, oracle_keywords)
    oracle_time = time.time() - t0
    oracle_wer = compute_wer(reference_text, oracle_text)

    # KWS accuracy
    detected_set = set(kw.lower() for kw in detected_keywords)
    tp = len(detected_set & positive_set)
    fp = len(detected_set - positive_set)
    fn = len(positive_set - detected_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    kws_metrics = (
        f"Precision: {precision:.0%} | Recall: {recall:.0%} | F1: {f1:.0%}\n"
        f"TP: {tp} | FP: {fp} | FN: {fn} | Time: {kws_time:.2f}s"
    )

    logger.info("Transcribed %s: Baseline=%.2f%%, Combined=%.2f%%, Oracle=%.2f%%, F1=%.0f%%",
                sample_name, baseline_wer * 100, combined_wer * 100, oracle_wer * 100, f1 * 100)

    return (
        audio_path,
        reference_text,
        candidates_str,
        detected_str,
        kws_metrics,
        f"{baseline_text}\n\nWER: {baseline_wer:.2%} | Time: {baseline_time:.2f}s",
        f"{combined_text}\n\nWER: {combined_wer:.2%} | Time: {combined_time:.2f}s",
        f"{oracle_text}\n\nWER: {oracle_wer:.2%} | Time: {oracle_time:.2f}s",
    )


# ── Gradio callback: batch ──

def transcribe_all(custom_kws_text=""):
    random.seed(42)

    # Parse custom keyword list if provided
    custom_keywords = None
    if custom_kws_text and custom_kws_text.strip():
        custom_keywords = [kw.strip().lower() for kw in custom_kws_text.replace("\n", ",").split(",") if kw.strip()]

    gt = get_current_gt()
    audio_dir = get_current_audio_dir()
    rows = []
    total_baseline = 0
    total_combined = 0

    for filename in sorted(gt.keys()):
        audio_path = str(audio_dir / filename)
        reference_text = gt[filename]["text"]
        precomputed_kws = gt[filename].get("detected_keywords")

        mel = prepare_mel(audio_path)

        # Baseline
        baseline_text = transcribe_with_keywords(mel, [])
        baseline_wer = compute_wer(reference_text, baseline_text)

        if custom_keywords is not None:
            # Use custom keyword list + live AdaKWS detection
            mel_kws = prepare_mel_adakws(audio_path)
            detected = run_adakws_detect(mel_kws, custom_keywords, threshold=0.3)
            seen = set()
            detected_keywords = []
            for kw, _ in detected:
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    detected_keywords.append(kw)
        elif precomputed_kws is not None:
            # Use pre-computed keywords from ground_truth.json
            detected_keywords = precomputed_kws
        else:
            # Live detection: sample keywords + run AdaKWS
            mel_kws = prepare_mel_adakws(audio_path)
            all_keywords, positive_set = sample_keywords(reference_text)
            detected = run_adakws_detect(mel_kws, all_keywords, threshold=0.3)
            seen = set()
            detected_keywords = []
            for kw, _ in detected:
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    detected_keywords.append(kw)

        # Combined
        combined_text = transcribe_with_keywords(mel, detected_keywords)
        combined_wer = compute_wer(reference_text, combined_text)

        total_baseline += baseline_wer
        total_combined += combined_wer

        det_str = ", ".join(detected_keywords) if detected_keywords else "-"

        rows.append([
            filename,
            reference_text,
            baseline_text,
            combined_text,
            f"{baseline_wer:.2%}",
            f"{combined_wer:.2%}",
            det_str,
        ])

    n = len(rows)
    rows.append([
        "AVERAGE", "", "", "",
        f"{total_baseline/n:.2%}" if n else "-",
        f"{total_combined/n:.2%}" if n else "-",
        "",
    ])

    # Save CSV for download
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Sample", "Reference", "Baseline Transcription", "Combined Transcription", "Baseline WER", "Combined WER", "Detected Keywords"])
    for row in rows:
        writer.writerow(row)
    csv_path = str(DEMO_DIR / "batch_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())

    return rows, csv_path


# ── Gradio callback: custom keyword list ──

def load_keywords_file(file):
    """Load keywords from uploaded txt file."""
    if file is None:
        return gr.update()
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
    # Support comma-separated or one-per-line
    keywords = [kw.strip().lower() for kw in content.replace("\n", ",").split(",") if kw.strip()]
    return ", ".join(keywords)


def transcribe_with_custom_kws(audio_file, custom_keywords_text):
    if not audio_file:
        return ("",) * 5

    # Parse keywords from text input
    keywords = [kw.strip().lower() for kw in custom_keywords_text.replace("\n", ",").split(",") if kw.strip()]
    if not keywords:
        return ("No keywords provided",) * 5

    mel = prepare_mel(audio_file)
    mel_kws = prepare_mel_adakws(audio_file)

    # Baseline
    t0 = time.time()
    baseline_text = transcribe_with_keywords(mel, [])
    baseline_time = time.time() - t0

    # AdaKWS detection from custom keyword list
    t0 = time.time()
    detected = run_adakws_detect(mel_kws, keywords, threshold=0.3)
    kws_time = time.time() - t0
    seen = set()
    detected_keywords = []
    for kw, _ in detected:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            detected_keywords.append(kw)
    detected_str = ", ".join([f"{kw} ({prob:.0%})" for kw, prob in detected]) if detected else "None"

    # Combined
    t0 = time.time()
    combined_text = transcribe_with_keywords(mel, detected_keywords)
    combined_time = time.time() - t0

    candidates_str = ", ".join(keywords)

    return (
        candidates_str,
        detected_str,
        f"Detection time: {kws_time:.2f}s",
        f"{baseline_text}\n\nTime: {baseline_time:.2f}s",
        f"{combined_text}\n\nTime: {combined_time:.2f}s",
    )


# ── Build Gradio UI ──

def build_ui():
    gt = models[f"gt_{models['current_dataset']}"]
    sample_names = sorted(gt.keys())
    dataset_names = list(DATASETS.keys())

    demo = gr.Blocks(title="KG-Whisper ASRU² Demo")

    with demo:
        gr.Markdown(
            "# KG-Whisper's ASRU² Demo\n"
            "Pipeline: Baseline (no kws) → AdaKWS keyword detection → Combined (with kws) → Oracle (perfect kws)"
        )

        with gr.Tab("Single Sample"):
            with gr.Row():
                dataset_dropdown = gr.Dropdown(
                    choices=dataset_names,
                    value=models["current_dataset"],
                    label="Dataset",
                    scale=1,
                )
                sample_dropdown = gr.Dropdown(
                    choices=sample_names,
                    value=sample_names[0] if sample_names else None,
                    label="Audio Sample",
                    scale=2,
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary", scale=1)

            dataset_dropdown.change(
                fn=switch_dataset,
                inputs=[dataset_dropdown],
                outputs=[sample_dropdown],
            )

            audio_player = gr.Audio(label="Audio", type="filepath", interactive=False)

            gr.Markdown("---")

            gr.Markdown("### Ground Truth")
            reference_output = gr.Textbox(label="Reference", interactive=False)

            gr.Markdown("---")

            gr.Markdown("### Step 1: Candidate Keywords (3 positive + 17 negative)")
            gr.Markdown("*Bold = positive (from transcript), plain = negative (generated)*")
            candidates_output = gr.Markdown()

            gr.Markdown("### Step 2: AdaKWS Detection")
            detected_output = gr.Textbox(label="Detected Keywords", interactive=False)
            kws_metrics_output = gr.Textbox(label="KWS Metrics", interactive=False, lines=2)

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Baseline (no keywords)")
                    baseline_output = gr.Textbox(label="Transcription", interactive=False, lines=3)
                with gr.Column():
                    gr.Markdown("### Combined (AdaKWS keywords)")
                    combined_output = gr.Textbox(label="Transcription", interactive=False, lines=3)
                with gr.Column():
                    gr.Markdown("### Oracle (perfect keywords)")
                    oracle_output = gr.Textbox(label="Transcription", interactive=False, lines=3)

            transcribe_btn.click(
                fn=transcribe_sample,
                inputs=[sample_dropdown],
                outputs=[
                    audio_player,
                    reference_output,
                    candidates_output,
                    detected_output,
                    kws_metrics_output,
                    baseline_output,
                    combined_output,
                    oracle_output,
                ],
            )

        with gr.Tab("Custom Keywords"):
            gr.Markdown(
                "### Real-world Mode: Custom Keyword List\n"
                "Upload your own audio and provide a domain-specific keyword list (10-50 recommended).\n"
                "AdaKWS detects which keywords are present, then KG-Whisper-PT uses them to guide transcription."
            )
            with gr.Row():
                custom_audio = gr.Audio(label="Upload Audio", type="filepath", scale=2)
                with gr.Column(scale=2):
                    custom_keywords_input = gr.Textbox(
                        label="Keyword List (comma-separated or one per line)",
                        placeholder="e.g. myocardial, hemoglobin, biopsy, catheter, ...",
                        lines=3,
                        value=", ".join(models.get("medical_keywords", [])),
                    )
                    keywords_file = gr.File(label="Or upload keywords .txt file", file_types=[".txt"])
            keywords_file.change(
                fn=load_keywords_file,
                inputs=[keywords_file],
                outputs=[custom_keywords_input],
            )
            custom_btn = gr.Button("Transcribe with Custom Keywords", variant="primary")

            gr.Markdown("---")
            custom_candidates = gr.Textbox(label="Candidate Keywords", interactive=False)
            custom_detected = gr.Textbox(label="Detected Keywords (AdaKWS)", interactive=False)
            custom_kws_info = gr.Textbox(label="Detection Info", interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Baseline (no keywords)")
                    custom_baseline = gr.Textbox(label="Transcription", interactive=False, lines=3)
                with gr.Column():
                    gr.Markdown("### Combined (with detected keywords)")
                    custom_combined = gr.Textbox(label="Transcription", interactive=False, lines=3)

            custom_btn.click(
                fn=transcribe_with_custom_kws,
                inputs=[custom_audio, custom_keywords_input],
                outputs=[
                    custom_candidates,
                    custom_detected,
                    custom_kws_info,
                    custom_baseline,
                    custom_combined,
                ],
            )

        with gr.Tab("Batch Evaluation"):
            with gr.Row():
                batch_kws_input = gr.Textbox(
                    label="Custom Keyword List (optional, comma-separated. Leave empty to use precomputed keywords)",
                    placeholder="e.g. pain, heart, cough, dizzy, ...",
                    lines=2,
                )
                batch_kws_file = gr.File(label="Or upload .txt", file_types=[".txt"])
            batch_kws_file.change(
                fn=load_keywords_file,
                inputs=[batch_kws_file],
                outputs=[batch_kws_input],
            )
            run_all_btn = gr.Button("Run All Samples", variant="primary")
            results_table = gr.Dataframe(
                headers=["Sample", "Reference", "Baseline Transcription", "Combined Transcription", "Baseline WER", "Combined WER", "Detected Keywords"],
                label="Results",
                wrap=True,
                max_height=800,
            )
            download_file = gr.File(label="Download Results (CSV)", visible=True)
            run_all_btn.click(
                fn=transcribe_all,
                inputs=[batch_kws_input],
                outputs=[results_table, download_file],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="KG-Whisper Demo (Gradio)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--adakws_checkpoint", type=str,
        default="kg_whisper/outputs/all_language_train/adakws_checkpoint_78000.pt",
    )
    parser.add_argument(
        "--whisper_pt_checkpoint", type=str,
        default="kg_whisper/outputs/checkpoint_final.pt",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    init_models(device, args.adakws_checkpoint, args.whisper_pt_checkpoint)

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
