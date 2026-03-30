"""Evaluation script for AdaKWS on VoxPopuli test set.

Computes F1, AUC, and EER following the paper's evaluation protocol.

Usage:
    python -m kg_whisper.adakws_eval --checkpoint kg_whisper/outputs/adakws/adakws_checkpoint_final.pt --device cuda --cache_dir "F:/"
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

import whisper
import whisper.audio

from .adakws_config import AdaKWSConfig
from .adakws_data import CharVocab, NegativeSampler
from .adakws_model import AdaKWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_adakws_model(checkpoint_path: Path, device: str):
    """Load AdaKWS model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = ckpt["config"]

    logger.info("Checkpoint: model=%s, language=%s", meta["whisper_model"], meta["language"])

    whisper_model = whisper.load_model(meta["whisper_model"], device=device)

    # Infer vocab_size from checkpoint embedding weight shape
    ckpt_vocab_size = ckpt["text_encoder"]["embedding.weight"].shape[0]
    char_vocab = CharVocab()
    if char_vocab.vocab_size != ckpt_vocab_size:
        logger.info(
            "CharVocab size mismatch: default=%d, checkpoint=%d. "
            "Using multilingual char set.",
            char_vocab.vocab_size, ckpt_vocab_size,
        )
        # Exact char set from multi-language VoxPopuli training (16 languages)
        # Sorted set of all characters observed in training data
        MULTILANG_CHARS = list(
            "'1abcdefghijklmnopqrstuvwxyz"
            "ßàáâäçèéêëìíîïñòóôõöùúûüý"
            "ăąćčďđėęěįĺľłńňőœŕřśšťūůűųźżžșț"
        )
        char_vocab = CharVocab(chars=MULTILANG_CHARS)
        logger.info("Multilingual CharVocab: vocab_size=%d", char_vocab.vocab_size)

    model = AdaKWS(
        whisper_model=whisper_model,
        char_vocab_size=char_vocab.vocab_size,
        char_embed_dim=meta["char_embed_dim"],
        lstm_hidden_dim=meta["lstm_hidden_dim"],
        lstm_num_layers=meta["lstm_num_layers"],
        n_adaptive_modules=meta["n_adaptive_modules"],
    ).to(device)

    model.text_encoder.load_state_dict(ckpt["text_encoder"])
    model.adaptive_modules.load_state_dict(ckpt["adaptive_modules"])
    try:
        model.classifier.load_state_dict(ckpt["classifier"])
    except RuntimeError:
        logger.warning("Classifier shape mismatch — using random init (old checkpoint with Linear classifier)")
    model.eval()

    return model, char_vocab, meta


def evaluate_adakws(
    model: AdaKWS,
    char_vocab: CharVocab,
    config: AdaKWSConfig,
    device: str,
    show_samples: int = 10,
) -> Dict[str, float]:
    """Evaluate AdaKWS on VoxPopuli test set.

    Evaluation protocol from paper Section 4:
    - Positive keywords: randomly sampled from transcript
    - Negatives: equal mix of random, concatenation, swap
    - Metrics: F1, AUC, EER
    """
    from datasets import load_dataset

    logger.info("Loading VoxPopuli %s test split...", config.language)
    ds = load_dataset(
        config.dataset_name, config.language,
        split="test", cache_dir=config.cache_dir,
    )
    total = len(ds)
    logger.info("Test set: %d samples", total)

    # Build word pool for negative sampling
    text_col = "normalized_text" if "normalized_text" in ds.column_names else "raw_text"
    word_pool = []
    texts = ds[text_col]
    for text in texts:
        if text:
            cleaned = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.strip().split()]
            word_pool.extend([w for w in cleaned if w])
    word_pool = list(set(word_pool))
    neg_sampler = NegativeSampler(word_pool)

    all_labels = []
    all_probs = []
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for idx in range(total):
            item = ds[idx]
            text = (item.get("normalized_text") or item.get("raw_text", "")).strip()
            words = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.split()]
            words = [w for w in words if w]
            if not words:
                continue

            # Audio → mel
            audio_array = np.array(item["audio"]["array"], dtype=np.float32)
            audio = torch.from_numpy(audio_array)
            audio = whisper.audio.pad_or_trim(audio)
            mel = whisper.audio.log_mel_spectrogram(
                audio, n_mels=config.n_mels
            ).unsqueeze(0).to(device)

            positive_words = set(w.lower() for w in words)

            # Test with positive keyword
            pos_kw = random.choice(words)
            pos_ids = char_vocab.encode(pos_kw.lower())
            pos_char = torch.tensor([pos_ids], dtype=torch.long, device=device)
            pos_len = torch.tensor([len(pos_ids)], dtype=torch.long)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pos_logit = model(mel, pos_char, pos_len)
            pos_prob = torch.sigmoid(pos_logit).item()
            all_labels.append(1)
            all_probs.append(pos_prob)

            # Test with negative keyword (equal mix of 3 strategies: random, concat, swap)
            strategy = random.randint(0, 2)  # 0=random, 1=concat, 2=char_sub
            if strategy == 0:
                neg_kw = neg_sampler.random_negative(positive_words)
            elif strategy == 1:
                neg_kw = neg_sampler.concatenation(pos_kw, positive_words)
            else:
                neg_kw = neg_sampler.char_substitution(pos_kw)

            neg_ids = char_vocab.encode(neg_kw.lower())
            neg_char = torch.tensor([neg_ids], dtype=torch.long, device=device)
            neg_len = torch.tensor([len(neg_ids)], dtype=torch.long)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                neg_logit = model(mel, neg_char, neg_len)
            neg_prob = torch.sigmoid(neg_logit).item()
            all_labels.append(0)
            all_probs.append(neg_prob)

            # Show samples
            done = idx + 1
            if done <= show_samples:
                pos_det = "detected" if pos_prob > 0.5 else "missed"
                neg_det = "rejected" if neg_prob <= 0.5 else "false alarm"
                logger.info(
                    "[%d] POS '%s' → %.3f (%s) | NEG '%s' → %.3f (%s)",
                    done, pos_kw, pos_prob, pos_det, neg_kw, neg_prob, neg_det,
                )

            # Progress
            if done % 200 == 0 or done == total:
                elapsed = time.time() - start_time
                speed = elapsed / done
                eta = speed * (total - done)
                logger.info(
                    "Progress: %d/%d (%.1f%%) | %.2fs/sample | ETA %.0fs",
                    done, total, 100.0 * done / total, speed, eta,
                )

    elapsed = time.time() - start_time

    # Compute metrics
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    preds = (probs > 0.5).astype(float)

    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    # EER
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    logger.info("=" * 60)
    logger.info("AdaKWS EVALUATION RESULT")
    logger.info("  Samples: %d (pairs: %d)", total, len(labels))
    logger.info("  F1:  %.2f%%", f1 * 100)
    logger.info("  AUC: %.2f%%", auc * 100)
    logger.info("  EER: %.2f%%", eer * 100)
    logger.info("  Time: %.1fs", elapsed)
    logger.info("=" * 60)

    return {"f1": f1, "auc": auc, "eer": eer}


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Evaluate AdaKWS")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--show_samples", type=int, default=10)
    parser.add_argument("--language", type=str, default=None,
                        help="Language to evaluate (default: from checkpoint). Use 'all16' to run all 16 languages.")
    args = parser.parse_args()

    device = args.device or _detect_device()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return

    model, char_vocab, meta = load_adakws_model(checkpoint_path, device)
    n_mels = model.encoder.conv1.weight.shape[1]

    if args.language == "all16":
        all_langs = ["cs","de","en","es","et","fi","fr","hr","hu","it","lt","nl","pl","ro","sk","sl"]
        all_results = {}
        for lang in all_langs:
            logger.info("\n>>> Evaluating language: %s <<<", lang)
            config = AdaKWSConfig(
                whisper_model=meta["whisper_model"],
                language=lang,
                cache_dir=args.cache_dir,
                n_mels=n_mels,
            )
            result = evaluate_adakws(model, char_vocab, config, device, show_samples=0)
            all_results[lang] = result

        # Summary table
        logger.info("\n" + "=" * 60)
        logger.info("ALL 16 LANGUAGES SUMMARY")
        logger.info("%-5s  %6s  %6s  %6s", "Lang", "F1", "AUC", "EER")
        logger.info("-" * 30)
        f1_sum = 0
        for lang in all_langs:
            r = all_results[lang]
            logger.info("%-5s  %5.2f%%  %5.2f%%  %5.2f%%", lang, r["f1"]*100, r["auc"]*100, r["eer"]*100)
            f1_sum += r["f1"]
        avg_f1 = f1_sum / len(all_langs) * 100
        logger.info("-" * 30)
        logger.info("%-5s  %5.2f%%", "AVG", avg_f1)
        logger.info("=" * 60)
    else:
        lang = args.language or meta["language"]
        config = AdaKWSConfig(
            whisper_model=meta["whisper_model"],
            language=lang,
            cache_dir=args.cache_dir,
            n_mels=n_mels,
        )
        evaluate_adakws(model, char_vocab, config, device, show_samples=args.show_samples)


if __name__ == "__main__":
    main()
