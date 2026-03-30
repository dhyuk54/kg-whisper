"""Combined AdaKWS + KG-Whisper-PT evaluation on Voxpopuli test set.

Follows the paper's evaluation protocol:
- For each audio sample, query AdaKWS with 20 keywords (3 positive + 17 negative)
- Pass detected keywords to KG-Whisper-PT for transcription
- Compute WER and compare with Oracle (perfect keywords) and baseline (no keywords)

Usage:
    python -m kg_whisper.eval_combined \
        --pt_checkpoint kg_whisper/outputs/checkpoint_final.pt \
        --adakws_checkpoint kg_whisper/outputs/adakws/adakws_checkpoint_final.pt \
        --device cuda
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path
from typing import List, Set, Tuple

import jiwer
import numpy as np
import torch

import whisper
import whisper.audio
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

from .model import KGWhisperPT
from .adakws_model import AdaKWS
from .adakws_data import CharVocab, NegativeSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_pt_model(
    checkpoint_path: Path, device: str
) -> Tuple[KGWhisperPT, object, dict]:
    """Load KG-Whisper-PT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]

    whisper_model = whisper.load_model(cfg["whisper_model"], device=device)
    tokenizer = get_tokenizer(
        whisper_model.is_multilingual,
        num_languages=whisper_model.num_languages,
        language=cfg.get("language", "en"),
        task="transcribe",
    )

    model = KGWhisperPT(
        whisper_model=whisper_model,
        prefix_length=cfg["prefix_length"],
        eot_id=tokenizer.eot,
    ).to(device)
    model.prefix.data = ckpt["prefix"].to(device)
    model.eval()

    logger.info(
        "PT: model=%s, prefix=%d, language=%s, step=%s",
        cfg["whisper_model"], cfg["prefix_length"],
        cfg.get("language", "en"), cfg.get("step", "?"),
    )
    return model, tokenizer, cfg


def load_adakws_model(
    checkpoint_path: Path, device: str
) -> Tuple[AdaKWS, CharVocab]:
    """Load AdaKWS model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    meta = ckpt["config"]

    whisper_model = whisper.load_model(meta["whisper_model"], device=device)

    # Match CharVocab to checkpoint embedding size
    ckpt_vocab_size = ckpt["text_encoder"]["embedding.weight"].shape[0]
    char_vocab = CharVocab()
    if char_vocab.vocab_size != ckpt_vocab_size:
        MULTILANG_CHARS = list(
            "'1abcdefghijklmnopqrstuvwxyz"
            "ßàáâäçèéêëìíîïñòóôõöùúûüý"
            "ăąćčďđėęěįĺľłńňőœŕřśšťūůűųźżžșț"
        )
        char_vocab = CharVocab(chars=MULTILANG_CHARS)
        logger.info("Using multilingual CharVocab: %d", char_vocab.vocab_size)

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
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()

    logger.info("AdaKWS: model=%s, step=%s", meta["whisper_model"], meta.get("step", "?"))
    return model, char_vocab


def sample_keywords(
    text: str,
    neg_sampler: NegativeSampler,
    n_positive: int = 3,
    n_negative: int = 17,
) -> Tuple[List[str], Set[str]]:
    """Sample keywords for evaluation following paper protocol.

    Args:
        text: Reference transcript.
        neg_sampler: Negative keyword sampler.
        n_positive: Number of positive keywords (paper: 3).
        n_negative: Number of negative keywords (paper: 17).

    Returns:
        (keywords_list, positive_set) tuple.
    """
    words = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.strip().split()]
    words = [w for w in words if w]
    if not words:
        return [], set()

    positive_words_set = set(w.lower() for w in words)

    # Sample positive keywords (from transcript)
    unique_words = list(set(words))
    n_pos = min(n_positive, len(unique_words))
    positives = random.sample(unique_words, n_pos)

    # Sample negative keywords
    negatives = []
    for _ in range(n_negative):
        pos_kw = random.choice(words)
        neg = neg_sampler.sample(positive_words_set, pos_kw)
        negatives.append(neg)

    all_keywords = positives + negatives
    random.shuffle(all_keywords)

    return all_keywords, set(w.lower() for w in positives)


def detect_with_adakws(
    model: AdaKWS,
    char_vocab: CharVocab,
    mel: torch.Tensor,
    keywords: List[str],
    threshold: float = 0.5,
) -> List[str]:
    """Run AdaKWS detection on a list of keywords.

    Args:
        model: AdaKWS model.
        char_vocab: Character vocabulary.
        mel: Mel spectrogram [1, n_mels, 3000].
        keywords: List of candidate keywords.
        threshold: Detection threshold.

    Returns:
        List of detected keywords.
    """
    detected = model.predict(mel, keywords, char_vocab, threshold=threshold)
    return [kw for kw, _ in detected]


def transcribe_with_pt(
    model: KGWhisperPT,
    tokenizer,
    mel: torch.Tensor,
    keywords: List[str],
    max_kws_tokens: int,
    device: str,
) -> str:
    """Transcribe audio with KG-Whisper-PT using keyword guidance.

    Args:
        model: KG-Whisper-PT model.
        tokenizer: Whisper tokenizer.
        mel: Mel spectrogram [1, n_mels, 3000].
        keywords: List of keywords to use.
        max_kws_tokens: Max BPE tokens for keywords.
        device: Device string.

    Returns:
        Transcription text.
    """
    kws_str = " | ".join(keywords) if keywords else ""
    kws_token_ids = tokenizer.encoding.encode(kws_str) if kws_str else []
    kws_token_ids = kws_token_ids[:max_kws_tokens]
    pad_len = max_kws_tokens - len(kws_token_ids)
    kws_padded = kws_token_ids + [tokenizer.eot] * pad_len

    prompt_tokens = (
        [tokenizer.sot_prev]
        + kws_padded
        + list(tokenizer.sot_sequence_including_notimestamps)
    )
    prompt_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    gen_ids = model.generate(mel, prompt_tokens, eot_id=tokenizer.eot)
    return tokenizer.decode(gen_ids).strip()


def evaluate_combined(
    pt_model: KGWhisperPT,
    tokenizer,
    adakws_model: AdaKWS,
    char_vocab: CharVocab,
    device: str,
    mode: str = "all",
    cache_dir: str | None = None,
    language: str = "en",
    max_kws_tokens: int = 20,
    threshold: float = 0.5,
    show_samples: int = 10,
) -> None:
    """Run combined evaluation on Voxpopuli test set.

    Args:
        mode: Evaluation mode - "combined", "oracle", "baseline", "whisper", or "all".
    """
    from datasets import load_dataset

    logger.info("Loading Voxpopuli %s test split...", language)
    ds = load_dataset("facebook/voxpopuli", language, split="test", cache_dir=cache_dir)
    total = len(ds)
    logger.info("Test set: %d samples", total)

    n_mels = pt_model.whisper.dims.n_mels

    # Build word pool for negative sampling
    text_col = "normalized_text" if "normalized_text" in ds.column_names else "raw_text"
    word_pool = []
    for text in ds[text_col]:
        if text:
            cleaned = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.strip().split()]
            word_pool.extend([w for w in cleaned if w])
    word_pool = list(set(word_pool))
    neg_sampler = NegativeSampler(word_pool)
    logger.info("Word pool: %d unique words", len(word_pool))

    normalizer = EnglishTextNormalizer()

    refs = []
    hyps_combined = []   # AdaKWS + PT
    hyps_oracle = []     # Oracle + PT
    hyps_baseline = []   # PT + no keywords
    hyps_whisper = []    # Pure Whisper (no PT)

    start_time = time.time()

    for idx in range(total):
        item = ds[idx]
        ref_text = (item.get("normalized_text") or item.get("raw_text", "")).strip()
        if not ref_text:
            continue

        # Audio -> mel
        audio_array = np.array(item["audio"]["array"], dtype=np.float32)
        audio = torch.from_numpy(audio_array)
        audio = whisper.audio.pad_or_trim(audio)
        mel = whisper.audio.log_mel_spectrogram(
            audio, n_mels=n_mels
        ).unsqueeze(0).to(device)

        # Sample 20 keywords (3 positive + 17 negative)
        all_keywords, positive_set = sample_keywords(ref_text, neg_sampler)

        run_combined = mode in ("all", "combined")
        run_oracle = mode in ("all", "oracle")
        run_baseline = mode in ("all", "baseline")
        run_whisper = mode in ("all", "whisper")

        # 1. AdaKWS detection -> PT transcription
        hyp_combined = ""
        if run_combined:
            detected = detect_with_adakws(adakws_model, char_vocab, mel, all_keywords, threshold)
            hyp_combined = transcribe_with_pt(
                pt_model, tokenizer, mel, detected, max_kws_tokens, device
            )
        else:
            detected = []

        # 2. Oracle: use only positive keywords -> PT transcription
        hyp_oracle = ""
        if run_oracle:
            oracle_keywords = [kw for kw in all_keywords if kw.lower() in positive_set]
            hyp_oracle = transcribe_with_pt(
                pt_model, tokenizer, mel, oracle_keywords, max_kws_tokens, device
            )
        else:
            oracle_keywords = []

        # 3. PT + no keywords
        hyp_baseline = ""
        if run_baseline:
            hyp_baseline = transcribe_with_pt(
                pt_model, tokenizer, mel, [], max_kws_tokens, device
            )

        # 4. Pure Whisper (no PT, no keywords)
        hyp_whisper = ""
        if run_whisper:
            with torch.no_grad():
                result = whisper.decode(
                    pt_model.whisper, mel.squeeze(0),
                    whisper.DecodingOptions(language=language, without_timestamps=True),
                )
            hyp_whisper = result.text.strip()  # type: ignore[union-attr]

        refs.append(normalizer(ref_text))
        hyps_combined.append(normalizer(hyp_combined))
        hyps_oracle.append(normalizer(hyp_oracle))
        hyps_baseline.append(normalizer(hyp_baseline))
        hyps_whisper.append(normalizer(hyp_whisper))

        done = idx + 1

        # Show first N samples
        if done <= show_samples:
            logger.info("[%d] REF:      %s", done, ref_text)
            logger.info("[%d] COMBINED: %s (kws: %s)", done, hyp_combined, detected)
            logger.info("[%d] ORACLE:   %s (kws: %s)", done, hyp_oracle, oracle_keywords)
            logger.info("[%d] BASELINE: %s", done, hyp_baseline)
            logger.info("[%d] WHISPER:  %s", done, hyp_whisper)
            logger.info("---")

        # Progress
        if done % 100 == 0 or done == total:
            elapsed = time.time() - start_time
            speed = elapsed / done
            eta = speed * (total - done)
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.1fs/sample | ETA %.0fs",
                done, total, 100.0 * done / total, speed, eta,
            )

    elapsed = time.time() - start_time

    # Compute WER
    logger.info("=" * 60)
    logger.info("EVALUATION RESULT (Voxpopuli test, mode=%s)", mode)
    logger.info("  Samples: %d", len(refs))

    if mode in ("all", "combined"):
        wer_combined = jiwer.wer(refs, hyps_combined) * 100
        logger.info("  WER (AdaKWS + PT):  %.2f%%", wer_combined)
    if mode in ("all", "oracle"):
        wer_oracle = jiwer.wer(refs, hyps_oracle) * 100
        logger.info("  WER (Oracle + PT):  %.2f%%", wer_oracle)
    if mode in ("all", "baseline"):
        wer_baseline = jiwer.wer(refs, hyps_baseline) * 100
        logger.info("  WER (PT no kws):    %.2f%%", wer_baseline)
    if mode in ("all", "whisper"):
        wer_whisper = jiwer.wer(refs, hyps_whisper) * 100
        logger.info("  WER (Pure Whisper): %.2f%%", wer_whisper)

    logger.info("  Time: %.1fs (%.1fs/sample)", elapsed, elapsed / len(refs))
    logger.info("=" * 60)
    logger.info("")
    logger.info("Paper Table 1 reference (large-v2, 30K steps):")
    logger.info("  KG-Whisper-PT (COMBINED):  11.10%%")
    logger.info("  KG-Whisper-PT – Oracle:    10.86%%")
    logger.info("  Whisper PT (BASELINE):     12.83%%")
    logger.info("  Whisper (pure):            13.39%%")


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
# charvocab是什么作用?
# NS是什么作用?

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined AdaKWS + KG-Whisper-PT Evaluation"
    )
    parser.add_argument("--pt_checkpoint", required=True, help="PT checkpoint path")
    parser.add_argument("--adakws_checkpoint", required=True, help="AdaKWS checkpoint path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_kws_tokens", type=int, default=20)
    parser.add_argument("--show_samples", type=int, default=10)
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "combined", "oracle", "baseline", "whisper"],
        help="Evaluation mode: run only one method or all",
    )
    args = parser.parse_args()

    device = args.device or _detect_device()

    # Load both models
    pt_model, tokenizer, pt_cfg = load_pt_model(Path(args.pt_checkpoint), device)
    adakws_model, char_vocab = load_adakws_model(Path(args.adakws_checkpoint), device)

    evaluate_combined(
        pt_model=pt_model,
        tokenizer=tokenizer,
        adakws_model=adakws_model,
        char_vocab=char_vocab,
        device=device,
        mode=args.mode,
        cache_dir=args.cache_dir,
        language=pt_cfg.get("language", "en"),
        max_kws_tokens=args.max_kws_tokens,
        threshold=args.threshold,
        show_samples=args.show_samples,
    )


if __name__ == "__main__":
    main()
