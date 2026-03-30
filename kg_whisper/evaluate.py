"""Evaluation for KG-Whisper-PT: teacher-forced loss and autoregressive WER."""

import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader

from whisper.tokenizer import Tokenizer

from .model import KGWhisperPT

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_teacher_forced(
    model: KGWhisperPT,
    eval_loader: DataLoader,
    device: str,
) -> float:
    """Compute average loss with teacher forcing.

    Args:
        model: KG-Whisper-PT model.
        eval_loader: Evaluation DataLoader.
        device: Device string.

    Returns:
        Average cross-entropy loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in eval_loader:
        mel = batch["mel"].to(device)
        prompt_tokens = batch["prompt_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        text_lengths = batch["text_lengths"].to(device)

        loss, _ = model(mel, prompt_tokens, text_tokens, text_lengths)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_wer(
    model: KGWhisperPT,
    eval_loader: DataLoader,
    tokenizer: Tokenizer,
    device: str,
    max_samples: int = 5,
) -> Dict[str, float]:
    """Compute WER with autoregressive greedy decoding.

    Args:
        model: KG-Whisper-PT model.
        eval_loader: Evaluation DataLoader.
        tokenizer: Whisper tokenizer.
        device: Device string.
        max_samples: Max samples to evaluate (autoregressive is slow).

    Returns:
        Dict with "wer" score (0.0-1.0), or -1.0 if jiwer not installed.
    """
    try:
        from jiwer import wer as compute_wer
    except ImportError:
        logger.warning("jiwer not installed — skipping WER evaluation")
        return {"wer": -1.0}

    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    count = 0

    for batch in eval_loader:
        if count >= max_samples:
            break

        mel = batch["mel"].to(device)
        prompt_tokens = batch["prompt_tokens"].to(device)
        text_tokens = batch["text_tokens"]
        text_lengths = batch["text_lengths"]

        for i in range(mel.size(0)):
            if count >= max_samples:
                break

            tl = text_lengths[i].item()
            ref_ids = text_tokens[i, :tl].tolist()
            ref_text = tokenizer.decode(ref_ids).strip()
            if not ref_text:
                continue

            gen_ids = model.generate(
                mel[i : i + 1],
                prompt_tokens[i : i + 1],
                eot_id=tokenizer.eot,
            )
            hyp_text = tokenizer.decode(gen_ids).strip() or " "

            references.append(ref_text)
            hypotheses.append(hyp_text)
            count += 1

            if count <= 3:
                logger.info("[REF] %s", ref_text)
                logger.info("[HYP] %s", hyp_text)
                logger.info("---")

    model.train()

    if not references:
        return {"wer": -1.0}

    wer_score = compute_wer(references, hypotheses)
    logger.info("WER: %.2f%% (%d samples)", wer_score * 100, len(references))
    return {"wer": wer_score}
