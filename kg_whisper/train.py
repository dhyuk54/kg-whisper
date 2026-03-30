"""Training script for KG-Whisper-PT.

Usage:
    python -m kg_whisper.train --whisper_model small --max_steps 1000 --max_train_samples 500
"""

import argparse
import logging
import math
import time
from typing import Optional

import torch

import whisper
from whisper.tokenizer import get_tokenizer

from .config import KGWhisperConfig
from .data import create_dataloader
from .evaluate import evaluate_teacher_forced, evaluate_wer
from .model import KGWhisperPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float
) -> float:
    """Linear warmup + cosine decay schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: KGWhisperConfig) -> None:
    """Main training loop."""
    device = config.device

    # ── Load Whisper model ──
    logger.info("Loading Whisper '%s' ...", config.whisper_model)
    whisper_model = whisper.load_model(config.whisper_model, device=device)
    config.n_mels = whisper_model.dims.n_mels
    logger.info(
        "Whisper dims: n_text_state=%d, n_text_ctx=%d, n_mels=%d",
        whisper_model.dims.n_text_state,
        whisper_model.dims.n_text_ctx,
        whisper_model.dims.n_mels,
    )

    # ── Tokenizer ──
    tokenizer = get_tokenizer(
        whisper_model.is_multilingual,
        num_languages=whisper_model.num_languages,
        language=config.language,
        task="transcribe",
    )
    logger.info("EOT=%d, SOP=%d, SOT sequence=%s",
                tokenizer.eot, tokenizer.sot_prev, tokenizer.sot_sequence)

    # ── Model ──
    model = KGWhisperPT(
        whisper_model=whisper_model,
        prefix_length=config.prefix_length,
        eot_id=tokenizer.eot,
    ).to(device)

    # ── Data ──
    logger.info("Creating data loaders ...")
    train_loader = create_dataloader("train", config, tokenizer, shuffle=True, whisper_model=whisper_model)
    eval_loader = create_dataloader("validation", config, tokenizer, shuffle=False, whisper_model=whisper_model)
    logger.info(
        "Train: %d samples, Eval: %d samples",
        len(train_loader.dataset),
        len(eval_loader.dataset),
    )

    # ── Optimizer — only the prefix is trainable ──
    optimizer = torch.optim.AdamW(
        [model.prefix],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ── Resume from checkpoint ──
    global_step = 0
    resume_step = 0
    if config.resume_checkpoint:
        ckpt = torch.load(config.resume_checkpoint, map_location=device, weights_only=True)
        model.prefix.data.copy_(ckpt["prefix"].to(device))
        # Don't load optimizer state — reset for new data/language mix
        resume_step = ckpt.get("step", 0)
        global_step = resume_step
        logger.info("Resumed prefix from checkpoint: step=%d (optimizer reset, warmup restarted)", global_step)

    # ── Training ──
    config.output_dir.mkdir(parents=True, exist_ok=True)
    accum_loss = 0.0
    accum_count = 0
    start_time = time.time()

    logger.info("Training for %d steps (batch=%d, accum=%d, eff_batch=%d)",
                config.max_steps, config.batch_size,
                config.gradient_accumulation_steps,
                config.batch_size * config.gradient_accumulation_steps)

    model.train()

    while global_step < config.max_steps:
        for batch in train_loader:
            if global_step >= config.max_steps:
                break

            mel = batch["mel"].to(device)
            prompt_tokens = batch["prompt_tokens"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_lengths = batch["text_lengths"].to(device)

            loss, _ = model(mel, prompt_tokens, text_tokens, text_lengths)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            accum_loss += loss.item() * config.gradient_accumulation_steps
            accum_count += 1

            if accum_count % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([model.prefix], config.max_grad_norm)

                lr = get_lr(
                    global_step - resume_step,
                    config.warmup_steps,
                    config.max_steps - resume_step,
                    config.learning_rate,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Log ──
                if global_step % config.log_steps == 0:
                    avg_loss = accum_loss / accum_count
                    elapsed = time.time() - start_time
                    logger.info(
                        "step %d/%d | loss %.4f | lr %.2e | %.0fs",
                        global_step,
                        config.max_steps,
                        avg_loss,
                        lr,
                        elapsed,
                    )
                    accum_loss = 0.0
                    accum_count = 0

                # ── Eval ──
                if global_step % config.eval_steps == 0:
                    eval_loss = evaluate_teacher_forced(model, eval_loader, device)
                    logger.info("step %d | eval_loss %.4f", global_step, eval_loss)

                    wer_result = evaluate_wer(
                        model, eval_loader, tokenizer, device,
                        max_samples=config.max_generate_samples,
                    )
                    if wer_result["wer"] >= 0:
                        logger.info("step %d | WER %.2f%%",
                                    global_step, wer_result["wer"] * 100)
                    model.train()

                # ── Save ──
                if global_step % config.save_steps == 0:
                    _save_checkpoint(model, optimizer, global_step, config)

    # ── Final save ──
    _save_checkpoint(model, optimizer, global_step, config, tag="final")
    logger.info("Training complete at step %d.", global_step)


def _save_checkpoint(
    model: KGWhisperPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: KGWhisperConfig,
    tag: Optional[str] = None,
) -> None:
    """Save prefix weights + optimizer state."""
    name = f"checkpoint_{tag or step}.pt"
    path = config.output_dir / name
    torch.save(
        {
            "step": step,
            "prefix": model.prefix.data.cpu(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "whisper_model": config.whisper_model,
                "prefix_length": config.prefix_length,
                "language": config.language,
            },
        },
        path,
    )
    logger.info("Saved %s", path)


def _detect_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train KG-Whisper-PT")
    parser.add_argument("--whisper_model", default="small")
    parser.add_argument("--prefix_length", type=int, default=12)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Multiple languages (overrides --language), e.g. --languages en de fr")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--cache_dir", default=None, help="HuggingFace datasets cache directory")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=100)
    args = parser.parse_args()

    config = KGWhisperConfig(
        whisper_model=args.whisper_model,
        prefix_length=args.prefix_length,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device or _detect_device(),
        language=args.language,
        languages=args.languages,
        resume_checkpoint=args.resume,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )

    logger.info("Config: %s", config)
    train(config)


if __name__ == "__main__":
    main()
