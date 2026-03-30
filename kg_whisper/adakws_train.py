"""Training script for AdaKWS.

Usage:
    python -m kg_whisper.adakws_train --whisper_model large-v2 --device cuda --cache_dir "F:/"
"""

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import whisper

from .adakws_config import AdaKWSConfig
from .adakws_data import CharVocab, create_dataloader
from .adakws_model import AdaKWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_lr(
    step: int, warmup_steps: int, total_steps: int, max_lr: float
) -> float:
    """Linear warmup + cosine decay schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: AdaKWSConfig) -> None:
    """Main training loop."""
    device = config.device

    # ── Load Whisper model ──
    logger.info("Loading Whisper '%s' ...", config.whisper_model)
    whisper_model = whisper.load_model(config.whisper_model, device=device)
    config.n_mels = whisper_model.dims.n_mels
    logger.info(
        "Whisper encoder: n_audio_state=%d, n_audio_head=%d, n_audio_layer=%d",
        whisper_model.dims.n_audio_state,
        whisper_model.dims.n_audio_head,
        whisper_model.dims.n_audio_layer,
    )

    # ── Character vocab ──
    if config.languages and len(config.languages) > 1:
        # Multi-language: build CharVocab from training data
        from datasets import load_dataset
        logger.info("Building CharVocab from %d languages...", len(config.languages))
        ds_list = []
        for lang in config.languages:
            ds = load_dataset(config.dataset_name, lang, split="train[:2000]", cache_dir=config.cache_dir)
            ds_list.append(ds)
        text_col = "normalized_text" if "normalized_text" in ds_list[0].column_names else "raw_text"
        char_vocab = CharVocab.from_datasets(ds_list, text_col=text_col, max_samples=2000)
        logger.info("CharVocab built from data: %d chars, vocab_size=%d", char_vocab.vocab_size - 2, char_vocab.vocab_size)
    else:
        char_vocab = CharVocab()
        logger.info("CharVocab default: vocab_size=%d", char_vocab.vocab_size)

    # ── Model ──
    model = AdaKWS(
        whisper_model=whisper_model,
        char_vocab_size=char_vocab.vocab_size,
        char_embed_dim=config.char_embed_dim,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_num_layers=config.lstm_num_layers,
        n_adaptive_modules=config.n_adaptive_modules,
    ).to(device)

    # ── Data ──
    logger.info("Creating data loaders ...")
    train_loader = create_dataloader("train", config, char_vocab, shuffle=True)
    eval_loader = create_dataloader("validation", config, char_vocab, shuffle=False)
    logger.info(
        "Train: %d samples, Eval: %d samples",
        len(train_loader.dataset),
        len(eval_loader.dataset),
    )

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ── Compute total steps ──
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    # ── Resume from checkpoint ──
    start_epoch = 0
    global_step = 0
    if config.resume_checkpoint:
        ckpt = torch.load(config.resume_checkpoint, map_location=device, weights_only=True)
        model.text_encoder.load_state_dict(ckpt["text_encoder"])
        model.adaptive_modules.load_state_dict(ckpt["adaptive_modules"])
        # Try loading classifier; if shape mismatch (e.g. Linear→MLP), skip it
        try:
            model.classifier.load_state_dict(ckpt["classifier"])
            logger.info("Loaded classifier from checkpoint")
        except (RuntimeError, KeyError):
            logger.warning("Classifier shape mismatch — reinitializing classifier (new architecture)")
        # Only restore optimizer if classifier loaded successfully
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (RuntimeError, ValueError):
            logger.warning("Optimizer state mismatch — reinitializing optimizer")
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        logger.info(
            "Resumed from checkpoint: epoch=%d, step=%d",
            start_epoch, global_step,
        )

    logger.info(
        "Training: %d epochs, %d steps/epoch, %d total steps, %d warmup steps",
        config.num_epochs, steps_per_epoch, total_steps, warmup_steps,
    )
    logger.info(
        "Batch: %d × accum %d = effective %d",
        config.batch_size,
        config.gradient_accumulation_steps,
        config.batch_size * config.gradient_accumulation_steps,
    )

    # ── Mixed precision scaler ──
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Training ──
    config.output_dir.mkdir(parents=True, exist_ok=True)
    accum_loss = 0.0
    accum_count = 0
    start_time = time.time()

    model.train()

    for epoch in range(start_epoch, config.num_epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, config.num_epochs)

        for batch in train_loader:
            mel = batch["mel"].to(device)                         # [B, 80, 3000]
            pos_char_ids = batch["pos_char_ids"].to(device)       # [B, L_pos]
            pos_char_lengths = batch["pos_char_lengths"].to(device)
            neg_char_ids = batch["neg_char_ids"].to(device)       # [B, L_neg]
            neg_char_lengths = batch["neg_char_lengths"].to(device)
            B = mel.size(0)

            # ── NK sampling (paper Section 3) ──
            # Encode positive keywords through LSTM → embeddings
            with torch.no_grad():
                emb = model.text_encoder.embedding(pos_char_ids)
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    emb, pos_char_lengths.cpu().clamp(min=1),
                    batch_first=True, enforce_sorted=False,
                )
                _, (h_n, _) = model.text_encoder.lstm(packed)
                # Bidirectional: concatenate forward and backward last layer
                pos_embeddings = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, hidden_dim * 2]

            # Cosine distance matrix within batch
            pos_norm = F.normalize(pos_embeddings, dim=1)      # [B, hidden_dim]
            cos_sim = pos_norm @ pos_norm.T                     # [B, B]
            cos_sim.fill_diagonal_(-2.0)                        # mask self
            nk_indices = cos_sim.argmax(dim=1)                  # [B]

            # NK negative = positive keyword of the nearest sample in batch
            nk_char_ids = pos_char_ids[nk_indices]              # [B, L_pos]
            nk_char_lengths = pos_char_lengths[nk_indices]      # [B]

            # ── Build mixed batch of B samples (paper Algorithm 1) ──
            # Half positive, half negative (mixed in one batch, no doubling)
            is_positive = torch.rand(B, device=device) < 0.5
            is_negative = ~is_positive

            # For negative samples: 25% use NK, 75% use pre-sampled
            use_nk = is_negative & (torch.rand(B, device=device) < 0.25)
            use_presamp = is_negative & ~use_nk

            # Pad all char_ids to same length
            max_len = max(pos_char_ids.size(1), neg_char_ids.size(1), nk_char_ids.size(1))
            pos_padded = F.pad(pos_char_ids, (0, max(0, max_len - pos_char_ids.size(1))))
            neg_padded = F.pad(neg_char_ids, (0, max(0, max_len - neg_char_ids.size(1))))
            nk_padded = F.pad(nk_char_ids, (0, max(0, max_len - nk_char_ids.size(1))))

            # Select char_ids and lengths per sample
            char_ids = pos_padded.clone()    # start with positive
            char_lengths = pos_char_lengths.clone()

            # Override negatives with pre-sampled
            for i in range(B):
                if use_presamp[i]:
                    char_ids[i] = neg_padded[i]
                    char_lengths[i] = neg_char_lengths[i]
                elif use_nk[i]:
                    char_ids[i] = nk_padded[i]
                    char_lengths[i] = nk_char_lengths[i]

            labels = is_positive.float()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(mel, char_ids, char_lengths)
                loss_main = F.binary_cross_entropy_with_logits(logits, labels)

            # ── Cross-audio negative sampling ──
            # Use other samples' positive keywords as negatives for each audio.
            # This teaches the model to reject unrelated words, not just similar-sounding ones.
            cross_indices = torch.randperm(B, device=device)
            # Ensure no self-pairing
            for i in range(B):
                if cross_indices[i] == i:
                    cross_indices[i] = (i + 1) % B

            cross_char_ids = pos_char_ids[cross_indices]
            cross_char_lengths = pos_char_lengths[cross_indices]
            cross_labels = torch.zeros(B, device=device)  # all negative

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                cross_logits = model(mel, cross_char_ids, cross_char_lengths)
                loss_cross = F.binary_cross_entropy_with_logits(cross_logits, cross_labels)

            loss = loss_main + loss_cross

            loss_scaled = loss / config.gradient_accumulation_steps
            scaler.scale(loss_scaled).backward()

            accum_loss += loss.item()
            accum_count += 1

            if accum_count % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)

                lr = get_lr(global_step, warmup_steps, total_steps, config.learning_rate)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # ── Log ──
                if global_step % config.log_steps == 0:
                    avg_loss = accum_loss / accum_count
                    elapsed = time.time() - start_time
                    logger.info(
                        "epoch %d | step %d/%d | loss %.4f | lr %.2e | %.0fs",
                        epoch + 1, global_step, total_steps, avg_loss, lr, elapsed,
                    )
                    accum_loss = 0.0
                    accum_count = 0

                # ── Eval ──
                if global_step % config.eval_steps == 0:
                    eval_loss, eval_acc = evaluate_quick(model, eval_loader, device)
                    logger.info(
                        "step %d | eval_loss %.4f | eval_acc %.2f%%",
                        global_step, eval_loss, eval_acc * 100,
                    )
                    model.train()

                # ── Save ──
                if global_step % config.save_steps == 0:
                    _save_checkpoint(model, optimizer, global_step, epoch, config)

    # ── Final save ──
    _save_checkpoint(model, optimizer, global_step, config.num_epochs, config, tag="final")
    logger.info("Training complete at step %d.", global_step)


@torch.no_grad()
def evaluate_quick(model, eval_loader, device) -> tuple:
    """Quick evaluation: compute loss and accuracy on mixed pos/neg batch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in eval_loader:
        mel = batch["mel"].to(device)
        pos_char_ids = batch["pos_char_ids"].to(device)
        pos_char_lengths = batch["pos_char_lengths"].to(device)
        neg_char_ids = batch["neg_char_ids"].to(device)
        neg_char_lengths = batch["neg_char_lengths"].to(device)
        B = mel.size(0)

        # 50/50 positive/negative
        is_positive = torch.rand(B, device=device) < 0.5
        max_len = max(pos_char_ids.size(1), neg_char_ids.size(1))
        pos_padded = F.pad(pos_char_ids, (0, max(0, max_len - pos_char_ids.size(1))))
        neg_padded = F.pad(neg_char_ids, (0, max(0, max_len - neg_char_ids.size(1))))

        char_ids = torch.where(
            is_positive.unsqueeze(1).expand(-1, max_len),
            pos_padded, neg_padded,
        )
        char_lengths = torch.where(is_positive, pos_char_lengths, neg_char_lengths)
        labels = is_positive.float()

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(mel, char_ids, char_lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(eval_loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def _save_checkpoint(model, optimizer, step, epoch, config, tag=None):
    """Save AdaKWS checkpoint."""
    name = f"adakws_checkpoint_{tag or step}.pt"
    path = config.output_dir / name
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "text_encoder": model.text_encoder.state_dict(),
            "adaptive_modules": model.adaptive_modules.state_dict(),
            "classifier": model.classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "whisper_model": config.whisper_model,
                "lstm_hidden_dim": config.lstm_hidden_dim,
                "lstm_num_layers": config.lstm_num_layers,
                "char_embed_dim": config.char_embed_dim,
                "n_adaptive_modules": config.n_adaptive_modules,
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
    parser = argparse.ArgumentParser(description="Train AdaKWS")
    parser.add_argument("--whisper_model", default="large-v2")
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--languages", nargs="+", default=None, help="Multiple languages (overrides --language)")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    args = parser.parse_args()

    config = AdaKWSConfig(
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        device=args.device or _detect_device(),
        language=args.language,
        languages=args.languages,
        cache_dir=args.cache_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        resume_checkpoint=args.resume,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    logger.info("Config: %s", config)
    train(config)


if __name__ == "__main__":
    main()
