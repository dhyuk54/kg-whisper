"""Data loading for KG-Whisper-PT training with Voxpopuli."""

import logging
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import whisper.audio
from whisper.tokenizer import Tokenizer, get_tokenizer

from .config import KGWhisperConfig
from .kws_simulator import KWSSimulator

logger = logging.getLogger(__name__)


class VoxpopuliDataset(Dataset):
    """Voxpopuli dataset for KG-Whisper-PT.

    Supports single or multi-language training.

    Each sample returns:
        mel: [n_mels, 3000] mel spectrogram (30s padded)
        prompt_tokens: [prompt_len] = [SOP, kws_padded..., SOT, LANG, TASK, no_timestamps]
        text_tokens: [text_len] transcript BPE token IDs
    """

    def __init__(
        self,
        split: str,
        config: KGWhisperConfig,
        tokenizer: Tokenizer,
        whisper_model=None,
    ) -> None:
        from datasets import load_dataset, concatenate_datasets

        languages = config.languages or [config.language]

        max_samples = (
            config.max_train_samples if split == "train" else config.max_eval_samples
        )
        split_str = split
        if max_samples is not None:
            split_str = f"{split}[:{max_samples}]"

        # Load and concatenate all languages
        datasets_list = []
        lang_labels = []  # track which language each sample belongs to
        for lang in languages:
            logger.info("Loading VoxPopuli %s split: %s", lang, split_str)
            ds = load_dataset(
                config.dataset_name,
                lang,
                split=split_str,
                cache_dir=config.cache_dir,
            )
            logger.info("  %s: %d samples", lang, len(ds))
            datasets_list.append(ds)
            lang_labels.extend([lang] * len(ds))

        if len(datasets_list) == 1:
            self.dataset = datasets_list[0]
        else:
            self.dataset = concatenate_datasets(datasets_list)
        self.lang_labels = lang_labels
        logger.info("Total loaded: %d samples (%d languages)",
                    len(self.dataset), len(languages))

        self.config = config
        self.tokenizer = tokenizer
        self.whisper_model = whisper_model

        # Build per-language tokenizers for SOT sequence
        self.lang_tokenizers = {}
        if whisper_model is not None:
            for lang in languages:
                self.lang_tokenizers[lang] = get_tokenizer(
                    whisper_model.is_multilingual,
                    num_languages=whisper_model.num_languages,
                    language=lang,
                    task="transcribe",
                )

        # KWS simulator
        self.kws_sim = KWSSimulator(
            encoding=tokenizer.encoding,
            max_keywords=config.max_keywords,
            min_keywords=config.min_keywords,
            max_bpe_tokens=config.max_keyword_bpe_tokens,
            positive_ratio=config.positive_ratio,
            delimiter=config.kws_delimiter,
        )

        # Build negative pool from a subset (text only, no audio decoding)
        neg_words: List[str] = []
        pool_size = min(2000, len(self.dataset))
        text_col = "normalized_text" if "normalized_text" in self.dataset.column_names else "raw_text"
        texts = self.dataset.select(range(pool_size))[text_col]
        for text in texts:
            if text:
                neg_words.extend(text.strip().split())
        self.kws_sim.set_negative_pool(neg_words)

    def _get_text(self, idx: int) -> str:
        """Get normalized transcript text."""
        item = self.dataset[idx]
        return item.get("normalized_text") or item.get("raw_text", "")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Audio → mel spectrogram
        audio_array = item["audio"]["array"]
        audio = torch.from_numpy(np.array(audio_array, dtype=np.float32))
        audio = whisper.audio.pad_or_trim(audio)
        mel = whisper.audio.log_mel_spectrogram(audio, n_mels=self.config.n_mels)

        # Transcript
        text = self._get_text(idx)
        text_token_ids = self.tokenizer.encoding.encode(text)
        text_token_ids = text_token_ids[: self.config.max_text_tokens]

        # Language-specific SOT sequence
        lang = self.lang_labels[idx]
        if lang in self.lang_tokenizers:
            sot_sequence = list(self.lang_tokenizers[lang].sot_sequence_including_notimestamps)
        else:
            sot_sequence = list(self.tokenizer.sot_sequence_including_notimestamps)

        return {
            "mel": mel,
            "text": text,  # raw text for batch-level keyword sampling
            "text_tokens": torch.tensor(text_token_ids, dtype=torch.long),
            "sot_sequence": sot_sequence,
        }


def _collate_fn(
    batch: List[dict], eot_id: int, kws_sim=None,
    tokenizer=None, max_kws_tokens: int = 20, sot_prev: int = 50361,
) -> Dict[str, torch.Tensor]:
    """Collate with batch-level keyword sampling (paper Section 2.3)."""
    B = len(batch)
    mels = torch.stack([b["mel"] for b in batch])

    # Batch-level keyword sampling: negative keywords from other samples in batch
    all_texts = [b["text"] for b in batch]
    prompt_tokens_list = []
    for i in range(B):
        # Other transcripts in batch as negative source
        other_texts = [all_texts[j] for j in range(B) if j != i]
        kws_string = kws_sim.sample_keywords(all_texts[i], batch_transcripts=other_texts)

        kws_token_ids = tokenizer.encoding.encode(kws_string)
        kws_token_ids = kws_token_ids[:max_kws_tokens]
        pad_len = max_kws_tokens - len(kws_token_ids)
        kws_padded = kws_token_ids + [eot_id] * pad_len

        prompt = [sot_prev] + kws_padded + batch[i]["sot_sequence"]
        prompt_tokens_list.append(torch.tensor(prompt, dtype=torch.long))

    prompt_tokens = torch.stack(prompt_tokens_list)

    # Pad text tokens to max length in batch
    text_lens = [b["text_tokens"].size(0) for b in batch]
    max_text_len = max(max(text_lens), 1)

    text_tokens = torch.full(
        (B, max_text_len), eot_id, dtype=torch.long
    )
    for i, b in enumerate(batch):
        tl = b["text_tokens"].size(0)
        if tl > 0:
            text_tokens[i, :tl] = b["text_tokens"]

    return {
        "mel": mels,
        "prompt_tokens": prompt_tokens,
        "text_tokens": text_tokens,
        "text_lengths": torch.tensor(text_lens, dtype=torch.long),
    }


def create_dataloader(
    split: str,
    config: KGWhisperConfig,
    tokenizer: Tokenizer,
    shuffle: bool = True,
    whisper_model=None,
) -> DataLoader:
    """Create a DataLoader for the specified split."""
    dataset = VoxpopuliDataset(
        split=split, config=config, tokenizer=tokenizer,
        whisper_model=whisper_model,
    )
    collate = partial(
        _collate_fn,
        eot_id=tokenizer.eot,
        kws_sim=dataset.kws_sim,
        tokenizer=tokenizer,
        max_kws_tokens=config.max_kws_tokens,
        sot_prev=tokenizer.sot_prev,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,  # must be 0 since collate uses kws_sim with state
        pin_memory=False,
        drop_last=(split == "train"),
    )
