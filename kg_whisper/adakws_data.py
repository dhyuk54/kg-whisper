"""Data loading for AdaKWS training.

Provides character vocabulary, negative keyword sampling (Section 3),
and dataset/dataloader for binary keyword detection training.
"""

import logging
import random
import re
import string
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import whisper.audio
from .adakws_config import AdaKWSConfig

logger = logging.getLogger(__name__)

# Acoustically similar character pairs for substitution (paper Section 3)
SIMILAR_CHARS = {
    "s": "z", "z": "s",
    "p": "b", "b": "p",
    "t": "d", "d": "t",
    "k": "g", "g": "k",
    "f": "v", "v": "f",
    "m": "n", "n": "m",
    "c": "k", "i": "e", "e": "i",
    "a": "o", "o": "a",
    "u": "w", "w": "u",
}


class CharVocab:
    """Character-level vocabulary for keyword encoding.

    Maps characters to integer IDs. Index 0 is PAD, index 1 is UNK.
    Supports two modes:
    - Default: hardcoded ASCII chars (backward compatible for EN)
    - From data: auto-collected from training texts (for multilingual)
    """

    PAD = 0
    UNK = 1

    def __init__(self, chars=None):
        if chars is None:
            # Default: ASCII (backward compatible)
            chars = list(
                string.ascii_lowercase
                + string.ascii_uppercase
                + string.digits
                + " '-"
            )
        else:
            chars = sorted(set(chars))
        self.char2idx = {c: i + 2 for i, c in enumerate(chars)}
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.idx2char[self.PAD] = "<PAD>"
        self.idx2char[self.UNK] = "<UNK>"
        self.vocab_size = len(chars) + 2  # +2 for PAD and UNK

    @staticmethod
    def from_datasets(datasets_list, text_col="normalized_text", max_samples=5000):
        """Build CharVocab from training data (paper Section 4.2: character set observed during training)."""
        all_chars = set()
        for ds in datasets_list:
            n = min(max_samples, len(ds))
            for i in range(n):
                text = ds[i].get(text_col) or ds[i].get("raw_text", "")
                if text:
                    cleaned = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.lower().split()]
                    for w in cleaned:
                        all_chars.update(w)
        return CharVocab(chars=list(all_chars))

    def encode(self, text: str) -> List[int]:
        """Encode text to character IDs."""
        return [self.char2idx.get(c, self.UNK) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Decode character IDs to text."""
        return "".join(self.idx2char.get(i, "?") for i in ids if i > 1)


class NegativeSampler:
    """Negative keyword sampling strategies (paper Section 3).

    Four strategies:
    1. Random: pick a word not in the transcript
    2. Character substitution: swap acoustically similar characters
    3. Concatenation: combine two random words
    4. Nearest keyword: find a similar word (by edit distance)
    """

    def __init__(self, word_pool: List[str]):
        """
        Args:
            word_pool: Pool of words to sample negatives from.
        """
        self.word_pool = word_pool

    def random_negative(self, positive_words: Set[str]) -> str:
        """Pick a random word not in the transcript."""
        for _ in range(100):
            word = random.choice(self.word_pool)
            if word.lower() not in positive_words:
                return word
        return random.choice(self.word_pool)

    def char_substitution(self, keyword: str) -> str:
        """Substitute one or more acoustically similar characters (paper Section 3)."""
        if not keyword:
            return keyword
        chars = list(keyword.lower())
        positions = [i for i, c in enumerate(chars) if c in SIMILAR_CHARS]
        if positions:
            # Substitute 1 to len(positions) characters randomly
            n_subs = random.randint(1, max(1, len(positions)))
            chosen = random.sample(positions, min(n_subs, len(positions)))
            for pos in chosen:
                chars[pos] = SIMILAR_CHARS[chars[pos]]
        else:
            pos = random.randint(0, len(chars) - 1)
            chars[pos] = random.choice(string.ascii_lowercase)
        return "".join(chars)

    def concatenation(self, positive_keyword: str, positive_words: Set[str]) -> str:
        """Concatenate random keyword with positive keyword (paper Section 3).

        v⁻ = v ∘ v⁺ or v⁻ = v⁺ ∘ v, where v ∈ V \\ v.
        """
        v = self.random_negative(positive_words)
        if random.random() < 0.5:
            return v + positive_keyword    # v ∘ v⁺
        else:
            return positive_keyword + v    # v⁺ ∘ v

    def nearest_keyword(self, keyword: str, positive_words: Set[str]) -> str:
        """Find a word with small edit distance to the keyword.

        Uses sampling + edit distance for efficiency.
        """
        keyword_lower = keyword.lower()
        candidates = random.sample(
            self.word_pool, min(200, len(self.word_pool))
        )
        best_word = None
        best_dist = float("inf")

        for w in candidates:
            if w.lower() in positive_words:
                continue
            dist = _edit_distance(keyword_lower, w.lower())
            if 0 < dist < best_dist:
                best_dist = dist
                best_word = w

        return best_word or self.random_negative(positive_words)

    def sample(self, positive_words: Set[str], positive_keyword: str) -> str:
        """Sample a negative keyword using a random strategy.

        Args:
            positive_words: Set of words in the transcript (lowercase).
            positive_keyword: A positive keyword to base substitution/NK on.

        Returns:
            Negative keyword string.
        """
        strategy = random.randint(0, 3)

        if strategy == 0:
            return self.random_negative(positive_words)
        elif strategy == 1:
            return self.char_substitution(positive_keyword)
        elif strategy == 2:
            return self.concatenation(positive_keyword, positive_words)
        else:
            return self.nearest_keyword(positive_keyword, positive_words)


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,    # replace
            ))
        prev_row = curr_row
    return prev_row[-1]


class AdaKWSDataset(Dataset):
    """VoxPopuli dataset for AdaKWS keyword detection training.

    Each sample returns (mel, char_ids, label):
    - mel: [n_mels, 3000] mel spectrogram
    - char_ids: character IDs of the keyword
    - label: 1.0 (positive) or 0.0 (negative)
    """

    def __init__(
        self,
        split: str,
        config: AdaKWSConfig,
        char_vocab: CharVocab,
    ) -> None:
        from datasets import load_dataset, concatenate_datasets

        languages = config.languages or [config.language]

        max_samples = (
            config.max_train_samples if split == "train"
            else config.max_eval_samples
        )
        split_str = split
        if max_samples is not None:
            split_str = f"{split}[:{max_samples}]"

        # Load and concatenate all languages
        datasets_list = []
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

        if len(datasets_list) == 1:
            self.dataset = datasets_list[0]
        else:
            self.dataset = concatenate_datasets(datasets_list)
        logger.info("Total loaded: %d samples (%d languages)", len(self.dataset), len(languages))

        self.config = config
        self.char_vocab = char_vocab

        # Build word pool for negative sampling
        word_pool: List[str] = []
        pool_size = min(5000, len(self.dataset))
        text_col = "normalized_text" if "normalized_text" in self.dataset.column_names else "raw_text"
        texts = self.dataset.select(range(pool_size))[text_col]
        for text in texts:
            if text:
                cleaned = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.strip().split()]
                word_pool.extend([w for w in cleaned if w])
        self.word_pool = list(set(word_pool))
        self.neg_sampler = NegativeSampler(self.word_pool)
        logger.info("Word pool: %d unique words", len(self.word_pool))

    def _get_text(self, idx: int) -> str:
        """Get normalized transcript text."""
        item = self.dataset[idx]
        return item.get("normalized_text") or item.get("raw_text", "")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Audio → mel spectrogram
        audio_array = item["audio"]["array"]
        audio = torch.from_numpy(np.array(audio_array, dtype=np.float32))
        audio = whisper.audio.pad_or_trim(audio)
        mel = whisper.audio.log_mel_spectrogram(audio, n_mels=self.config.n_mels)

        # Get transcript words (strip punctuation)
        text = self._get_text(idx)
        words = [re.sub(r"[^\w'-]", "", w).strip("'") for w in text.strip().split()]
        words = [w for w in words if w]  # remove empty strings
        if not words:
            words = ["the"]  # fallback

        positive_words = set(w.lower() for w in words)

        # Positive keyword: random word from transcript (paper Section 3)
        pos_keyword = random.choice(words)
        pos_char_ids = self.char_vocab.encode(pos_keyword.lower())

        # Non-NK negative: Random / Char Substitution / Concatenation
        # (NK is computed in training loop using LSTM embeddings per paper)
        strategy = random.randint(0, 2)
        if strategy == 0:
            neg_keyword = self.neg_sampler.random_negative(positive_words)
        elif strategy == 1:
            neg_keyword = self.neg_sampler.char_substitution(pos_keyword)
        else:
            neg_keyword = self.neg_sampler.concatenation(pos_keyword, positive_words)
        neg_char_ids = self.char_vocab.encode(neg_keyword.lower())

        return {
            "mel": mel,
            "pos_char_ids": torch.tensor(pos_char_ids, dtype=torch.long),
            "neg_char_ids": torch.tensor(neg_char_ids, dtype=torch.long),
        }


def _pad_char_ids(batch_ids: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length char ID tensors and return lengths."""
    lengths = [t.size(0) for t in batch_ids]
    max_len = max(max(lengths), 1)
    padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long)
    for i, t in enumerate(batch_ids):
        if t.size(0) > 0:
            padded[i, : t.size(0)] = t
    return padded, torch.tensor(lengths, dtype=torch.long)


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate with dynamic padding for positive and negative char_ids."""
    mels = torch.stack([b["mel"] for b in batch])

    pos_ids, pos_lens = _pad_char_ids([b["pos_char_ids"] for b in batch])
    neg_ids, neg_lens = _pad_char_ids([b["neg_char_ids"] for b in batch])

    return {
        "mel": mels,
        "pos_char_ids": pos_ids,
        "pos_char_lengths": pos_lens,
        "neg_char_ids": neg_ids,
        "neg_char_lengths": neg_lens,
    }


def create_dataloader(
    split: str,
    config: AdaKWSConfig,
    char_vocab: CharVocab,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for AdaKWS training/evaluation."""
    dataset = AdaKWSDataset(split=split, config=config, char_vocab=char_vocab)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=(split == "train"),
    )
