"""Simulated AdaKWS: keyword sampling from transcripts (Section 2.3).

Paper algorithm:
    1. Randomly select num_keywords ∈ [1, 5]
    2. For each keyword:
       a. Coin flip: positive (0.9) or negative (0.1)
       b. Randomly pick target BPE token length L ∈ [1, 4]
       c. Find a word with exactly L BPE tokens
    3. Concatenate with | delimiter
    4. Place between SOP and SOT tokens
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _group_by_bpe_length(
    words: List[str], encoding, max_bpe: int
) -> Dict[int, List[str]]:
    """Group words by their BPE token count.

    Args:
        words: List of words to group.
        encoding: tiktoken Encoding object.
        max_bpe: Maximum BPE tokens per word.

    Returns:
        Dict mapping token count → list of words.
    """
    groups: Dict[int, List[str]] = defaultdict(list)
    for w in words:
        n = len(encoding.encode(w))
        if 1 <= n <= max_bpe:
            groups[n].append(w)
    return dict(groups)


class KWSSimulator:
    """Simulates KWS by sampling keywords following Section 2.3.

    Args:
        encoding: tiktoken Encoding for BPE tokenization.
        max_keywords: Maximum keywords per utterance (paper: 5).
        min_keywords: Minimum keywords per utterance (paper: 1).
        max_bpe_tokens: Maximum BPE tokens per keyword (paper: 4).
        positive_ratio: Probability of positive keyword (paper: 0.9).
        delimiter: Keyword concatenation delimiter (paper: |).
    """

    def __init__(
        self,
        encoding,
        max_keywords: int = 5,
        min_keywords: int = 1,
        max_bpe_tokens: int = 4,
        positive_ratio: float = 0.9,
        delimiter: str = " | ",
    ) -> None:
        self.encoding = encoding
        self.max_keywords = max_keywords
        self.min_keywords = min_keywords
        self.max_bpe_tokens = max_bpe_tokens
        self.positive_ratio = positive_ratio
        self.delimiter = delimiter
        self.negative_pool: List[str] = []
        self._neg_by_length: Dict[int, List[str]] = {}

    def set_negative_pool(self, words: List[str]) -> None:
        """Build pool of valid negative keywords, grouped by BPE length."""
        unique_words = list(set(words))
        self._neg_by_length = _group_by_bpe_length(
            unique_words, self.encoding, self.max_bpe_tokens
        )
        self.negative_pool = [
            w for ws in self._neg_by_length.values() for w in ws
        ]
        logger.info("Negative pool: %d unique words", len(self.negative_pool))

    def sample_keywords(
        self, transcript: str, batch_transcripts: Optional[List[str]] = None
    ) -> str:
        """Sample keywords following the paper's Section 2.3 algorithm.

        Steps:
            1. Pick num_keywords ∈ [1, 5]
            2. For each keyword:
               - Coin flip → positive (0.9) or negative (0.1)
               - Random BPE length L ∈ [1, 4]
               - Pick word with exactly L BPE tokens
            3. Join with | delimiter

        Args:
            transcript: Ground-truth transcript text.
            batch_transcripts: Other transcripts in the batch for negative sampling.

        Returns:
            Keywords string, e.g. "weather | today | forecast".
        """
        words = transcript.strip().split()
        if not words:
            return ""

        # Group transcript words by BPE token length
        pos_by_length = _group_by_bpe_length(
            words, self.encoding, self.max_bpe_tokens
        )
        all_valid = [w for ws in pos_by_length.values() for w in ws]
        if not all_valid:
            all_valid = words[:1]

        # Build negative pool from batch transcripts (paper Section 2.3)
        if batch_transcripts:
            batch_neg_words = []
            for t in batch_transcripts:
                batch_neg_words.extend(t.strip().split())
            batch_neg_by_length = _group_by_bpe_length(
                batch_neg_words, self.encoding, self.max_bpe_tokens
            )
            batch_neg_all = [w for ws in batch_neg_by_length.values() for w in ws]
        else:
            batch_neg_by_length = self._neg_by_length
            batch_neg_all = self.negative_pool

        num_kw = random.randint(
            self.min_keywords, min(self.max_keywords, max(len(all_valid), 1))
        )

        keywords: List[str] = []
        for _ in range(num_kw):
            # Step a: positive or negative
            is_positive = (
                random.random() < self.positive_ratio or not batch_neg_all
            )

            # Step b: random BPE token length
            target_len = random.randint(1, self.max_bpe_tokens)

            # Step c: find word with matching length
            if is_positive:
                candidates = pos_by_length.get(target_len, [])
                if not candidates:
                    candidates = all_valid  # fallback
                keywords.append(random.choice(candidates))
            else:
                candidates = batch_neg_by_length.get(target_len, [])
                if not candidates:
                    candidates = batch_neg_all  # fallback
                keywords.append(random.choice(candidates))

        return self.delimiter.join(keywords)
