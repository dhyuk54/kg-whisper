"""Configuration for KG-Whisper-PT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KGWhisperConfig:
    """KG-Whisper-PT training configuration.

    Based on paper: "Keyword-Guided Adaptation of Automatic Speech Recognition"
    (arXiv:2406.02649v1), Section 3 - Experiments.
    """

    # Model
    whisper_model: str = "small"
    prefix_length: int = 12  # N in paper, Table 4 shows 12 is optimal

    # Training (Section 3)
    learning_rate: float = 5e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 30000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # KWS simulation (Section 2.3)
    max_keywords: int = 5
    min_keywords: int = 1
    max_keyword_bpe_tokens: int = 4
    positive_ratio: float = 0.9
    max_kws_tokens: int = 20  # fixed padding length for KWS token sequence
    kws_delimiter: str = " | "

    # Data
    dataset_name: str = "facebook/voxpopuli"
    cache_dir: Optional[str] = None  # HuggingFace datasets cache directory
    language: str = "en"              # single language (backward compatible)
    languages: Optional[list] = None  # multi-language: ["en","de","fr",...] (overrides language)
    max_text_tokens: int = 200
    max_train_samples: Optional[int] = None  # None = use all
    max_eval_samples: Optional[int] = None
    n_mels: int = 80  # overridden by model.dims at runtime

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("kg_whisper/outputs"))
    resume_checkpoint: Optional[str] = None

    # Logging & evaluation
    eval_steps: int = 1000
    save_steps: int = 5000
    log_steps: int = 100
    max_generate_samples: int = 5  # samples for autoregressive WER eval

    # Device
    device: str = "cpu"
