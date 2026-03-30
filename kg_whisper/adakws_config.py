"""Configuration for AdaKWS (Open-vocabulary Keyword Spotting with AdaIN).

Paper: "Open-vocabulary Keyword-spotting with Adaptive Instance Normalization"
(arXiv:2309.08561)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AdaKWSConfig:
    """AdaKWS training configuration."""

    # Model
    whisper_model: str = "large-v2"
    n_adaptive_modules: int = 2       # paper: 2 sequential keyword-adaptive modules

    # Text Encoder (Character LSTM) — paper Section 2
    char_embed_dim: int = 64
    lstm_hidden_dim: int = 256        # paper: 256
    lstm_num_layers: int = 4          # paper: 4

    # Training — paper Section 4
    learning_rate: float = 1e-4       # paper: 1e-4
    batch_size: int = 24              # limited by 16GB VRAM (peak ~13.5GB)
    gradient_accumulation_steps: int = 6  # effective batch = 144 (paper: 144)
    num_epochs: int = 25              # paper: 25
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05       # 5% warmup (not specified in paper)

    # Data
    dataset_name: str = "facebook/voxpopuli"
    cache_dir: Optional[str] = None
    language: str = "en"              # single language (backward compatible)
    languages: Optional[List[str]] = None  # multi-language: ["en","de","fr",...] (overrides language)
    n_mels: int = 80                  # overridden by model.dims at runtime

    # Negative sampling — paper Section 3
    neg_random_ratio: float = 0.25
    neg_char_sub_ratio: float = 0.25
    neg_concat_ratio: float = 0.25
    neg_nearest_ratio: float = 0.25

    # Evaluation & logging
    eval_steps: int = 500
    save_steps: int = 2500
    log_steps: int = 50
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("kg_whisper/outputs/adakws"))
    resume_checkpoint: Optional[str] = None  # path to checkpoint for resuming training

    # Device
    device: str = "cpu"
