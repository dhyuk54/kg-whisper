"""AdaKWS: Open-vocabulary Keyword Spotting with Adaptive Instance Normalization.

Implements the AdaKWS model from arXiv:2309.08561.

Architecture:
    - Whisper audio encoder (FROZEN)
    - Character-based LSTM text encoder → AdaIN parameters
    - 2 Keyword-Adaptive Modules (Transformer blocks with AdaIN replacing LayerNorm)
    - Max pooling → Linear classifier → P(keyword | audio)
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from whisper.model import MultiHeadAttention, Linear, Whisper

logger = logging.getLogger(__name__)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (Eq. 1 in paper).

    AdaIN(z, v) = sigma_v * ((z - mu_z) / sigma_z) + mu_v

    Normalization is per-instance, per-channel, across the time dimension.
    """

    def __init__(self, n_state: int):
        super().__init__()
        self.n_state = n_state

    def forward(self, z: Tensor, mu_v: Tensor, sigma_v: Tensor) -> Tensor:
        """
        Args:
            z: Audio features [B, T, C].
            mu_v: Keyword-conditioned mean [B, C].
            sigma_v: Keyword-conditioned std [B, C] (positive).

        Returns:
            Normalized features [B, T, C].
        """
        # Instance statistics across time dimension
        mu_z = z.mean(dim=1, keepdim=True)             # [B, 1, C]
        sigma_z = z.std(dim=1, keepdim=True) + 1e-6     # [B, 1, C]

        z_norm = (z - mu_z) / sigma_z
        return sigma_v.unsqueeze(1) * z_norm + mu_v.unsqueeze(1)


class KeywordAdaptiveModule(nn.Module):
    """Transformer encoder block with AdaIN replacing LayerNorm.

    Mirrors whisper.model.ResidualAttentionBlock (lines 142-171)
    but swaps attn_ln and mlp_ln with AdaIN layers.

    Structure:
        x = x + attn(adain_attn(x, mu_attn, sigma_attn))
        x = x + mlp(adain_mlp(x, mu_mlp, sigma_mlp))
    """

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.adain_attn = AdaIN(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.adain_mlp = AdaIN(n_state)

    def forward(
        self,
        x: Tensor,
        mu_attn: Tensor,
        sigma_attn: Tensor,
        mu_mlp: Tensor,
        sigma_mlp: Tensor,
    ) -> Tensor:
        """
        Args:
            x: Audio features [B, T, C].
            mu_attn, sigma_attn: AdaIN params for attention [B, C].
            mu_mlp, sigma_mlp: AdaIN params for FFN [B, C].

        Returns:
            Processed features [B, T, C].
        """
        x = x + self.attn(self.adain_attn(x, mu_attn, sigma_attn))[0]
        x = x + self.mlp(self.adain_mlp(x, mu_mlp, sigma_mlp))
        return x


class CharLSTMEncoder(nn.Module):
    """Character-based LSTM text encoder (paper Section 2).

    Maps a keyword (as character sequence) to AdaIN normalization
    parameters for all AdaIN layers.

    Args:
        vocab_size: Character vocabulary size.
        embed_dim: Character embedding dimension.
        hidden_dim: LSTM hidden dimension (paper: 256).
        num_layers: Number of LSTM layers (paper: 4).
        n_adain_layers: Total number of AdaIN layers (2 per module × 2 modules = 4).
        n_state: Audio feature dimension (e.g., 1280 for large-v2).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_adain_layers: int,
        n_state: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Project LSTM output to AdaIN parameters: mu and sigma for each layer
        # Bidirectional: hidden_dim * 2
        # Total output: n_adain_layers * 2 (mu + sigma) * n_state
        self.param_proj = nn.Linear(hidden_dim * 2, n_adain_layers * 2 * n_state)
        self.n_adain_layers = n_adain_layers
        self.n_state = n_state

        # Initialize sigma bias so softplus output ≈ 1.0
        # softplus(x) = ln(1 + exp(x)), softplus(0.5413) ≈ 1.0
        nn.init.zeros_(self.param_proj.bias)
        with torch.no_grad():
            sigma_start = n_adain_layers * n_state  # sigma params start here
            self.param_proj.bias[sigma_start:] = 0.5413

    def forward(
        self, char_ids: Tensor, char_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            char_ids: Character token IDs [B, max_char_len].
            char_lengths: Actual character lengths [B].

        Returns:
            mu: [B, n_adain_layers, n_state]
            sigma: [B, n_adain_layers, n_state] (positive)
        """
        emb = self.embedding(char_ids)  # [B, L, embed_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, char_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: [num_layers * 2, B, hidden_dim] for bidirectional
        # Concatenate forward and backward last layer
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, hidden_dim * 2]

        params = self.param_proj(h)  # [B, n_adain_layers * 2 * n_state]
        params = params.view(-1, self.n_adain_layers, 2, self.n_state)

        mu = params[:, :, 0, :]       # [B, n_adain_layers, n_state]
        sigma_raw = params[:, :, 1, :]  # [B, n_adain_layers, n_state]
        sigma = F.softplus(sigma_raw)   # ensure positive, initialized near 1.0

        return mu, sigma


class AdaKWS(nn.Module):
    """AdaKWS: Open-vocabulary Keyword Spotting with Adaptive Instance Normalization.

    Args:
        whisper_model: Pre-trained Whisper model (encoder will be frozen).
        config: AdaKWS configuration.
        char_vocab_size: Size of the character vocabulary.
    """

    def __init__(
        self,
        whisper_model: Whisper,
        char_vocab_size: int,
        char_embed_dim: int = 64,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 4,
        n_adaptive_modules: int = 2,
    ):
        super().__init__()

        self.encoder = whisper_model.encoder
        n_state = whisper_model.dims.n_audio_state
        n_head = whisper_model.dims.n_audio_head
        self.n_adaptive_modules = n_adaptive_modules

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Number of AdaIN layers: 2 per adaptive module (attn + mlp)
        n_adain_layers = n_adaptive_modules * 2

        # Text encoder
        self.text_encoder = CharLSTMEncoder(
            vocab_size=char_vocab_size,
            embed_dim=char_embed_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            n_adain_layers=n_adain_layers,
            n_state=n_state,
        )

        # Keyword-adaptive modules
        self.adaptive_modules = nn.ModuleList([
            KeywordAdaptiveModule(n_state, n_head)
            for _ in range(n_adaptive_modules)
        ])

        # Binary classifier
        self.classifier = nn.Linear(n_state, 1)

        # Log parameter counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            "AdaKWS: trainable=%d params, frozen encoder=%d params",
            trainable, frozen,
        )

    def forward(
        self,
        mel: Tensor,
        char_ids: Tensor,
        char_lengths: Tensor,
    ) -> Tensor:
        """Forward pass: detect if keyword is present in audio.

        Args:
            mel: Mel spectrogram [B, n_mels, 3000].
            char_ids: Keyword character IDs [B, max_char_len].
            char_lengths: Actual character lengths [B].

        Returns:
            logits: [B] — binary classification logits.
        """
        # 1. Encode audio (frozen)
        with torch.no_grad():
            audio_features = self.encoder(mel)  # [B, 1500, n_state]

        # 2. Encode keyword text → AdaIN parameters
        mu, sigma = self.text_encoder(char_ids, char_lengths)
        # mu, sigma: [B, n_adain_layers, n_state]

        # 3. Apply keyword-adaptive modules
        x = audio_features.float()
        adain_idx = 0
        for module in self.adaptive_modules:
            x = module(
                x,
                mu[:, adain_idx], sigma[:, adain_idx],          # attn AdaIN
                mu[:, adain_idx + 1], sigma[:, adain_idx + 1],  # mlp AdaIN
            )
            adain_idx += 2

        # 4. Max pool over time dimension
        x = x.max(dim=1).values  # [B, n_state]

        # 5. Classify
        logits = self.classifier(x).squeeze(-1)  # [B]
        return logits

    def predict(
        self,
        mel: Tensor,
        keywords: List[str],
        char_vocab,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Predict which keywords are present in the audio.

        Args:
            mel: Mel spectrogram [1, n_mels, 3000].
            keywords: List of candidate keywords.
            char_vocab: CharVocab instance for encoding.
            threshold: Detection threshold.

        Returns:
            List of (keyword, probability) for detected keywords.
        """
        self.eval()
        device = mel.device
        detected = []

        with torch.no_grad():
            audio_features = self.encoder(mel)

            for kw in keywords:
                ids = char_vocab.encode(kw.lower())
                char_ids = torch.tensor([ids], dtype=torch.long, device=device)
                char_lengths = torch.tensor([len(ids)], dtype=torch.long)

                mu, sigma = self.text_encoder(char_ids, char_lengths)

                x = audio_features.float()
                adain_idx = 0
                for module in self.adaptive_modules:
                    x = module(
                        x,
                        mu[:, adain_idx], sigma[:, adain_idx],
                        mu[:, adain_idx + 1], sigma[:, adain_idx + 1],
                    )
                    adain_idx += 2

                x = x.max(dim=1).values
                logit = self.classifier(x).squeeze(-1)
                prob = torch.sigmoid(logit).item()

                if prob >= threshold:
                    detected.append((kw, prob))

        return detected
