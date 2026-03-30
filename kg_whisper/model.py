"""KG-Whisper-PT model: Keyword-Guided Whisper with Prompt Tuning.

Implements the Learned Prefix approach from Section 2.2 of
"Keyword-Guided Adaptation of Automatic Speech Recognition" (arXiv:2406.02649v1).

Architecture:
    - Entire Whisper model (encoder + decoder) is FROZEN
    - Only a learned prefix q ∈ R^{N×D} is trainable
    - Prefix is inserted after <|startofprev|> in the decoder prompt

Prompt structure (Figure 1b + Whisper standard decoding):
    [SOP] [prefix₁..prefixₙ] [kws_tokens] [SOT] [LANG] [TASK] [no_timestamps] [text₁..textₜ]
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from whisper.model import Whisper

logger = logging.getLogger(__name__)


class KGWhisperPT(nn.Module):
    """KG-Whisper with Learned Prefix Prompt Tuning.

    Args:
        whisper_model: Pre-trained Whisper model (frozen).
        prefix_length: Number of learned prefix vectors (N).
        eot_id: End-of-text token ID for loss target.
    """

    def __init__(
        self,
        whisper_model: Whisper,
        prefix_length: int = 12,
        eot_id: int = 50257,
    ) -> None:
        super().__init__()
        self.whisper = whisper_model
        self.prefix_length = prefix_length
        self.eot_id = eot_id

        n_state = whisper_model.dims.n_text_state

        # Learned prefix — the ONLY trainable parameters
        # q ∈ R^{N×D}, initialized from N(0, 0.02)
        self.prefix = nn.Parameter(torch.randn(prefix_length, n_state) * 0.02)

        # Freeze all Whisper parameters
        for param in self.whisper.parameters():
            param.requires_grad = False

        trainable = self.prefix.numel()
        total = sum(p.numel() for p in self.whisper.parameters())
        logger.info(
            "Trainable: %d params (prefix %d×%d) / %d total (%.4f%%)",
            trainable,
            prefix_length,
            n_state,
            total,
            100.0 * trainable / total,
        )

    def _build_embeddings(
        self,
        prompt_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Build the full decoder embedding sequence with prefix injected.

        Input token layout:
            prompt_tokens = [SOP, kws₁..kwsₘ, SOT, LANG, TASK, no_timestamps]
            text_tokens   = [text₁, text₂, ..., textₜ]

        Output embedding layout:
            [SOP_emb, prefix₁..ₙ, kws_embs, SOT_emb, LANG_emb, TASK_emb, noTS_emb, text_embs]

        Returns:
            Tensor of shape [B, seq_len, D].
        """
        decoder = self.whisper.decoder
        B = prompt_tokens.size(0)

        prompt_embs = decoder.token_embedding(prompt_tokens)  # [B, P, D]
        text_embs = decoder.token_embedding(text_tokens)  # [B, T, D]
        prefix_embs = self.prefix.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]

        return torch.cat(
            [
                prompt_embs[:, :1, :],  # SOP
                prefix_embs,  # learned prefix
                prompt_embs[:, 1:, :],  # kws + SOT + LANG + TASK
                text_embs,  # text tokens
            ],
            dim=1,
        )

    def forward(
        self,
        mel: torch.Tensor,
        prompt_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with CE loss on text positions.

        Args:
            mel: [B, n_mels, 3000] mel spectrogram.
            prompt_tokens: [B, prompt_len] token IDs for prompt.
            text_tokens: [B, max_text_len] token IDs for text (padded).
            text_lengths: [B] actual text length per sample.

        Returns:
            (loss, logits) tuple.
        """
        B = mel.size(0)
        device = mel.device
        decoder = self.whisper.decoder

        # 1. Encode audio (frozen encoder)
        with torch.no_grad():
            audio_features = self.whisper.encoder(mel)

        # 2. Build embedding sequence with prefix
        full_embs = self._build_embeddings(prompt_tokens, text_tokens)
        seq_len = full_embs.size(1)

        # 3. Add positional embeddings
        full_embs = full_embs + decoder.positional_embedding[:seq_len]
        full_embs = full_embs.to(audio_features.dtype)

        # 4. Run through decoder blocks
        # Blocks are frozen but gradients flow through prefix via the embedding
        for block in decoder.blocks:
            full_embs = block(full_embs, audio_features, mask=decoder.mask)

        # 5. Project to logits
        full_embs = decoder.ln(full_embs)
        logits = (
            full_embs @ decoder.token_embedding.weight.to(full_embs.dtype).T
        ).float()

        # 6. Build labels — CE loss only on text token positions
        #
        #    Position layout in full_embs:
        #      [0]        : SOP
        #      [1..N]     : prefix (N = prefix_length)
        #      [N+1..N+P-1] : rest of prompt (P-1 tokens)
        #      [N+P..]    : text tokens
        #
        #    text_start_pos = prefix_length + prompt_tokens.size(1)
        #
        #    Autoregressive targets (shifted by 1):
        #      logits[text_start-1] → text[0]   (TASK position predicts first text token)
        #      logits[text_start+j] → text[j+1]
        #      logits[text_start+tl-1] → EOT
        #
        text_start_pos = self.prefix_length + prompt_tokens.size(1)
        labels = torch.full((B, seq_len), -100, dtype=torch.long, device=device)

        for i in range(B):
            tl = text_lengths[i].item()
            if tl > 0:
                start = text_start_pos - 1
                labels[i, start : start + tl] = text_tokens[i, :tl]
                labels[i, start + tl] = self.eot_id

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        prompt_tokens: torch.Tensor,
        eot_id: int,
        max_new_tokens: int = 200,
    ) -> List[int]:
        """Autoregressive greedy generation with prefix.

        Args:
            mel: [1, n_mels, 3000] single mel spectrogram.
            prompt_tokens: [1, prompt_len] prompt token IDs.
            eot_id: End-of-text token ID.
            max_new_tokens: Max tokens to generate.

        Returns:
            List of generated token IDs (excluding EOT).
        """
        device = mel.device
        decoder = self.whisper.decoder

        audio_features = self.whisper.encoder(mel)

        # Build initial embedding: [SOP, prefix, rest_prompt]
        prompt_embs = decoder.token_embedding(prompt_tokens)
        prefix_embs = self.prefix.unsqueeze(0)

        current_embs = torch.cat(
            [
                prompt_embs[:, :1, :],
                prefix_embs,
                prompt_embs[:, 1:, :],
            ],
            dim=1,
        )

        generated: List[int] = []

        for _ in range(max_new_tokens):
            seq_len = current_embs.size(1)
            if seq_len > decoder.positional_embedding.size(0):
                break

            x = current_embs + decoder.positional_embedding[:seq_len]
            x = x.to(audio_features.dtype)

            for block in decoder.blocks:
                x = block(x, audio_features, mask=decoder.mask)

            x = decoder.ln(x)
            logit = (
                x[:, -1:, :] @ decoder.token_embedding.weight.to(x.dtype).T
            ).float()

            next_id = logit.argmax(dim=-1).item()
            if next_id == eot_id:
                break

            generated.append(next_id)
            next_emb = decoder.token_embedding(
                torch.tensor([[next_id]], device=device)
            )
            current_embs = torch.cat([current_embs, next_emb], dim=1)

        return generated
