"""A small, real Transformer LM for adapter and benchmark tests.

This is a genuine (if small) transformer, not a stub: it has ``nn.MultiheadAttention``
blocks, LayerNorm, an MLP and an LM head, because the T1 layer hooks in W4 must be
tested against the module structure they will actually see. A toy ``nn.Linear``
would not exercise named-module traversal, shared parameters, or parameter-free
submodules (a real source of hook bugs).

Deterministic: `make_tiny_lm(seed=...)` produces identical weights across runs, so
scenario golden files are stable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TinyTransformerLM", "TinyLMConfig", "make_tiny_lm", "synthetic_batch"]


class TinyLMConfig:
    def __init__(
        self,
        *,
        vocab_size: int = 128,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 32,
        dropout: float = 0.0,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class TransformerBlock(nn.Module):
    """Pre-norm transformer block. Pre-norm so gradients stay well behaved
    enough that a *deliberate* explosion is distinguishable from a structural
    one -- which matters for the W5 scenario corpus."""

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Embedding -> N blocks -> LayerNorm -> tied LM head."""

    _model_id = "tiny-transformer-lm"

    def __init__(self, config: Optional[TinyLMConfig] = None) -> None:
        super().__init__()
        self.config = config or TinyLMConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                "sequence length %d exceeds max_seq_len %d"
                % (seq_len, self.config.max_seq_len)
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return out


def make_tiny_lm(
    *, seed: int = 1234, device: Optional[torch.device] = None, **config_kwargs
) -> Tuple[TinyTransformerLM, TinyLMConfig]:
    """Deterministic tiny LM. Returns ``(model, config)``."""
    torch.manual_seed(seed)
    config = TinyLMConfig(**config_kwargs)
    model = TinyTransformerLM(config)
    if device is not None:
        model = model.to(device)
    return model, config


def synthetic_batch(
    *,
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Deterministic random batch with labels offset by one (next-token)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=generator, dtype=torch.long
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
    return {"input_ids": input_ids, "labels": labels}
