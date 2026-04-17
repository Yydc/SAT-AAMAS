"""A minimal causal LM with an attached value head.

This replaces ``verl.models.causal_lm.ModelWithValueHead`` so that SAT runs
without a veRL checkout. The interface is intentionally small: ``forward``
returns ``(logits, values)`` for an input of token ids, and ``generate``
delegates to the underlying Hugging Face model.

Use the built-in ``sat:tiny`` backend for no-download smoke tests. This
wrapper is for paper-scale Hugging Face checkpoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


class ModelWithValueHead(nn.Module):
    """Causal LM backbone with a scalar value head on the last hidden state."""

    def __init__(
        self,
        model_path: str,
        model_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            config=config,
        )
        hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", 768)
        self.value_head = nn.Linear(hidden, 1, dtype=model_dtype)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.value_head.weight, std=1e-3)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        values = self.value_head(hidden).squeeze(-1)
        return out.logits, values

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)
