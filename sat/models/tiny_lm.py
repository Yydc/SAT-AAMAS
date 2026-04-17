"""Tiny local causal LM used by the SAT demo.

The production configs use Hugging Face causal LMs.  The demo config uses
``sat:tiny`` so a fresh checkout can exercise the full SAT training,
checkpointing, and evaluation path without downloading model weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


class TinyBatch(dict):
    """Dictionary returned by ``TinyTokenizer`` with a Hugging Face-like ``to``."""

    def to(self, device: torch.device | str) -> "TinyBatch":
        return TinyBatch({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class TinyTokenizer:
    """Small byte-level tokenizer for local smoke tests."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(self) -> None:
        chars = [chr(i) for i in range(32, 127)] + ["\n"]
        self.id_to_token = [self.pad_token, self.eos_token, self.unk_token] + chars
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.id_to_token)}
        self.vocab_size = len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(ch, self.unk_token_id) for ch in text]

    def decode(self, token_ids: Iterable[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        out = []
        for token_id in token_ids:
            idx = int(token_id)
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id, self.unk_token_id}:
                continue
            if 0 <= idx < len(self.id_to_token):
                tok = self.id_to_token[idx]
                if tok not in {self.pad_token, self.eos_token, self.unk_token}:
                    out.append(tok)
        return "".join(out)

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: object,
    ) -> TinyBatch:
        del padding
        ids = self.encode(text)
        if truncation and max_length is not None:
            ids = ids[-max_length:]
        if not ids:
            ids = [self.eos_token_id]
        if return_tensors != "pt":
            raise ValueError("TinyTokenizer only supports return_tensors='pt'")
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return TinyBatch({"input_ids": input_ids, "attention_mask": attention_mask})


@dataclass
class TinyGenerateOutput:
    sequences: torch.LongTensor
    scores: tuple[torch.Tensor, ...]
    hidden_states: tuple[tuple[torch.Tensor], ...]


class TinyModelWithValueHead(nn.Module):
    """A tiny GRU causal LM with the same interface as ``ModelWithValueHead``."""

    def __init__(self, vocab_size: int = 99, hidden_size: int = 48) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def _forward_hidden(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attention_mask
        emb = self.embedding(input_ids)
        hidden, _ = self.rnn(emb)
        logits = self.lm_head(hidden)
        return logits, hidden

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden = self._forward_hidden(input_ids, attention_mask)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 1.0,
        return_dict_in_generate: bool = True,
        output_scores: bool = True,
        output_hidden_states: bool = True,
        **_: object,
    ) -> TinyGenerateOutput:
        del do_sample, temperature, top_p, return_dict_in_generate, output_scores, output_hidden_states
        tokenizer = TinyTokenizer()
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        scripted = _scripted_answer(prompt)
        scripted_ids = tokenizer.encode(scripted)[:max_new_tokens]
        if not scripted_ids:
            scripted_ids = [tokenizer.eos_token_id]

        seq = input_ids.clone()
        scores = []
        hidden_states = []
        for token_id in scripted_ids:
            logits, hidden = self._forward_hidden(seq, attention_mask=None)
            scores.append(logits[:, -1, :])
            hidden_states.append((hidden,))
            next_token = torch.tensor([[token_id]], dtype=torch.long, device=seq.device)
            seq = torch.cat([seq, next_token], dim=1)
        return TinyGenerateOutput(
            sequences=seq,
            scores=tuple(scores),
            hidden_states=tuple(hidden_states),
        )

    def gradient_checkpointing_enable(self, **_: object) -> None:
        return None


def is_tiny_model_path(model_path: str | None) -> bool:
    return str(model_path).lower() in {"sat:tiny", "sat/tiny", "tiny", "local:tiny"}


def _scripted_answer(prompt: str) -> str:
    numbers = [int(x) for x in re.findall(r"-?\d+", prompt)]
    if len(numbers) >= 2:
        a, b = numbers[0], numbers[1]
        if "+" in prompt:
            return f" Answer: {a + b}"
        if "*" in prompt or "times" in prompt.lower() or "product" in prompt.lower():
            return f" Answer: {a * b}"
    return " Answer: 0"
