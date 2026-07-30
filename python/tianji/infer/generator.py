"""Token generator driving the QAT model with incremental Mamba-2 state.

Pre-allocates a fixed-size buffer so the autoregressive loop never calls
``torch.cat``, eliminating O(N) allocation per token. Uses Mamba-2 state
carry for incremental decoding: the full prompt is processed once, then
each subsequent token is fed individually with the carried state, making
inference ~10x faster for long sequences.

Supports temperature, top-k, and top-p (nucleus) sampling for diverse
and controllable generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    """Sampling parameters for token generation."""
    temperature: float = 1.0
    top_k: int = 0       # 0 = disabled
    top_p: float = 0.0   # 0.0 = disabled


@dataclass
class GenerateConfig:
    max_tokens: int = 16
    paged_kv_blocks: int = 8
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class GenerateStep:
    token: int


def _sample_token(logits: torch.Tensor, cfg: SamplingConfig) -> int:
    """Sample a token from logits using temperature + top-k + top-p.

    Args:
        logits: Raw logits of shape (..., vocab_size).
        cfg: Sampling configuration.

    Returns:
        Sampled token index.
    """
    if cfg.temperature <= 0:
        return int(logits.argmax(dim=-1).item())

    # Apply temperature
    logits = logits / cfg.temperature

    # Top-k filtering
    if cfg.top_k > 0:
        k = min(cfg.top_k, logits.shape[-1])
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = torch.where(logits < threshold, float("-inf"), logits)

    # Top-p (nucleus) filtering
    if 0.0 < cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above top_p
        sorted_idx_to_remove = cum_probs > cfg.top_p
        # Shift so we keep at least one token
        sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
        sorted_idx_to_remove[..., 0] = False
        # Scatter sorted indices back to original order
        idx_to_remove = sorted_idx_to_remove.scatter(
            -1, sorted_idx, sorted_idx_to_remove)
        logits = logits.masked_fill(idx_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


class Generator:
    def __init__(self, qat, cfg: GenerateConfig):
        self.qat = qat
        self.cfg = cfg
        self.model = qat.model
        self.device = qat.device
        self._buf: torch.Tensor | None = None
        self._mamba_state: Tuple[torch.Tensor, ...] | None = None
        self._engine = None  # set by StateAwareGenerator

    def _ensure_buf(self, n: int) -> torch.Tensor:
        needed = n + self.cfg.max_tokens + 1
        if self._buf is None or needed > self._buf.shape[1]:
            self._buf = torch.zeros(
                1, needed, dtype=torch.long, device=self.device)
        return self._buf

    def _next_token(self, logits: torch.Tensor) -> int:
        """Sample next token using configured sampling strategy."""
        return _sample_token(logits, self.cfg.sampling)
    def __init__(self, qat, cfg: GenerateConfig):
        self.qat = qat
        self.cfg = cfg
        self.model = qat.model
        self.device = qat.device
        self._buf: torch.Tensor | None = None
        self._mamba_state: Tuple[torch.Tensor, ...] | None = None

    def _ensure_buf(self, n: int) -> torch.Tensor:
        needed = n + self.cfg.max_tokens + 1
        if self._buf is None or needed > self._buf.shape[1]:
            self._buf = torch.zeros(
                1, needed, dtype=torch.long, device=self.device)
        return self._buf

    def _next_token(self, logits: torch.Tensor) -> int:
        """Sample next token using configured sampling strategy."""
        return _sample_token(logits, self.cfg.sampling)

    def generate(self, prompt_ids) -> Iterator[GenerateStep]:
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(self.device).unsqueeze(0)
        plen = ids.shape[1]
        buf = self._ensure_buf(plen)
        buf[:, :plen].copy_(ids)
        cur_len = plen

        # ── First step: full forward on the prompt, cache Mamba-2 state ──
        with torch.no_grad():
            h = self.model.embed(buf[:, :cur_len])
            h, _, next_state = self.model.stack(h)
            logits = self.model.head(h)
        nxt = self._next_token(logits[0, -1])
        yield GenerateStep(token=nxt)
        buf[0, cur_len] = nxt
        cur_len += 1
        self._mamba_state = next_state

        # ── Incremental steps: feed only the last token with cached state ──
        for _ in range(1, self.cfg.max_tokens):
            last = buf[:, cur_len - 1:cur_len]
            with torch.no_grad():
                h = self.model.embed(last)
                h, _, next_state = self.model.stack(h, state_tuple=self._mamba_state)
                logits = self.model.head(h)
            nxt = self._next_token(logits[0, -1])
            yield GenerateStep(token=nxt)
            buf[0, cur_len] = nxt
            cur_len += 1
            self._mamba_state = next_state

    def reset_state(self) -> None:
        """Clear cached Mamba-2 state for a new generation."""
        self._mamba_state = None


class StateAwareGenerator(Generator):
    """Generator that uses state head predictions to bias token output.

    The state head predicts the next event kind (tool_call, cot, etc.).
    This biases the token distribution toward special tokens matching
    the predicted kind, making outputs more agent-like.
    """
    def __init__(self, qat, cfg: GenerateConfig, engine=None):
        super().__init__(qat, cfg)
        self._engine = engine
        self._kind_to_bias: dict[int, list[int]] = {}
        if engine is not None:
            from ..tokens.apt import SPECIAL_IDS
            from ..state.transition import EVENT_KINDS, KIND_TO_IDX
            kind_tags = {
                "system_prompt": ["<system>"],
                "cot": ["<cot>"],
                "tool_call": ["<tool_call>"],
                "tool_result": ["<bash_output>"],
                "context": [],
                "exec_trace": [],
                "trace_end": [],
            }
            for kind, tags in kind_tags.items():
                if kind in KIND_TO_IDX:
                    idx = KIND_TO_IDX[kind]
                    self._kind_to_bias[idx] = [SPECIAL_IDS[t] for t in tags if t in SPECIAL_IDS]

    def _next_token(self, logits: torch.Tensor) -> int:
        if self._engine is not None and self._engine._prev_state is not None:
            action = self._engine.predict_next_kind(self._engine._prev_state)
            bias_ids = self._kind_to_bias.get(action, [])
            for tid in bias_ids:
                if tid < logits.shape[-1]:
                    logits[..., tid] += 2.0
        return _sample_token(logits, self.cfg.sampling)