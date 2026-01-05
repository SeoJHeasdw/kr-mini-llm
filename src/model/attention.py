from __future__ import annotations

"""
Attention 구현(스켈레톤).

- Phase 2에서 GQA(Grouped Query Attention) 및 KV cache 등을 구현합니다.
"""

import torch
import torch.nn as nn

from .config import TransformerConfig


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # TODO: q/k/v projection + 출력 projection 구현
        # TODO: RoPE 적용
        # TODO: causal mask 및 KV cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2에서 GQA를 구현하세요.")


