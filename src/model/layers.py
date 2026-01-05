from __future__ import annotations

"""
모델 레이어 모음(스켈레톤).

- Phase 2에서 RoPE, RMSNorm 등을 구현합니다.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm (TODO: Phase 2에서 구현)"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 2에서 RMSNorm을 구현하세요.")


class RotaryPositionEmbedding(nn.Module):
    """RoPE (TODO: Phase 2에서 구현)"""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Phase 2에서 RoPE를 구현하세요.")


