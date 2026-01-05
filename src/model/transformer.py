from __future__ import annotations

"""
Transformer 언어모델(스켈레톤).

- Phase 2에서 전체 모델을 통합합니다.
- 목표: forward pass가 정상 동작하고, 더미 데이터로 1 step 학습이 가능하도록 만들기.
"""

import torch
import torch.nn as nn

from .config import TransformerConfig


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # TODO: 임베딩/포지션/블록/헤드 구성

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        입력:
          - input_ids: (B, T) 토큰 ID
        출력:
          - logits: (B, T, vocab_size)
        """
        raise NotImplementedError("Phase 2에서 TransformerLM을 구현하세요.")


