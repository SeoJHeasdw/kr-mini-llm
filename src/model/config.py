from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """
    Transformer 언어모델 설정값.

    Phase 2에서 모델 구현 시 이 설정을 기준으로 레이어를 구성합니다.
    """

    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4  # GQA용
    intermediate_size: int = 2048  # SwiGLU용
    max_seq_length: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0


