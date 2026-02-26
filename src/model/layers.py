from __future__ import annotations

"""
Transformer 핵심 레이어 구현

- RoPE (Rotary Position Embedding)
- RMSNorm (Root Mean Square Layer Normalization)
- SwiGLU (Swish-Gated Linear Unit)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    논문: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    RoPE는 절대 위치 정보를 상대적 위치 정보로 변환하여
    긴 시퀀스에서도 효과적인 위치 인코딩을 제공합니다.
    """
    
    def __init__(self, dim: int, max_seq_length: int = 2048, theta: float = 10000.0):
        """
        Args:
            dim: 헤드당 차원 (hidden_size // num_heads)
            max_seq_length: 최대 시퀀스 길이
            theta: RoPE의 기본 주파수 (기본값: 10000.0)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        
        # 사전 계산된 cos, sin 값 (캐싱)
        self._build_cache(max_seq_length)
    
    def _build_cache(self, seq_length: int):
        """cos, sin 캐시 생성"""
        # 주파수 계산: theta^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # 위치 인덱스: [0, 1, 2, ..., seq_length-1]
        t = torch.arange(seq_length, dtype=torch.float32)
        
        # 외적: (seq_length, dim/2)
        freqs = torch.outer(t, inv_freq)
        
        # cos, sin 계산 및 복제 (dim/2 -> dim)
        # (seq_length, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # 버퍼로 등록 (학습되지 않는 파라미터)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, 
                start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query와 Key에 RoPE 적용
        
        Args:
            q: Query tensor (batch, seq_len, num_heads, head_dim)
            k: Key tensor (batch, seq_len, num_kv_heads, head_dim)
            start_pos: KV cache 사용 시 시작 위치
            
        Returns:
            (rotated_q, rotated_k)
        """
        seq_len = q.shape[1]
        
        # 캐시 확장 필요 시
        if start_pos + seq_len > self.max_seq_length:
            self._build_cache(start_pos + seq_len)
        
        # 해당 위치의 cos, sin 가져오기
        cos = self.cos_cached[start_pos:start_pos + seq_len]
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        
        # (seq_len, dim) -> (1, seq_len, 1, dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # RoPE 적용
        q_rot = self._apply_rotary_emb(q, cos, sin)
        k_rot = self._apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    @staticmethod
    def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, 
                          sin: torch.Tensor) -> torch.Tensor:
        """
        실제 회전 변환 적용
        
        x를 복소수로 간주하여 회전:
        x_rot = x * cos + rotate_half(x) * sin
        """
        # x를 두 부분으로 나누기
        x1, x2 = x.chunk(2, dim=-1)
        
        # 회전 적용
        # [x1, x2] -> [-x2, x1] (90도 회전)
        x_rotated = torch.cat([-x2, x1], dim=-1)
        
        # 최종 회전: x * cos + x_rotated * sin
        return x * cos + x_rotated * sin


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    논문: "Root Mean Square Layer Normalization"
    
    LayerNorm보다 계산이 간단하면서도 비슷한 성능을 보입니다.
    평균을 빼는 대신 RMS로 정규화합니다.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: 정규화할 차원
            eps: 수치 안정성을 위한 작은 값
        """
        super().__init__()
        self.eps = eps
        # 학습 가능한 스케일 파라미터
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            
        Returns:
            정규화된 텐서
        """
        # RMS 계산: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 정규화 및 스케일 적용
        x_normed = x / rms
        return self.weight * x_normed


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) Feed-Forward Network
    
    논문: "GLU Variants Improve Transformer"
    
    표준 FFN보다 성능이 좋으며, LLaMA 등 최신 모델에서 사용됩니다.
    
    구조:
    FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        """
        Args:
            hidden_size: 입력/출력 차원
            intermediate_size: 중간 차원 (보통 4 * hidden_size)
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # Gate projection (for Swish activation)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # Up projection (parallel to gate)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # Down projection (back to hidden_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            
        Returns:
            (batch, seq_len, hidden_size)
        """
        # Gate: Swish(xW_gate)
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish
        
        # Up: xW_up
        up = self.up_proj(x)
        
        # Element-wise multiplication
        hidden = gate * up
        
        # Down projection
        output = self.down_proj(hidden)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output


def init_weights(module: nn.Module, std: float = 0.02):
    """
    가중치 초기화 (GPT-2 스타일)
    
    Args:
        module: 초기화할 모듈
        std: 표준편차
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)


# Made with Bob
