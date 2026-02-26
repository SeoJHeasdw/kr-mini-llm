from __future__ import annotations

"""
Grouped Query Attention (GQA) 구현

GQA는 Multi-Head Attention의 효율적인 변형으로,
Key와 Value의 헤드 수를 줄여 메모리와 계산량을 절약합니다.

논문: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import TransformerConfig
from .layers import RoPE


@dataclass
class KVCache:
    """
    Key-Value 캐시 구조체
    
    추론 시 이전에 계산한 K, V를 재사용하여 속도를 향상시킵니다.
    """
    k: torch.Tensor  # (batch, seq_len, num_kv_heads, head_dim)
    v: torch.Tensor  # (batch, seq_len, num_kv_heads, head_dim)
    
    @staticmethod
    def empty(batch_size: int, max_seq_len: int, num_kv_heads: int, 
              head_dim: int, dtype: torch.dtype, device: torch.device) -> "KVCache":
        """빈 캐시 생성"""
        k = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim, 
                       dtype=dtype, device=device)
        v = torch.zeros(batch_size, max_seq_len, num_kv_heads, head_dim,
                       dtype=dtype, device=device)
        return KVCache(k=k, v=v)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    
    특징:
    - Query는 num_heads개의 헤드 사용
    - Key, Value는 num_kv_heads개의 헤드 사용 (num_kv_heads < num_heads)
    - 각 KV 헤드는 여러 Query 헤드와 공유됨
    - RoPE를 사용한 위치 인코딩
    - Causal masking (자기회귀 생성)
    """
    
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        
        assert cfg.hidden_size % cfg.num_heads == 0, \
            f"hidden_size ({cfg.hidden_size})는 num_heads ({cfg.num_heads})로 나누어떨어져야 합니다"
        
        assert cfg.num_heads % cfg.num_kv_heads == 0, \
            f"num_heads ({cfg.num_heads})는 num_kv_heads ({cfg.num_kv_heads})로 나누어떨어져야 합니다"
        
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.num_queries_per_kv = cfg.num_heads // cfg.num_kv_heads
        
        # Query projection: (hidden_size) -> (num_heads * head_dim)
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * self.head_dim, bias=False)
        
        # Key, Value projections: (hidden_size) -> (num_kv_heads * head_dim)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(cfg.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        
        # RoPE
        self.rope = RoPE(
            dim=self.head_dim,
            max_seq_length=cfg.max_seq_length,
            theta=cfg.rope_theta
        )
        
        # Dropout
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else None
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, seq_len, seq_len) 또는 None
            kv_cache: 이전 KV 캐시 (추론 시)
            use_cache: 캐시 사용 여부
            start_pos: 캐시 사용 시 시작 위치
            
        Returns:
            (output, new_kv_cache)
            - output: (batch, seq_len, hidden_size)
            - new_kv_cache: 업데이트된 캐시 (use_cache=True일 때만)
        """
        batch_size, seq_len, _ = x.shape
        
        # === 1. Q, K, V Projection ===
        # Q: (batch, seq_len, num_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # K, V: (batch, seq_len, num_kv_heads, head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # === 2. RoPE 적용 ===
        q, k = self.rope(q, k, start_pos=start_pos)
        
        # === 3. KV Cache 처리 ===
        if use_cache:
            if kv_cache is not None:
                # 기존 캐시에 새로운 K, V 추가
                k = torch.cat([kv_cache.k[:, :start_pos], k], dim=1)
                v = torch.cat([kv_cache.v[:, :start_pos], v], dim=1)
            
            # 새로운 캐시 생성
            new_kv_cache = KVCache(k=k, v=v)
        else:
            new_kv_cache = None
        
        # === 4. Repeat KV heads (GQA) ===
        # num_kv_heads를 num_heads로 확장
        # (batch, seq_len, num_kv_heads, head_dim) -> (batch, seq_len, num_heads, head_dim)
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        
        # === 5. Attention 계산 ===
        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # === 6. Causal Mask 적용 ===
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # 기본 causal mask (하삼각 행렬)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # === 7. Value와 곱하기 ===
        # (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # === 8. Reshape 및 Output Projection ===
        # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        output = self.o_proj(attn_output)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output, new_kv_cache


# Made with Bob
