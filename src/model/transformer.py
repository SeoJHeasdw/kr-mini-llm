from __future__ import annotations

"""
Transformer 언어모델 구현

최신 아키텍처 특징:
- RoPE (Rotary Position Embedding)
- GQA (Grouped Query Attention)
- SwiGLU Feed-Forward Network
- RMSNorm
- Pre-normalization (Norm before attention/FFN)
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .config import TransformerConfig
from .attention import GroupedQueryAttention, KVCache
from .layers import RMSNorm, SwiGLU, init_weights


class TransformerBlock(nn.Module):
    """
    단일 Transformer 블록
    
    구조:
    1. RMSNorm -> Attention -> Residual
    2. RMSNorm -> FFN -> Residual
    """
    
    def __init__(self, cfg: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-normalization
        self.attention_norm = RMSNorm(cfg.hidden_size)
        self.ffn_norm = RMSNorm(cfg.hidden_size)
        
        # Attention
        self.attention = GroupedQueryAttention(cfg)
        
        # Feed-Forward Network
        self.ffn = SwiGLU(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            dropout=cfg.dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            kv_cache: KV cache for this layer
            use_cache: Whether to use/return cache
            start_pos: Starting position for cache
            
        Returns:
            (output, new_kv_cache)
        """
        # === 1. Attention Block ===
        # Pre-norm
        normed = self.attention_norm(x)
        
        # Attention
        attn_output, new_kv_cache = self.attention(
            normed,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            use_cache=use_cache,
            start_pos=start_pos
        )
        
        # Residual connection
        x = x + attn_output
        
        # === 2. FFN Block ===
        # Pre-norm
        normed = self.ffn_norm(x)
        
        # FFN
        ffn_output = self.ffn(normed)
        
        # Residual connection
        x = x + ffn_output
        
        return x, new_kv_cache


class TransformerLM(nn.Module):
    """
    Transformer 언어모델
    
    전체 구조:
    1. Token Embedding
    2. N개의 Transformer Blocks
    3. Final RMSNorm
    4. LM Head (vocab projection)
    """
    
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        
        # Token embedding
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i)
            for i in range(cfg.num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(cfg.hidden_size)
        
        # LM head (output projection)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        
        # Weight tying (embedding과 lm_head 가중치 공유)
        self.lm_head.weight = self.token_embedding.weight
        
        # 가중치 초기화
        self.apply(lambda m: init_weights(m, std=0.02))
        
        # 특별 초기화: residual projection에 대해 스케일링
        for layer in self.layers:
            # Attention output projection
            nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, 
                          std=0.02 / (2 * cfg.num_layers) ** 0.5)
            # FFN down projection
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0,
                          std=0.02 / (2 * cfg.num_layers) ** 0.5)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> tuple[torch.Tensor, Optional[List[KVCache]]]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) 토큰 ID
            attention_mask: Attention mask
            kv_caches: List of KV caches for each layer
            use_cache: Whether to use/return caches
            start_pos: Starting position for cache
            
        Returns:
            (logits, new_kv_caches)
            - logits: (batch, seq_len, vocab_size)
            - new_kv_caches: Updated caches (if use_cache=True)
        """
        # === 1. Token Embedding ===
        x = self.token_embedding(input_ids)  # (batch, seq_len, hidden_size)
        
        # === 2. Transformer Blocks ===
        new_kv_caches = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            # 해당 레이어의 캐시 가져오기
            layer_cache = kv_caches[i] if kv_caches is not None else None
            
            # Forward through layer
            x, new_cache = layer(
                x,
                attention_mask=attention_mask,
                kv_cache=layer_cache,
                use_cache=use_cache,
                start_pos=start_pos
            )
            
            if use_cache:
                new_kv_caches.append(new_cache)
        
        # === 3. Final Norm ===
        x = self.norm(x)
        
        # === 4. LM Head ===
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits, new_kv_caches
    
    def count_parameters(self) -> dict[str, int]:
        """
        모델의 파라미터 수 계산
        
        Returns:
            파라미터 수 딕셔너리
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 컴포넌트별 파라미터 수
        embedding = sum(p.numel() for p in self.token_embedding.parameters())
        
        attention = sum(
            sum(p.numel() for p in layer.attention.parameters())
            for layer in self.layers
        )
        
        ffn = sum(
            sum(p.numel() for p in layer.ffn.parameters())
            for layer in self.layers
        )
        
        norm = sum(
            sum(p.numel() for p in layer.attention_norm.parameters()) +
            sum(p.numel() for p in layer.ffn_norm.parameters())
            for layer in self.layers
        ) + sum(p.numel() for p in self.norm.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'embedding': embedding,
            'attention': attention,
            'ffn': ffn,
            'norm': norm,
            'lm_head': 0  # weight tying으로 embedding과 공유
        }
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        파라미터 수 반환 (GPT-2 스타일)
        
        Args:
            non_embedding: True면 embedding 제외
            
        Returns:
            파라미터 수
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        자기회귀 텍스트 생성
        
        Args:
            input_ids: (batch, seq_len) 시작 토큰들
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도 (높을수록 다양함)
            top_k: Top-K 샘플링
            top_p: Nucleus (Top-P) 샘플링
            eos_token_id: 종료 토큰 ID
            
        Returns:
            (batch, seq_len + max_new_tokens) 생성된 토큰들
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Context가 너무 길면 자르기
            idx_cond = input_ids if input_ids.size(1) <= self.cfg.max_seq_length \
                      else input_ids[:, -self.cfg.max_seq_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # 마지막 토큰의 logits만 사용
            logits = logits[:, -1, :] / temperature
            
            # Top-K 샘플링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-P (nucleus) 샘플링
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # 누적 확률이 top_p를 초과하는 토큰 제거
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # 샘플링
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 생성된 토큰 추가
            input_ids = torch.cat([input_ids, idx_next], dim=1)
            
            # EOS 토큰 체크
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break
        
        return input_ids


# Made with Bob
