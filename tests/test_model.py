"""
모델 컴포넌트 테스트

각 컴포넌트가 올바르게 작동하는지 검증합니다.
"""

import torch
import pytest
from pathlib import Path
import sys

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.config import TransformerConfig
from src.model.layers import RoPE, RMSNorm, SwiGLU
from src.model.attention import GroupedQueryAttention, KVCache
from src.model.transformer import TransformerBlock, TransformerLM


class TestRoPE:
    """RoPE 테스트"""
    
    def test_rope_initialization(self):
        """RoPE 초기화 테스트"""
        rope = RoPE(dim=64, max_seq_length=128)
        assert rope.dim == 64
        assert rope.max_seq_length == 128
        assert rope.cos_cached.shape == (128, 64)
        assert rope.sin_cached.shape == (128, 64)
    
    def test_rope_forward(self):
        """RoPE forward pass 테스트"""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        
        rope = RoPE(dim=head_dim, max_seq_length=128)
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q, q_rot)  # 회전이 적용되었는지 확인


class TestRMSNorm:
    """RMSNorm 테스트"""
    
    def test_rmsnorm_initialization(self):
        """RMSNorm 초기화 테스트"""
        norm = RMSNorm(dim=512)
        assert norm.weight.shape == (512,)
        assert torch.allclose(norm.weight, torch.ones(512))
    
    def test_rmsnorm_forward(self):
        """RMSNorm forward pass 테스트"""
        batch_size, seq_len, dim = 2, 10, 512
        
        norm = RMSNorm(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = norm(x)
        
        assert output.shape == x.shape
        
        # RMS가 대략 1이 되는지 확인
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestSwiGLU:
    """SwiGLU FFN 테스트"""
    
    def test_swiglu_initialization(self):
        """SwiGLU 초기화 테스트"""
        ffn = SwiGLU(hidden_size=512, intermediate_size=2048)
        assert ffn.gate_proj.in_features == 512
        assert ffn.gate_proj.out_features == 2048
        assert ffn.up_proj.in_features == 512
        assert ffn.up_proj.out_features == 2048
        assert ffn.down_proj.in_features == 2048
        assert ffn.down_proj.out_features == 512
    
    def test_swiglu_forward(self):
        """SwiGLU forward pass 테스트"""
        batch_size, seq_len, hidden_size = 2, 10, 512
        
        ffn = SwiGLU(hidden_size=hidden_size, intermediate_size=2048)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        output = ffn(x)
        
        assert output.shape == x.shape


class TestGroupedQueryAttention:
    """GQA 테스트"""
    
    def test_gqa_initialization(self):
        """GQA 초기화 테스트"""
        cfg = TransformerConfig.small()
        attn = GroupedQueryAttention(cfg)
        
        assert attn.num_heads == cfg.num_heads
        assert attn.num_kv_heads == cfg.num_kv_heads
        assert attn.head_dim == cfg.hidden_size // cfg.num_heads
    
    def test_gqa_forward(self):
        """GQA forward pass 테스트"""
        cfg = TransformerConfig.small()
        attn = GroupedQueryAttention(cfg)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, cfg.hidden_size)
        
        output, kv_cache = attn(x, use_cache=False)
        
        assert output.shape == x.shape
        assert kv_cache is None
    
    def test_gqa_with_cache(self):
        """KV cache 사용 테스트"""
        cfg = TransformerConfig.small()
        attn = GroupedQueryAttention(cfg)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, cfg.hidden_size)
        
        output, kv_cache = attn(x, use_cache=True)
        
        assert output.shape == x.shape
        assert kv_cache is not None
        assert kv_cache.k.shape == (batch_size, seq_len, cfg.num_kv_heads, 
                                    cfg.hidden_size // cfg.num_heads)


class TestTransformerBlock:
    """Transformer Block 테스트"""
    
    def test_block_initialization(self):
        """Block 초기화 테스트"""
        cfg = TransformerConfig.small()
        block = TransformerBlock(cfg, layer_idx=0)
        
        assert block.layer_idx == 0
        assert isinstance(block.attention, GroupedQueryAttention)
        assert isinstance(block.ffn, SwiGLU)
    
    def test_block_forward(self):
        """Block forward pass 테스트"""
        cfg = TransformerConfig.small()
        block = TransformerBlock(cfg, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, cfg.hidden_size)
        
        output, kv_cache = block(x, use_cache=False)
        
        assert output.shape == x.shape
        assert kv_cache is None


class TestTransformerLM:
    """전체 모델 테스트"""
    
    def test_model_initialization(self):
        """모델 초기화 테스트"""
        cfg = TransformerConfig.small()
        model = TransformerLM(cfg)
        
        assert len(model.layers) == cfg.num_layers
        assert model.token_embedding.num_embeddings == cfg.vocab_size
        assert model.token_embedding.embedding_dim == cfg.hidden_size
    
    def test_model_forward(self):
        """모델 forward pass 테스트"""
        cfg = TransformerConfig.small()
        model = TransformerLM(cfg)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        logits, kv_caches = model(input_ids, use_cache=False)
        
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
        assert kv_caches is None
    
    def test_model_with_cache(self):
        """KV cache 사용 테스트"""
        cfg = TransformerConfig.small()
        model = TransformerLM(cfg)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        logits, kv_caches = model(input_ids, use_cache=True)
        
        assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
        assert kv_caches is not None
        assert len(kv_caches) == cfg.num_layers
    
    def test_parameter_count(self):
        """파라미터 수 계산 테스트"""
        cfg = TransformerConfig.small()
        model = TransformerLM(cfg)
        
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert params['total'] > 0
        assert params['trainable'] == params['total']
        
        # 대략적인 파라미터 수 확인 (small 모델: ~100M)
        assert 90_000_000 < params['total'] < 110_000_000
    
    def test_generation(self):
        """텍스트 생성 테스트"""
        cfg = TransformerConfig.small()
        model = TransformerLM(cfg)
        model.eval()
        
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        
        # 짧은 생성 테스트
        output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len + 10


class TestModelConfigs:
    """모델 설정 테스트"""
    
    def test_small_config(self):
        """Small 설정 테스트"""
        cfg = TransformerConfig.small()
        assert cfg.hidden_size == 768
        assert cfg.num_layers == 12
        assert cfg.num_heads == 12
    
    def test_medium_config(self):
        """Medium 설정 테스트"""
        cfg = TransformerConfig.medium()
        assert cfg.hidden_size == 1024
        assert cfg.num_layers == 24
        assert cfg.num_heads == 16
        
        # 파라미터 수 확인 (~468M)
        assert 400_000_000 < cfg.num_parameters < 500_000_000
    
    def test_large_config(self):
        """Large 설정 테스트"""
        cfg = TransformerConfig.large()
        assert cfg.hidden_size == 1536
        assert cfg.num_layers == 24
        assert cfg.num_heads == 24
        
        # 파라미터 수 확인 (~1004M)
        assert 950_000_000 < cfg.num_parameters < 1_050_000_000


def test_full_pipeline():
    """전체 파이프라인 통합 테스트"""
    print("\n=== 전체 파이프라인 테스트 ===")
    
    # Small 모델로 테스트
    cfg = TransformerConfig.small()
    print(f"\n설정:\n{cfg}")
    
    model = TransformerLM(cfg)
    params = model.count_parameters()
    
    print(f"\n파라미터 수:")
    print(f"  Total: {params['total']:,}")
    print(f"  Embedding: {params['embedding']:,}")
    print(f"  Attention: {params['attention']:,}")
    print(f"  FFN: {params['ffn']:,}")
    print(f"  Norm: {params['norm']:,}")
    
    # Forward pass 테스트
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    print(f"\n입력 shape: {input_ids.shape}")
    
    # 학습 모드
    model.train()
    logits, _ = model(input_ids)
    print(f"출력 logits shape: {logits.shape}")
    
    # Loss 계산 테스트
    target = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        target.view(-1)
    )
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass 테스트
    loss.backward()
    print("✓ Backward pass 성공")
    
    # 생성 테스트
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids[:1, :5],
            max_new_tokens=20,
            temperature=0.8,
            top_k=50
        )
    print(f"생성된 토큰 shape: {generated.shape}")
    
    print("\n✓ 전체 파이프라인 테스트 성공!")


if __name__ == "__main__":
    # pytest 없이 직접 실행
    test_full_pipeline()
    
    print("\n=== 개별 컴포넌트 테스트 ===")
    
    # RoPE
    print("\n[RoPE 테스트]")
    test_rope = TestRoPE()
    test_rope.test_rope_initialization()
    test_rope.test_rope_forward()
    print("✓ RoPE 테스트 통과")
    
    # RMSNorm
    print("\n[RMSNorm 테스트]")
    test_norm = TestRMSNorm()
    test_norm.test_rmsnorm_initialization()
    test_norm.test_rmsnorm_forward()
    print("✓ RMSNorm 테스트 통과")
    
    # SwiGLU
    print("\n[SwiGLU 테스트]")
    test_ffn = TestSwiGLU()
    test_ffn.test_swiglu_initialization()
    test_ffn.test_swiglu_forward()
    print("✓ SwiGLU 테스트 통과")
    
    # GQA
    print("\n[GQA 테스트]")
    test_attn = TestGroupedQueryAttention()
    test_attn.test_gqa_initialization()
    test_attn.test_gqa_forward()
    test_attn.test_gqa_with_cache()
    print("✓ GQA 테스트 통과")
    
    # Transformer Block
    print("\n[Transformer Block 테스트]")
    test_block = TestTransformerBlock()
    test_block.test_block_initialization()
    test_block.test_block_forward()
    print("✓ Transformer Block 테스트 통과")
    
    # 전체 모델
    print("\n[TransformerLM 테스트]")
    test_model = TestTransformerLM()
    test_model.test_model_initialization()
    test_model.test_model_forward()
    test_model.test_model_with_cache()
    test_model.test_parameter_count()
    test_model.test_generation()
    print("✓ TransformerLM 테스트 통과")
    
    # 설정
    print("\n[Config 테스트]")
    test_config = TestModelConfigs()
    test_config.test_small_config()
    test_config.test_medium_config()
    test_config.test_large_config()
    print("✓ Config 테스트 통과")
    
    print("\n" + "="*50)
    print("모든 테스트 통과! 🎉")
    print("="*50)


# Made with Bob