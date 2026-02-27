#!/usr/bin/env python3
"""
모델 파라미터 수 계산 스크립트
"""

import sys
from pathlib import Path
import yaml

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import TransformerConfig
from src.model.transformer import TransformerLM


def calculate_params_theoretical(cfg: TransformerConfig) -> dict:
    """
    이론적 파라미터 수 계산 (모델 생성 없이)
    
    공식:
    - Embedding: vocab_size * hidden_size
    - Attention (per layer):
      * Q: hidden_size * hidden_size
      * K: hidden_size * (hidden_size * num_kv_heads / num_heads)
      * V: hidden_size * (hidden_size * num_kv_heads / num_heads)
      * O: hidden_size * hidden_size
    - FFN (per layer):
      * Gate: hidden_size * intermediate_size
      * Up: hidden_size * intermediate_size
      * Down: intermediate_size * hidden_size
    - Norm (per layer): hidden_size * 2 (attention_norm + ffn_norm)
    - Final Norm: hidden_size
    - LM Head: 0 (weight tying with embedding)
    """
    h = cfg.hidden_size
    n_layers = cfg.num_layers
    vocab = cfg.vocab_size
    n_heads = cfg.num_heads
    n_kv_heads = cfg.num_kv_heads
    intermediate = cfg.intermediate_size
    
    # Embedding (weight tying으로 lm_head와 공유)
    embedding = vocab * h
    
    # Attention (per layer)
    # GQA: K, V는 num_kv_heads만큼만 사용
    kv_dim = h * n_kv_heads // n_heads
    attn_per_layer = (
        h * h +           # Q projection
        h * kv_dim +      # K projection
        h * kv_dim +      # V projection
        h * h             # O projection
    )
    attention_total = attn_per_layer * n_layers
    
    # FFN (per layer) - SwiGLU
    ffn_per_layer = (
        h * intermediate +  # Gate projection
        h * intermediate +  # Up projection
        intermediate * h    # Down projection
    )
    ffn_total = ffn_per_layer * n_layers
    
    # Normalization
    norm_per_layer = h * 2  # attention_norm + ffn_norm
    norm_total = norm_per_layer * n_layers + h  # + final norm
    
    # 총합
    total = embedding + attention_total + ffn_total + norm_total
    
    return {
        'total': total,
        'embedding': embedding,
        'attention': attention_total,
        'ffn': ffn_total,
        'norm': norm_total,
        'lm_head': 0,  # weight tying
        'per_layer': {
            'attention': attn_per_layer,
            'ffn': ffn_per_layer,
            'norm': norm_per_layer
        }
    }


def format_number(num: int) -> str:
    """숫자를 읽기 쉬운 형식으로 변환"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def print_params_info(params: dict, config_name: str):
    """파라미터 정보 출력"""
    print(f"\n{'='*60}")
    print(f"📊 모델 파라미터 수: {config_name}")
    print(f"{'='*60}")
    
    print(f"\n🎯 총 파라미터 수: {format_number(params['total'])} ({params['total']:,})")
    print(f"\n📦 컴포넌트별 파라미터:")
    print(f"   - Embedding:     {format_number(params['embedding']):>8} ({params['embedding']:>12,})")
    print(f"   - Attention:     {format_number(params['attention']):>8} ({params['attention']:>12,})")
    print(f"   - FFN:           {format_number(params['ffn']):>8} ({params['ffn']:>12,})")
    print(f"   - Normalization: {format_number(params['norm']):>8} ({params['norm']:>12,})")
    print(f"   - LM Head:       {format_number(params['lm_head']):>8} (weight tying)")
    
    print(f"\n🔍 레이어당 파라미터:")
    print(f"   - Attention:     {format_number(params['per_layer']['attention']):>8} ({params['per_layer']['attention']:>12,})")
    print(f"   - FFN:           {format_number(params['per_layer']['ffn']):>8} ({params['per_layer']['ffn']:>12,})")
    print(f"   - Normalization: {format_number(params['per_layer']['norm']):>8} ({params['per_layer']['norm']:>12,})")
    
    print(f"\n{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 파라미터 수 계산")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_medium.yaml",
        help="모델 설정 파일"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 모델 설정 비교"
    )
    parser.add_argument(
        "--actual",
        action="store_true",
        help="실제 모델 생성하여 파라미터 수 계산 (느림)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # 모든 모델 설정 비교
        configs = [
            ("Small (150M)", "configs/model_small.yaml"),
            ("Medium (350M)", "configs/model_medium.yaml"),
            ("Large (800M)", "configs/model_large.yaml")
        ]
        
        print("\n" + "="*60)
        print("📊 모델 크기 비교")
        print("="*60)
        
        for name, config_path in configs:
            if not Path(config_path).exists():
                print(f"\n⚠️  {name}: 설정 파일 없음 ({config_path})")
                continue
            
            with open(config_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            
            cfg = TransformerConfig(**cfg_dict)
            params = calculate_params_theoretical(cfg)
            
            print(f"\n{name}:")
            print(f"   총 파라미터: {format_number(params['total'])}")
            print(f"   - Hidden: {cfg.hidden_size}, Layers: {cfg.num_layers}")
            print(f"   - Heads: {cfg.num_heads}, KV Heads: {cfg.num_kv_heads}")
            print(f"   - Intermediate: {cfg.intermediate_size}")
        
        print("\n" + "="*60 + "\n")
    
    else:
        # 단일 모델 상세 정보
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        cfg = TransformerConfig(**cfg_dict)
        
        if args.actual:
            # 실제 모델 생성하여 계산
            print("🔨 모델 생성 중...")
            model = TransformerLM(cfg)
            params = model.count_parameters()
            print("✅ 모델 생성 완료")
        else:
            # 이론적 계산 (빠름)
            params = calculate_params_theoretical(cfg)
        
        print_params_info(params, config_path.stem)
        
        # 설정 정보 출력
        print("⚙️  모델 설정:")
        print(f"   - Vocab size: {cfg.vocab_size:,}")
        print(f"   - Hidden size: {cfg.hidden_size}")
        print(f"   - Num layers: {cfg.num_layers}")
        print(f"   - Num heads: {cfg.num_heads}")
        print(f"   - Num KV heads: {cfg.num_kv_heads} (GQA)")
        print(f"   - Intermediate size: {cfg.intermediate_size}")
        print(f"   - Max seq length: {cfg.max_seq_length}")
        print(f"   - Dropout: {cfg.dropout}")
        print()


if __name__ == "__main__":
    main()

# Made with Bob
