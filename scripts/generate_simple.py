#!/usr/bin/env python3
"""
간소화된 텍스트 생성 스크립트 - 프로파일 기반

사용 예시:
  # 프로파일 사용 (권장)
  python scripts/generate_simple.py --profile generate_default --prompt "안녕하세요"
  
  # 프로파일 + 오버라이드
  python scripts/generate_simple.py --profile generate_default --prompt "한국어는" --temperature 1.2
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.transformer import TransformerLM
from src.model.config import TransformerConfig
from src.inference.generator import Generator, GenerationConfig
from src.data.tokenizer import KoreanTokenizer


def load_profile(profile_name: str) -> dict:
    """프로파일 로드"""
    profile_path = project_root / "configs" / "profiles" / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"프로파일을 찾을 수 없습니다: {profile_path}")
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> TransformerLM:
    """체크포인트에서 모델 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    cfg = TransformerConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_layers=config_dict['num_layers'],
        num_heads=config_dict['num_heads'],
        num_kv_heads=config_dict['num_kv_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_seq_length=config_dict['max_seq_length'],
        dropout=config_dict.get('dropout', 0.0),
        rope_theta=config_dict.get('rope_theta', 10000.0),
    )
    
    model = TransformerLM(cfg)
    
    if Path(checkpoint_path).exists():
        print(f"✅ 체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"⚠️  체크포인트 없음 - 초기화된 모델 사용")
    
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="한국어 LLM 텍스트 생성 (간소화 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 프로파일 기반
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="생성 프로파일 (generate_default 등)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="입력 프롬프트"
    )
    
    # 선택적 오버라이드
    parser.add_argument("--temperature", type=float, help="샘플링 온도")
    parser.add_argument("--max-new-tokens", type=int, help="최대 생성 토큰 수")
    parser.add_argument("--top-p", type=float, help="Top-P 샘플링")
    parser.add_argument("--device", type=str, help="디바이스")
    
    args = parser.parse_args()
    
    # 프로파일 로드
    print(f"📋 프로파일 로딩: {args.profile}")
    profile = load_profile(args.profile)
    
    # 디바이스 설정
    device_str = args.device or profile.get('device', 'auto')
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    print(f"🖥️  디바이스: {device}")
    
    # 모델 로드
    model = load_model(
        profile['model']['checkpoint'],
        profile['model']['config'],
        device
    )
    
    # 토크나이저 로드
    tokenizer = KoreanTokenizer(profile['model']['tokenizer'])
    print(f"✅ 토크나이저 로드: {profile['model']['tokenizer']}")
    
    # Generator 생성
    generator = Generator(model, tokenizer, device)
    
    # 생성 설정 (오버라이드 적용)
    gen_cfg = profile['generation']
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens or gen_cfg['max_new_tokens'],
        temperature=args.temperature or gen_cfg['temperature'],
        top_p=args.top_p or gen_cfg.get('top_p'),
        repetition_penalty=gen_cfg.get('repetition_penalty', 1.0),
        do_sample=True
    )
    
    print(f"\n⚙️  생성 설정:")
    print(f"   - 최대 새 토큰: {gen_config.max_new_tokens}")
    print(f"   - Temperature: {gen_config.temperature}")
    if gen_config.top_p:
        print(f"   - Top-P: {gen_config.top_p}")
    
    # 프롬프트 출력
    print(f"\n📝 프롬프트: {args.prompt}")
    print()
    
    # 텍스트 생성
    print("🔄 생성 중...")
    try:
        generated_text = generator.generate(args.prompt, gen_config)
        
        print(f"\n✨ 생성 결과:")
        print("=" * 60)
        print(generated_text)
        print("=" * 60)
        print(f"\n✅ 생성 완료!")
        
    except Exception as e:
        print(f"\n❌ 생성 실패: {e}")
        raise


if __name__ == "__main__":
    main()

# Made with Bob
