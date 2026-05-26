#!/usr/bin/env python3
"""
간소화된 학습 스크립트 - 프로파일 기반

사용 예시:
  # 프로파일 사용 (권장)
  python scripts/train_simple.py --profile quick_test
  python scripts/train_simple.py --profile medium_full
  
  # 프로파일 + 오버라이드
  python scripts/train_simple.py --profile quick_test --device cpu
  python scripts/train_simple.py --profile medium_full --resume checkpoints/medium/step_10000.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.config import TransformerConfig
from src.model.transformer import TransformerLM
from src.data.tokenizer import KoreanTokenizer
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.training.optimizer import create_optimizer


def load_profile(profile_name: str) -> dict:
    """프로파일 로드"""
    profile_path = project_root / "configs" / "profiles" / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"프로파일을 찾을 수 없습니다: {profile_path}")
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="한국어 LLM 학습 (간소화 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 프로파일 기반
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="학습 프로파일 (quick_test, medium_full 등)"
    )
    
    # 선택적 오버라이드
    parser.add_argument("--device", type=str, help="디바이스 오버라이드")
    parser.add_argument("--resume", type=str, help="체크포인트 재개")
    parser.add_argument("--output_dir", type=str, help="출력 디렉토리 오버라이드")
    
    args = parser.parse_args()
    
    # 프로파일 로드
    print(f"📋 프로파일 로딩: {args.profile}")
    profile = load_profile(args.profile)
    
    # 설정 로드
    model_config_dict = load_config(profile['model']['config'])
    training_config = load_config(profile['training']['config'])
    
    # 오버라이드 적용
    device_str = args.device or profile.get('device', 'auto')
    output_dir = args.output_dir or profile['output']['dir']
    
    # 디바이스 설정
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
    
    # 토크나이저 로드
    tokenizer_path = profile['data']['tokenizer']
    print(f"🔤 토크나이저 로딩: {tokenizer_path}")
    tokenizer = KoreanTokenizer(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    
    # 모델 설정
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=model_config_dict.get('hidden_size', 1024),
        num_layers=model_config_dict.get('num_layers', 24),
        num_heads=model_config_dict.get('num_heads', 16),
        num_kv_heads=model_config_dict.get('num_kv_heads', 4),
        intermediate_size=model_config_dict.get('intermediate_size', 4096),
        max_seq_length=model_config_dict.get('max_seq_length', 2048),
        dropout=model_config_dict.get('dropout', 0.1),
        rope_theta=model_config_dict.get('rope_theta', 10000.0),
    )
    
    print(f"\n📊 모델 설정:")
    print(f"   - Vocab size: {model_config.vocab_size:,}")
    print(f"   - Hidden size: {model_config.hidden_size}")
    print(f"   - Layers: {model_config.num_layers}")
    print(f"   - Heads: {model_config.num_heads}")
    
    # 데이터로더 생성
    print(f"\n📦 데이터로더 생성...")
    train_loader, valid_loader = create_dataloaders(
        train_data=profile['data']['train'],
        valid_data=profile['data']['valid'],
        tokenizer=tokenizer,
        batch_size=training_config['train']['batch_size'],
        max_length=model_config.max_seq_length,
        num_workers=training_config['train'].get('num_workers', 4),
        use_tokenized=True,
    )
    
    print(f"   ✅ 학습 배치: {len(train_loader):,}")
    print(f"   ✅ 검증 배치: {len(valid_loader):,}")
    
    # 모델 생성
    print(f"\n🏗️  모델 생성...")
    model = TransformerLM(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - 총 파라미터: {total_params:,}")
    print(f"   - 크기: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Optimizer 생성
    train_cfg = training_config['train']
    optimizer = create_optimizer(
        model=model,
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # Trainer 생성
    print(f"\n🎯 Trainer 초기화...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=str(device),
        output_dir=output_dir,
        max_steps=train_cfg['max_steps'],
        eval_steps=train_cfg.get('eval_steps', 1000),
        save_steps=train_cfg.get('save_steps', 5000),
        logging_steps=train_cfg.get('logging_steps', 100),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        use_amp=train_cfg.get('mixed_precision', True),
        warmup_steps=train_cfg.get('warmup_steps', 2000),
        lr_scheduler_type=train_cfg.get('lr_scheduler', 'cosine'),
        resume_from=args.resume,
    )
    
    # 학습 시작
    print(f"\n🚀 학습 시작!")
    print(f"   - 프로파일: {args.profile}")
    print(f"   - 출력 디렉토리: {output_dir}")
    print(f"   - 배치 크기: {train_cfg['batch_size']}")
    print(f"   - Learning rate: {train_cfg['learning_rate']}")
    print()
    
    try:
        trainer.train()
        print("\n✅ 학습 완료!")
    except KeyboardInterrupt:
        print("\n⚠️  학습 중단됨")
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()

# Made with Bob
