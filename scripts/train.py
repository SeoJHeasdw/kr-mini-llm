#!/usr/bin/env python3
"""
모델 학습 스크립트

사용 예시:
  # 토큰화된 데이터로 학습 (권장)
  python scripts/train.py \\
      --model_config configs/model_medium.yaml \\
      --training_config configs/training_m4max.yaml \\
      --train_data data/processed/train_tokens.npy \\
      --valid_data data/processed/valid_tokens.npy \\
      --output_dir checkpoints/medium \\
      --device mps
  
  # 텍스트 파일로 학습 (느림, 권장하지 않음)
  python scripts/train.py \\
      --model_config configs/model_small.yaml \\
      --training_config configs/training.yaml \\
      --train_data data/processed/train.txt \\
      --valid_data data/processed/valid.txt \\
      --tokenizer models/tokenizer.model \\
      --output_dir checkpoints/small \\
      --device mps \\
      --no_tokenized
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.config import TransformerConfig
from src.model.transformer import TransformerLM
from src.data.tokenizer import KoreanTokenizer
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.training.optimizer import create_optimizer


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="한국어 LLM 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 필수 인자
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="모델 설정 파일 (YAML)"
    )
    parser.add_argument(
        "--training_config",
        type=str,
        required=True,
        help="학습 설정 파일 (YAML)"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="학습 데이터 파일 (.npy 또는 .txt)"
    )
    parser.add_argument(
        "--valid_data",
        type=str,
        required=True,
        help="검증 데이터 파일 (.npy 또는 .txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="체크포인트 저장 디렉토리"
    )
    
    # 선택 인자
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="토크나이저 모델 파일 (텍스트 파일 사용 시 필요)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="디바이스 (cuda, mps, cpu). 기본값: 자동 선택"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="재개할 체크포인트 경로"
    )
    parser.add_argument(
        "--no_tokenized",
        action="store_true",
        help="토큰화된 파일 대신 텍스트 파일 사용"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="스트리밍 모드 사용 (텍스트 파일만)"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    print("📋 설정 로딩...")
    model_config_dict = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    # 디바이스 설정
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"🖥️  디바이스: {device}")
    
    # 토크나이저 로드 (텍스트 파일 사용 시)
    tokenizer = None
    if args.no_tokenized or args.streaming:
        if not args.tokenizer:
            raise ValueError("텍스트 파일 사용 시 --tokenizer 필요")
        print(f"🔤 토크나이저 로딩: {args.tokenizer}")
        tokenizer = KoreanTokenizer(args.tokenizer)
        vocab_size = tokenizer.vocab_size
    else:
        # 토큰화된 파일 사용 시 vocab_size는 모델 설정에서 가져옴
        vocab_size = model_config_dict['vocab_size']
    
    # 모델 설정 (TransformerConfig 사용)
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        hidden_size=model_config_dict.get('d_model', model_config_dict.get('hidden_size', 1024)),
        num_layers=model_config_dict.get('n_layers', model_config_dict.get('num_layers', 24)),
        num_heads=model_config_dict.get('n_heads', model_config_dict.get('num_heads', 16)),
        num_kv_heads=model_config_dict.get('n_kv_heads', model_config_dict.get('num_kv_heads', 4)),
        intermediate_size=model_config_dict.get('d_ff', model_config_dict.get('intermediate_size', 4096)),
        max_seq_length=model_config_dict.get('max_seq_len', model_config_dict.get('max_seq_length', 2048)),
        dropout=model_config_dict.get('dropout', 0.1),
        rope_theta=model_config_dict.get('rope_theta', 10000.0),
    )
    
    print(f"\n📊 모델 설정:")
    print(f"   - Vocab size: {model_config.vocab_size:,}")
    print(f"   - Hidden size: {model_config.hidden_size}")
    print(f"   - Layers: {model_config.num_layers}")
    print(f"   - Heads: {model_config.num_heads}")
    print(f"   - KV heads: {model_config.num_kv_heads}")
    print(f"   - Max seq len: {model_config.max_seq_length}")
    
    # 데이터로더 생성
    print(f"\n📦 데이터로더 생성...")
    
    use_tokenized = not args.no_tokenized
    
    train_loader, valid_loader = create_dataloaders(
        train_data=args.train_data,
        valid_data=args.valid_data,
        tokenizer=tokenizer,
        batch_size=training_config.get('train', {}).get('batch_size', 16),
        max_length=model_config.max_seq_length,
        num_workers=training_config.get('train', {}).get('num_workers', 4),
        use_tokenized=use_tokenized,
        streaming=args.streaming,
    )
    
    print(f"   ✅ 학습 배치: {len(train_loader):,}")
    print(f"   ✅ 검증 배치: {len(valid_loader):,}")
    
    # 모델 생성
    print(f"\n🏗️  모델 생성...")
    model = TransformerLM(model_config)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   - 총 파라미터: {total_params:,}")
    print(f"   - 학습 가능: {trainable_params:,}")
    print(f"   - 크기: {total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Optimizer 생성
    print(f"\n⚙️  Optimizer 생성...")
    train_cfg = training_config.get('train', {})
    optimizer = create_optimizer(
        model=model,
        lr=train_cfg.get('learning_rate', 2e-4),
        weight_decay=train_cfg.get('weight_decay', 0.1),
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
        output_dir=args.output_dir,
        max_steps=train_cfg.get('max_steps', 100000),
        eval_steps=train_cfg.get('eval_interval', 1000),
        save_steps=train_cfg.get('save_interval', 5000),
        logging_steps=train_cfg.get('logging_steps', 100),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        use_amp=train_cfg.get('use_amp', True),
        warmup_steps=train_cfg.get('warmup_steps', 2000),
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine'),
        resume_from=args.resume,
    )
    
    # 학습 시작
    print(f"\n🚀 학습 시작!")
    print(f"   - 출력 디렉토리: {args.output_dir}")
    print(f"   - 배치 크기: {train_cfg.get('batch_size', 16)}")
    print(f"   - Gradient accumulation: {train_cfg.get('gradient_accumulation_steps', 1)}")
    print(f"   - 유효 배치 크기: {train_cfg.get('batch_size', 16) * train_cfg.get('gradient_accumulation_steps', 1)}")
    print(f"   - Learning rate: {train_cfg.get('learning_rate', 2e-4)}")
    print(f"   - Warmup steps: {train_cfg.get('warmup_steps', 2000)}")
    print(f"   - Mixed precision: {train_cfg.get('use_amp', True)}")
    print()
    
    try:
        trainer.train()
        print("\n✅ 학습 완료!")
    except KeyboardInterrupt:
        print("\n⚠️  학습 중단됨")
        print(f"   체크포인트 저장 위치: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()



