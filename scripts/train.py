from __future__ import annotations

"""
학습 엔트리포인트 (M4 Max 최적화).

M4 Max의 Metal Performance Shaders (MPS)를 활용한 학습 스크립트.
Phase 4에서 본격 학습 루프/체크포인트/로깅을 구현합니다.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml


def setup_device(device_name: str = "auto") -> torch.device:
    """
    학습에 사용할 디바이스 설정
    
    Args:
        device_name: "auto", "mps", "cuda", "cpu" 중 하나
        
    Returns:
        torch.device 객체
    """
    if device_name == "auto":
        # 자동 감지: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🚀 M4 Max Metal Performance Shaders (MPS) 사용")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 CUDA 사용 (GPU: {torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("⚠️  CPU 사용 (학습 속도가 느릴 수 있습니다)")
    else:
        device = torch.device(device_name)
        print(f"🚀 디바이스: {device}")
    
    # MPS 메모리 최적화
    if device.type == "mps":
        try:
            # MPS 메모리의 80%까지 사용 (시스템 안정성 유지)
            torch.mps.set_per_process_memory_fraction(0.8)
            print("   MPS 메모리 제한: 80%")
        except Exception as e:
            print(f"   MPS 메모리 설정 실패 (무시): {e}")
    
    return device


def check_pytorch_version() -> None:
    """PyTorch 버전 확인 및 경고"""
    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    
    print(f"📦 PyTorch 버전: {version}")
    
    if major < 2:
        print("⚠️  경고: PyTorch 2.0+ 권장 (torch.compile 지원)")
    
    if torch.backends.mps.is_available():
        print("✅ MPS 백엔드 사용 가능")
        if major == 2 and minor < 1:
            print("⚠️  경고: PyTorch 2.1+ 권장 (MPS 안정성 향상)")
    else:
        print("❌ MPS 백엔드 사용 불가")


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    """설정 검증"""
    required_keys = ['seed', 'device', 'data', 'train']
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"필수 설정 키가 없습니다: {key}")
    
    # 학습 설정 검증
    train_cfg = cfg['train']
    required_train_keys = ['batch_size', 'learning_rate', 'max_steps']
    for key in required_train_keys:
        if key not in train_cfg:
            raise ValueError(f"필수 학습 설정 키가 없습니다: train.{key}")


def print_training_info(cfg: dict, model_cfg: dict, device: torch.device) -> None:
    """학습 정보 출력"""
    print("\n" + "="*60)
    print("📊 학습 설정 정보")
    print("="*60)
    
    # 모델 정보
    print(f"\n🤖 모델:")
    print(f"   - Hidden size: {model_cfg.get('hidden_size', 'N/A')}")
    print(f"   - Layers: {model_cfg.get('num_layers', 'N/A')}")
    print(f"   - Heads: {model_cfg.get('num_heads', 'N/A')}")
    print(f"   - Vocab size: {model_cfg.get('vocab_size', 'N/A')}")
    
    # 학습 설정
    train_cfg = cfg['train']
    print(f"\n🏋️  학습:")
    print(f"   - Batch size: {train_cfg.get('batch_size', 'N/A')}")
    print(f"   - Gradient accumulation: {train_cfg.get('gradient_accumulation_steps', 'N/A')}")
    effective_batch = train_cfg.get('batch_size', 0) * train_cfg.get('gradient_accumulation_steps', 1)
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {train_cfg.get('learning_rate', 'N/A')}")
    print(f"   - Max steps: {train_cfg.get('max_steps', 'N/A'):,}")
    print(f"   - Mixed precision: {train_cfg.get('mixed_precision', False)}")
    print(f"   - Gradient checkpointing: {train_cfg.get('gradient_checkpointing', False)}")
    print(f"   - Compile model: {train_cfg.get('compile_model', False)}")
    
    # 디바이스 정보
    print(f"\n💻 디바이스:")
    print(f"   - Device: {device}")
    if device.type == "mps":
        print(f"   - MPS 최적화: 활성화")
    
    # 데이터 정보
    data_cfg = cfg.get('data', {})
    print(f"\n📁 데이터:")
    print(f"   - Train: {data_cfg.get('train_path', 'N/A')}")
    print(f"   - Valid: {data_cfg.get('valid_path', 'N/A')}")
    print(f"   - Tokenizer: {data_cfg.get('tokenizer_model', 'N/A')}")
    
    print("\n" + "="*60 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="kr-mini-llm 학습 스크립트 (M4 Max 최적화)"
    )
    ap.add_argument(
        "--config",
        type=str,
        default="configs/training_m4max.yaml",
        help="학습 설정 파일 (YAML)"
    )
    ap.add_argument(
        "--model_config",
        type=str,
        default="configs/model_medium.yaml",
        help="모델 설정 파일 (YAML)"
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 (설정 파일의 값을 오버라이드)"
    )
    ap.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="체크포인트에서 학습 재개"
    )
    args = ap.parse_args()

    # PyTorch 버전 확인
    check_pytorch_version()
    
    # 설정 로드
    print(f"\n📄 설정 파일 로드 중...")
    try:
        cfg = load_config(args.config)
        model_cfg = load_config(args.model_config)
        print(f"   ✅ 학습 설정: {args.config}")
        print(f"   ✅ 모델 설정: {args.model_config}")
    except Exception as e:
        print(f"   ❌ 설정 로드 실패: {e}")
        sys.exit(1)
    
    # 설정 검증
    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"❌ 설정 검증 실패: {e}")
        sys.exit(1)
    
    # 디바이스 설정
    device = setup_device(cfg.get('device', 'auto'))
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg['train'].get('output_dir', 'checkpoints/'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 출력 디렉토리: {output_dir}")
    
    # 학습 정보 출력
    print_training_info(cfg, model_cfg, device)
    
    # Seed 설정
    seed = cfg.get('seed', 1337)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Random seed: {seed}")
    
    print("\n" + "="*60)
    print("⚠️  Phase 2-4 구현 필요")
    print("="*60)
    print("다음 단계:")
    print("1. src/model/* - 모델 아키텍처 구현 (RoPE, GQA, SwiGLU)")
    print("2. src/data/* - 데이터 파이프라인 구현")
    print("3. src/training/* - 학습 루프 구현")
    print("4. 이 스크립트에서 학습 루프 연결")
    print("="*60 + "\n")
    
    # 설정 저장
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"💾 설정 저장: {config_save_path}")
    
    model_config_save_path = output_dir / "model_config.yaml"
    with open(model_config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(model_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"💾 모델 설정 저장: {model_config_save_path}")


if __name__ == "__main__":
    main()


