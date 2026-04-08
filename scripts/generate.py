"""
텍스트 생성 스크립트
Korean LLM을 사용한 텍스트 생성
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import yaml

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.transformer import TransformerLM
from src.model.config import TransformerConfig
from src.inference.generator import Generator, GenerationConfig
from src.data.tokenizer import KoreanTokenizer


def load_model(checkpoint_path: str, config_path: str, device: torch.device) -> TransformerLM:
    """
    체크포인트에서 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        config_path: 설정 파일 경로
        device: 실행 디바이스
        
    Returns:
        로드된 모델
    """
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # TransformerConfig 생성
    model_config = config_dict.get('model', {})
    cfg = TransformerConfig(**model_config)
    
    # 모델 생성
    model = TransformerLM(cfg)
    
    # 체크포인트 로드
    if os.path.exists(checkpoint_path):
        print(f"✅ 체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # state_dict 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"   - 에폭: {checkpoint.get('epoch', 'N/A')}")
        if 'loss' in checkpoint:
            print(f"   - 손실: {checkpoint['loss']:.4f}")
    else:
        print(f"⚠️  체크포인트 없음: {checkpoint_path}")
        print("   - 초기화된 모델 사용 (테스트용)")
    
    model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Korean LLM 텍스트 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 모델 관련
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final.pt",
        help="체크포인트 파일 경로"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_small.yaml",
        help="모델 설정 파일 경로"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/korean_tokenizer.json",
        help="토크나이저 파일 경로"
    )
    
    # 생성 관련
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="입력 프롬프트"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="최대 생성 길이"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="최대 생성 토큰 수"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="샘플링 온도 (0.1=거의 greedy, 1.0=균형, 2.0=매우 다양함)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K 샘플링 (예: 50)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-P (nucleus) 샘플링 (예: 0.9)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="반복 페널티 (1.0=없음, 1.2=약간 억제, 1.5=강하게 억제)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search 빔 개수 (1=샘플링, >1=beam search)"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="샘플링 비활성화 (greedy decoding)"
    )
    
    # 기타
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="실행 디바이스"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (재현성을 위해)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 랜덤 시드: {args.seed}")
    
    # 디바이스 설정
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Korean LLM 텍스트 생성")
    print(f"   - 디바이스: {device}")
    print(f"   - 설정: {args.config}")
    print()
    
    # 모델 로드
    try:
        model = load_model(args.checkpoint, args.config, device)
        
        # 모델 정보 출력
        if args.verbose:
            param_counts = model.count_parameters()
            print(f"\n📊 모델 정보:")
            print(f"   - 총 파라미터: {param_counts['total']:,}")
            print(f"   - Embedding: {param_counts['embedding']:,}")
            print(f"   - Attention: {param_counts['attention']:,}")
            print(f"   - FFN: {param_counts['ffn']:,}")
            print()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 토크나이저 로드
    try:
        if os.path.exists(args.tokenizer):
            tokenizer = KoreanTokenizer.load(args.tokenizer)
            print(f"✅ 토크나이저 로드: {args.tokenizer}")
        else:
            print(f"⚠️  토크나이저 없음: {args.tokenizer}")
            print("   - 간단한 문자 기반 토크나이저 사용")
            tokenizer = None
    except Exception as e:
        print(f"⚠️  토크나이저 로드 실패: {e}")
        print("   - 간단한 문자 기반 토크나이저 사용")
        tokenizer = None
    
    # Generator 생성
    generator = Generator(model, tokenizer, device)
    
    # 생성 설정
    gen_config = GenerationConfig(
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        do_sample=not args.no_sample
    )
    
    # 생성 설정 출력
    print(f"\n⚙️  생성 설정:")
    print(f"   - 최대 새 토큰: {gen_config.max_new_tokens}")
    print(f"   - Temperature: {gen_config.temperature}")
    if gen_config.top_k:
        print(f"   - Top-K: {gen_config.top_k}")
    if gen_config.top_p:
        print(f"   - Top-P: {gen_config.top_p}")
    if gen_config.repetition_penalty != 1.0:
        print(f"   - Repetition penalty: {gen_config.repetition_penalty}")
    if gen_config.num_beams > 1:
        print(f"   - Beam search: {gen_config.num_beams} beams")
    print(f"   - 샘플링: {'활성화' if gen_config.do_sample else '비활성화 (greedy)'}")
    print()
    
    # 프롬프트 출력
    print(f"📝 프롬프트:")
    print(f"   {args.prompt}")
    print()
    
    # 텍스트 생성
    print("🔄 생성 중...")
    try:
        generated_text = generator.generate(args.prompt, gen_config)
        
        print(f"\n✨ 생성 결과:")
        print("=" * 60)
        print(generated_text)
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 생성 실패: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return
    
    print(f"\n✅ 생성 완료!")


if __name__ == "__main__":
    main()



