#!/usr/bin/env python3
"""
텍스트 데이터를 토큰화하여 바이너리 파일로 저장

한 번만 토큰화하고 저장해두면 학습 시 빠르게 로드 가능
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.tokenizer import KoreanTokenizer


def tokenize_and_save(
    input_file: str,
    output_file: str,
    tokenizer: KoreanTokenizer,
    max_length: int = 2048,
    stride: int = 1024,
):
    """
    텍스트 파일을 토큰화하여 numpy 배열로 저장
    
    Args:
        input_file: 입력 텍스트 파일
        output_file: 출력 numpy 파일 (.npy)
        tokenizer: 토크나이저
        max_length: 최대 시퀀스 길이
        stride: 슬라이딩 윈도우 스트라이드
    """
    print(f"📖 토큰화 시작: {input_file}")
    print(f"   - 최대 길이: {max_length}")
    print(f"   - 스트라이드: {stride}")
    
    all_tokens = []
    total_lines = 0
    total_tokens = 0
    
    # 1단계: 전체 라인 수 계산
    print("\n1️⃣ 파일 크기 확인 중...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"   - 총 라인 수: {total_lines:,}")
    
    # 2단계: 토큰화
    print("\n2️⃣ 토큰화 진행 중...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="토큰화"):
            line = line.strip()
            if not line:
                continue
            
            # 토큰화
            tokens = tokenizer.encode(line)
            total_tokens += len(tokens)
            
            # 슬라이딩 윈도우로 청크 생성
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 10:  # 최소 길이
                    all_tokens.append(chunk)
    
    print(f"\n✅ 토큰화 완료!")
    print(f"   - 총 토큰 수: {total_tokens:,}")
    print(f"   - 시퀀스 수: {len(all_tokens):,}")
    
    # 3단계: numpy 배열로 저장
    print(f"\n3️⃣ 저장 중: {output_file}")
    
    # 가변 길이 배열을 object 타입으로 저장
    tokens_array = np.array(all_tokens, dtype=object)
    np.save(output_file, tokens_array)
    
    # 파일 크기 확인 (.npy 파일 확장자 추가)
    npy_file = Path(str(output_file) + '.npy')
    if npy_file.exists():
        file_size = npy_file.stat().st_size / (1024 * 1024)
        print(f"   - 파일 크기: {file_size:.1f} MB")
    
    print(f"\n🎉 완료! 다음 학습 시 빠르게 로드됩니다.")


def main():
    parser = argparse.ArgumentParser(
        description="텍스트 데이터를 토큰화하여 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 학습 데이터 토큰화
  python scripts/tokenize_data.py \\
      --input data/processed/train.txt \\
      --output data/processed/train_tokens.npy \\
      --tokenizer models/tokenizer.model
  
  # 검증 데이터 토큰화
  python scripts/tokenize_data.py \\
      --input data/processed/valid.txt \\
      --output data/processed/valid_tokens.npy \\
      --tokenizer models/tokenizer.model
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 텍스트 파일"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 numpy 파일 (.npy)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="토크나이저 모델 파일"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이 (기본값: 2048)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
        help="슬라이딩 윈도우 스트라이드 (기본값: 1024)"
    )
    
    args = parser.parse_args()
    
    # 토크나이저 로드
    print("🔤 토크나이저 로딩...")
    tokenizer = KoreanTokenizer(args.tokenizer)
    print(f"   - Vocab size: {tokenizer.vocab_size:,}")
    
    # 토큰화 및 저장
    tokenize_and_save(
        input_file=args.input,
        output_file=args.output,
        tokenizer=tokenizer,
        max_length=args.max_length,
        stride=args.stride
    )


if __name__ == "__main__":
    main()


