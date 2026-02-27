#!/usr/bin/env python3
"""
학습/검증 데이터 분할 스크립트

제대로 된 검증 데이터 생성:
- 랜덤 셔플로 편향 방지
- 적절한 비율 (기본 95:5)
- 데이터 누수 방지
"""

import argparse
import random
from pathlib import Path
from tqdm import tqdm


def split_data(
    input_file: str,
    train_file: str,
    valid_file: str,
    valid_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
):
    """
    텍스트 파일을 학습/검증 데이터로 분할
    
    Args:
        input_file: 입력 텍스트 파일
        train_file: 학습 데이터 출력 파일
        valid_file: 검증 데이터 출력 파일
        valid_ratio: 검증 데이터 비율 (0.0 ~ 1.0)
        seed: 랜덤 시드
        shuffle: 셔플 여부
    """
    print(f"📂 데이터 분할 시작")
    print(f"   - 입력: {input_file}")
    print(f"   - 검증 비율: {valid_ratio * 100:.1f}%")
    print(f"   - 셔플: {shuffle}")
    print(f"   - 시드: {seed}")
    
    # 랜덤 시드 설정
    random.seed(seed)
    
    # 1단계: 전체 라인 로드
    print("\n1️⃣ 데이터 로딩 중...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"   - 총 라인 수: {total_lines:,}")
    
    # 2단계: 셔플 (선택적)
    if shuffle:
        print("\n2️⃣ 데이터 셔플 중...")
        random.shuffle(lines)
        print(f"   ✅ 셔플 완료")
    
    # 3단계: 분할
    print("\n3️⃣ 데이터 분할 중...")
    valid_size = int(total_lines * valid_ratio)
    train_size = total_lines - valid_size
    
    train_lines = lines[:train_size]
    valid_lines = lines[train_size:]
    
    print(f"   - 학습 데이터: {len(train_lines):,} 라인 ({len(train_lines)/total_lines*100:.1f}%)")
    print(f"   - 검증 데이터: {len(valid_lines):,} 라인 ({len(valid_lines)/total_lines*100:.1f}%)")
    
    # 4단계: 저장
    print("\n4️⃣ 파일 저장 중...")
    
    # 학습 데이터 저장
    print(f"   - 학습 데이터 저장: {train_file}")
    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in tqdm(train_lines, desc="학습 데이터"):
            f.write(line + '\n')
    
    # 검증 데이터 저장
    print(f"   - 검증 데이터 저장: {valid_file}")
    Path(valid_file).parent.mkdir(parents=True, exist_ok=True)
    with open(valid_file, 'w', encoding='utf-8') as f:
        for line in tqdm(valid_lines, desc="검증 데이터"):
            f.write(line + '\n')
    
    # 파일 크기 확인
    train_size_mb = Path(train_file).stat().st_size / (1024 * 1024)
    valid_size_mb = Path(valid_file).stat().st_size / (1024 * 1024)
    
    print(f"\n✅ 분할 완료!")
    print(f"   - 학습 데이터: {train_size_mb:.1f} MB")
    print(f"   - 검증 데이터: {valid_size_mb:.1f} MB")
    print(f"\n💡 다음 단계:")
    print(f"   1. 학습 데이터 토큰화:")
    print(f"      python scripts/tokenize_data.py \\")
    print(f"          --input {train_file} \\")
    print(f"          --output {train_file.replace('.txt', '_tokens.npy')} \\")
    print(f"          --tokenizer models/tokenizer.model")
    print(f"\n   2. 검증 데이터 토큰화:")
    print(f"      python scripts/tokenize_data.py \\")
    print(f"          --input {valid_file} \\")
    print(f"          --output {valid_file.replace('.txt', '_tokens.npy')} \\")
    print(f"          --tokenizer models/tokenizer.model")


def main():
    parser = argparse.ArgumentParser(
        description="학습/검증 데이터 분할",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 분할 (95:5 비율)
  python scripts/split_data.py \\
      --input data/processed/train.txt \\
      --train_output data/processed/train_split.txt \\
      --valid_output data/processed/valid.txt
  
  # 커스텀 비율 (90:10)
  python scripts/split_data.py \\
      --input data/processed/train.txt \\
      --train_output data/processed/train_split.txt \\
      --valid_output data/processed/valid.txt \\
      --valid_ratio 0.1
  
  # 셔플 없이 (순차적 분할)
  python scripts/split_data.py \\
      --input data/processed/train.txt \\
      --train_output data/processed/train_split.txt \\
      --valid_output data/processed/valid.txt \\
      --no_shuffle
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 텍스트 파일"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        required=True,
        help="학습 데이터 출력 파일"
    )
    parser.add_argument(
        "--valid_output",
        type=str,
        required=True,
        help="검증 데이터 출력 파일"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.05,
        help="검증 데이터 비율 (기본값: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="셔플하지 않음 (순차적 분할)"
    )
    
    args = parser.parse_args()
    
    # 검증
    if not (0.0 < args.valid_ratio < 1.0):
        raise ValueError(f"valid_ratio는 0.0과 1.0 사이여야 합니다: {args.valid_ratio}")
    
    # 분할 실행
    split_data(
        input_file=args.input,
        train_file=args.train_output,
        valid_file=args.valid_output,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()


# Made with Bob