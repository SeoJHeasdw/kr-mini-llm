#!/usr/bin/env python3
"""
데이터 준비 스크립트
HAERAE-HUB/KOREAN-WEBTEXT 데이터셋을 다운로드하고 전처리합니다.

M4 Max 36GB 메모리 최적화:
- 스트리밍 방식으로 메모리 효율적 처리
- 청크 단위 저장으로 대용량 데이터 처리
"""

import os
import argparse
from pathlib import Path
from typing import Iterator, Dict, List, Optional
import json
from tqdm import tqdm


def download_korean_webtext(
    output_dir: str = "data/raw",
    max_samples: Optional[int] = None,
    streaming: bool = True
) -> None:
    """
    HAERAE-HUB/KOREAN-WEBTEXT 데이터셋 다운로드
    
    Args:
        output_dir: 저장 디렉토리
        max_samples: 최대 샘플 수 (None이면 전체)
        streaming: 스트리밍 모드 사용 여부 (메모리 절약)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets 라이브러리가 필요합니다:")
        print("   pip install datasets")
        return
    
    print("📥 KOREAN-WEBTEXT 데이터셋 다운로드 중...")
    print(f"   출처: https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 로드 (스트리밍 모드로 메모리 절약)
    dataset = load_dataset(
        "HAERAE-HUB/KOREAN-WEBTEXT",
        streaming=streaming,
        split="train"
    )
    
    # 텍스트 파일로 저장
    output_file = output_path / "korean_webtext.txt"
    
    print(f"💾 텍스트 추출 중... → {output_file}")
    
    count = 0
    total_chars = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="처리 중"):
            # 텍스트 추출 (필드명은 데이터셋 구조에 따라 조정 필요)
            text = example.get('text', '') or example.get('content', '')
            
            if text and len(text.strip()) > 50:  # 최소 길이 필터
                f.write(text.strip() + '\n\n')
                count += 1
                total_chars += len(text)
            
            # 최대 샘플 수 제한
            if max_samples and count >= max_samples:
                break
    
    print(f"✅ 다운로드 완료!")
    print(f"   - 문서 수: {count:,}")
    print(f"   - 총 문자 수: {total_chars:,}")
    print(f"   - 예상 크기: {total_chars / 1024 / 1024:.1f} MB")
    print(f"   - 저장 위치: {output_file}")


def prepare_data_from_raw(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    train_out: str = "train.txt",
    valid_out: str = "valid.txt",
    valid_ratio: float = 0.01,
    chunk_size: int = 100000
) -> None:
    """
    원본 텍스트 파일을 학습/검증 데이터로 분할 (메모리 효율적)
    
    Args:
        raw_dir: 원본 텍스트 디렉토리
        out_dir: 출력 디렉토리
        train_out: 학습 데이터 파일명
        valid_out: 검증 데이터 파일명
        valid_ratio: 검증 데이터 비율
        chunk_size: 청크 크기 (라인 수)
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # .txt 파일 찾기
    txt_files = sorted(raw_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"❌ {raw_path}에 .txt 파일이 없습니다.\n"
            f"   먼저 데이터를 다운로드하세요:\n"
            f"   python scripts/prepare_data.py --download"
        )
    
    print(f"📂 원본 파일 {len(txt_files)}개 발견")
    print(f"💾 메모리 효율적 처리 모드 (청크 크기: {chunk_size:,} 라인)")
    
    # 1단계: 총 라인 수 계산
    print("\n1️⃣ 총 라인 수 계산 중...")
    total_lines = 0
    for txt_file in txt_files:
        print(f"   카운팅: {txt_file.name}")
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    total_lines += 1
    
    print(f"📊 총 라인 수: {total_lines:,}")
    
    # 검증 데이터 크기 계산
    n_valid = max(1, int(total_lines * valid_ratio))
    n_train = total_lines - n_valid
    
    print(f"   - 학습: {n_train:,} 라인 ({(1-valid_ratio)*100:.1f}%)")
    print(f"   - 검증: {n_valid:,} 라인 ({valid_ratio*100:.1f}%)")
    
    # 2단계: 청크 단위로 읽어서 분할 저장
    print("\n2️⃣ 데이터 분할 중...")
    train_file = out_path / train_out
    valid_file = out_path / valid_out
    
    line_count = 0
    train_chars = 0
    valid_chars = 0
    
    with open(train_file, 'w', encoding='utf-8') as f_train, \
         open(valid_file, 'w', encoding='utf-8') as f_valid:
        
        for txt_file in txt_files:
            print(f"   처리 중: {txt_file.name}")
            
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                for line in tqdm(f_in, desc=f"   {txt_file.name}", leave=False):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 검증 데이터에 먼저 할당
                    if line_count < n_valid:
                        f_valid.write(line + '\n')
                        valid_chars += len(line)
                    else:
                        f_train.write(line + '\n')
                        train_chars += len(line)
                    
                    line_count += 1
    
    print(f"\n✅ 데이터 준비 완료!")
    print(f"   - 학습: {train_file}")
    print(f"     • 라인 수: {n_train:,}")
    print(f"     • 크기: {train_chars / 1024 / 1024:.1f} MB")
    print(f"   - 검증: {valid_file}")
    print(f"     • 라인 수: {n_valid:,}")
    print(f"     • 크기: {valid_chars / 1024 / 1024:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="한국어 LLM 학습 데이터 준비",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 1. KOREAN-WEBTEXT 다운로드
  python scripts/prepare_data.py --download
  
  # 2. 샘플 수 제한하여 다운로드 (테스트용)
  python scripts/prepare_data.py --download --max_samples 10000
  
  # 3. 학습/검증 데이터 분할
  python scripts/prepare_data.py --prepare
  
  # 4. 한번에 실행
  python scripts/prepare_data.py --download --prepare
        """
    )
    
    # 다운로드 옵션
    parser.add_argument(
        "--download",
        action="store_true",
        help="KOREAN-WEBTEXT 데이터셋 다운로드"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 샘플 수 (테스트용, 기본값: 전체)"
    )
    
    # 전처리 옵션
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="학습/검증 데이터 분할"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="원본 텍스트 디렉토리"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--train_out",
        type=str,
        default="train.txt",
        help="학습 데이터 파일명"
    )
    parser.add_argument(
        "--valid_out",
        type=str,
        default="valid.txt",
        help="검증 데이터 파일명"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.01,
        help="검증 데이터 비율 (기본값: 0.01 = 1%%)"
    )
    
    args = parser.parse_args()
    
    # 아무 옵션도 없으면 도움말 표시
    if not (args.download or args.prepare):
        parser.print_help()
        return
    
    # 다운로드
    if args.download:
        download_korean_webtext(
            output_dir=args.raw_dir,
            max_samples=args.max_samples,
            streaming=True
        )
    
    # 전처리
    if args.prepare:
        prepare_data_from_raw(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            train_out=args.train_out,
            valid_out=args.valid_out,
            valid_ratio=args.valid_ratio
        )


if __name__ == "__main__":
    main()


