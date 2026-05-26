"""
KULLM-v2 데이터셋 다운로드 및 전처리
Hugging Face에서 한국어 instruction 데이터셋 다운로드
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_and_prepare_kullm(
    output_dir: str = "data/kullm",
    split: str = "train",
    max_samples: int = None,
    text_column: str = "text"
):
    """
    KULLM-v2 데이터셋 다운로드 및 텍스트 파일로 저장
    
    Args:
        output_dir: 출력 디렉토리
        split: 데이터셋 split (train/validation/test)
        max_samples: 최대 샘플 수 (None이면 전체)
        text_column: 텍스트 컬럼 이름
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 KULLM-v2 데이터셋 다운로드 중...")
    print(f"   - Split: {split}")
    print(f"   - 출력 디렉토리: {output_dir}")
    
    try:
        # 데이터셋 로드
        ds = load_dataset("nlpai-lab/kullm-v2", split=split)
        print(f"✅ 데이터셋 로드 완료: {len(ds):,} 샘플")
        
        # 샘플 수 제한
        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))
            print(f"   - 샘플 수 제한: {max_samples:,}")
        
        # 데이터셋 구조 확인
        print(f"\n📊 데이터셋 구조:")
        print(f"   - 컬럼: {ds.column_names}")
        if len(ds) > 0:
            print(f"   - 첫 번째 샘플 키: {list(ds[0].keys())}")
        
        # 텍스트 추출 및 저장
        output_file = output_path / f"kullm_{split}.txt"
        print(f"\n💾 텍스트 파일 저장 중: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(tqdm(ds, desc="처리 중")):
                # instruction + output 형식으로 결합
                if 'instruction' in example and 'output' in example:
                    text = f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['output']}\n\n"
                elif text_column in example:
                    text = example[text_column] + "\n\n"
                elif 'text' in example:
                    text = example['text'] + "\n\n"
                else:
                    # 모든 필드를 결합
                    text = " ".join(str(v) for v in example.values() if v) + "\n\n"
                
                f.write(text)
        
        # 통계 출력
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✅ 완료!")
        print(f"   - 저장된 샘플: {len(ds):,}")
        print(f"   - 파일 크기: {file_size_mb:.2f} MB")
        print(f"   - 파일 경로: {output_file}")
        
        # 샘플 출력
        print(f"\n📝 샘플 미리보기:")
        with open(output_file, 'r', encoding='utf-8') as f:
            sample = f.read(500)
            print("=" * 60)
            print(sample)
            print("=" * 60)
        
        return output_file
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="KULLM-v2 데이터셋 다운로드",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/kullm",
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="데이터셋 split"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="최대 샘플 수 (테스트용)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="텍스트 컬럼 이름"
    )
    
    args = parser.parse_args()
    
    output_file = download_and_prepare_kullm(
        output_dir=args.output_dir,
        split=args.split,
        max_samples=args.max_samples,
        text_column=args.text_column
    )
    
    if output_file:
        print(f"\n🎯 다음 단계:")
        print(f"1. 토크나이저 학습:")
        print(f"   python scripts/train_tokenizer.py --input {output_file} --vocab-size 32000")
        print(f"\n2. 데이터 토크나이징:")
        print(f"   python scripts/tokenize_data.py --input {output_file}")
        print(f"\n3. 모델 학습:")
        print(f"   python scripts/train.py --config configs/training_m4max.yaml")


if __name__ == "__main__":
    main()

# Made with Bob
