from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/processed/train.txt", help="토크나이저 학습용 텍스트 파일")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--model_prefix", type=str, default="models/tokenizer")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 파일이 없습니다: {input_path}\n"
            f"- 먼저 데이터 준비 후 `data/processed/train.txt`를 만들어주세요."
        )

    out_prefix = Path(args.model_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # SentencePiece 학습 (기본 unigram; 필요 시 BPE로 변경 가능)
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(out_prefix),
        vocab_size=args.vocab_size,
        character_coverage=0.9995,  # 한국어 커버리지
        model_type="unigram",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        user_defined_symbols=[],
    )

    print("✅ 토크나이저 학습 완료")
    print(f"- model: {out_prefix}.model")
    print(f"- vocab: {out_prefix}.vocab")


if __name__ == "__main__":
    main()


