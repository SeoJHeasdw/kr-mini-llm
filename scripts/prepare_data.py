from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="원본 텍스트(.txt) 디렉토리")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="전처리 결과 디렉토리")
    ap.add_argument("--train_out", type=str, default="train.txt")
    ap.add_argument("--valid_out", type=str, default="valid.txt")
    ap.add_argument("--valid_ratio", type=float, default=0.01)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"{raw_dir}에 .txt 파일이 없습니다.\n"
            f"- 예: data/raw/corpus_001.txt\n"
            f"- 준비 후 다시 `python scripts/prepare_data.py`를 실행하세요."
        )

    # 아주 단순한 Phase 1 전처리: 파일 합치기(정제/중복 제거/문장 분리는 Phase 3에서 확장)
    all_text = []
    for p in txt_files:
        all_text.append(p.read_text(encoding="utf-8", errors="ignore"))
    merged = "\n".join(all_text).strip() + "\n"

    # 매우 단순한 split: 라인 단위로 나눠 검증셋을 일부 떼어냄
    lines = [ln for ln in merged.splitlines() if ln.strip()]
    n_valid = max(1, int(len(lines) * float(args.valid_ratio)))
    valid_lines = lines[:n_valid]
    train_lines = lines[n_valid:]

    (out_dir / args.train_out).write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / args.valid_out).write_text("\n".join(valid_lines) + "\n", encoding="utf-8")

    print("✅ 데이터 준비 완료(Phase 1 수준: 합치기 + 단순 split)")
    print(f"- raw_dir: {raw_dir} (files={len(txt_files)})")
    print(f"- train: {out_dir / args.train_out} (lines={len(train_lines)})")
    print(f"- valid: {out_dir / args.valid_out} (lines={len(valid_lines)})")


if __name__ == "__main__":
    main()


