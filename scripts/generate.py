from __future__ import annotations

"""
추론/생성 엔트리포인트 (스켈레톤).

Phase 5에서 `src/inference/generator.py` 및 체크포인트 로딩을 구현합니다.
현재 Phase 1에서는 "구조"만 제공합니다.
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True, help="프롬프트")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/final", help="체크포인트 경로(예정)")
    args = ap.parse_args()

    print("✅ 생성 스크립트(스켈레톤) 실행")
    print("- 다음 단계: 모델/토크나이저/샘플링 구현 후 여기서 generate를 연결하세요.")
    print(f"- checkpoint: {args.checkpoint}")
    print(f"- prompt: {args.prompt}")


if __name__ == "__main__":
    main()


