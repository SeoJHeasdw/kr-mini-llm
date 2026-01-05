from __future__ import annotations

"""
학습 엔트리포인트 (스켈레톤).

Phase 4에서 본격 학습 루프/체크포인트/로깅을 구현합니다.
현재 Phase 1에서는 "구조"만 제공합니다.
"""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/training.yaml", help="학습 설정(YAML)")
    ap.add_argument("--model_config", type=str, default="configs/model_small.yaml", help="모델 설정(YAML)")
    ap.add_argument("--output_dir", type=str, default="checkpoints/", help="출력 디렉토리")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(Path(args.model_config).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("✅ 학습 스크립트(스켈레톤) 실행")
    print("- 다음 단계: src/model/*, src/training/* 구현 후 이 스크립트에서 학습 루프를 연결하세요.")
    print(f"- training_config: {args.config}")
    print(f"- model_config: {args.model_config}")
    print(f"- output_dir: {out}")
    print(f"- cfg keys: {list(cfg.keys())}")
    print(f"- model cfg keys: {list(model_cfg.keys())}")


if __name__ == "__main__":
    main()


