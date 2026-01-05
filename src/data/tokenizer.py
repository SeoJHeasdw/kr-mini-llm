from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from sentencepiece import SentencePieceProcessor


class KoreanTokenizer:
    """
    SentencePiece 기반 한국어 토크나이저 래퍼.

    - Phase 2에서 `scripts/train_tokenizer.py`로 학습한 `models/tokenizer.model`을 로드해서 사용합니다.
    - 여기서는 "구조"만 제공하고, 전처리/정규화/스페셜 토큰 정책은 이후 단계에서 확정합니다.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.sp = SentencePieceProcessor()
        if not self.sp.load(str(self.model_path)):
            raise FileNotFoundError(f"SentencePiece 모델을 로드하지 못했습니다: {self.model_path}")

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str) -> List[int]:
        # TODO: 필요하면 한국어 정규화(띄어쓰기/반각/특수문자) 정책을 추가하세요.
        return list(self.sp.encode(text, out_type=int))

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.sp.decode(list(ids)))


