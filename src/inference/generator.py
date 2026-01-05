from __future__ import annotations

"""
텍스트 생성기(스켈레톤).

- Phase 5에서 greedy/top-k/top-p/temperature 샘플링을 구현합니다.
"""


class Generator:
    def __init__(self) -> None:
        pass

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Phase 5에서 Generator.generate()를 구현하세요.")


