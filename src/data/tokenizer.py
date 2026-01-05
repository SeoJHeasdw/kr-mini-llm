from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class CharTokenizer:
    """
    Ultra-simple character-level tokenizer (UTF-8 safe via Python str).
    Good enough for an end-to-end smoke test on a MacBook Air.
    """

    stoi: Dict[str, int]
    itos: List[str]
    unk_token: str = "<unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def train_from_text(cls, text: str, *, add_unk: bool = True) -> "CharTokenizer":
        chars = sorted(set(text))
        itos: List[str] = []
        if add_unk:
            itos.append("<unk>")
        itos.extend(chars)
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        unk_id = self.stoi.get(self.unk_token, 0)
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append(self.unk_token)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"itos": self.itos, "unk_token": self.unk_token}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        itos: List[str] = payload["itos"]
        unk_token: str = payload.get("unk_token", "<unk>")
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, unk_token=unk_token)


