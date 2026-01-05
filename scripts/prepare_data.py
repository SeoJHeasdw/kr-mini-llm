from __future__ import annotations

from pathlib import Path

from src.data import CharTokenizer


SAMPLE_KO = """\
안녕하세요. 저는 작은 한국어 LLM을 만들고 있어요.
오늘은 맥북 에어에서도 돌아가는 아주 작은 모델을 먼저 만들어봅니다.
모델은 완벽할 필요가 없고, 파이프라인이 끝까지 연결되는 게 목표예요.

한국어는 조사와 어미 변화가 많아서 데이터가 중요합니다.
하지만 지금은 샘플 텍스트로도 학습/저장/생성까지 잘 되는지만 확인해요.

내일은 토크나이저를 SentencePiece로 바꾸거나,
더 좋은 아키텍처(RoPE, RMSNorm, SwiGLU, GQA)로 발전시킬 수 있어요.
"""


def main() -> None:
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    model_dir = Path("models")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "sample_ko.txt"
    train_path = proc_dir / "train.txt"
    tok_path = model_dir / "tokenizer.json"

    raw_path.write_text(SAMPLE_KO, encoding="utf-8")
    train_path.write_text(SAMPLE_KO, encoding="utf-8")

    tok = CharTokenizer.train_from_text(SAMPLE_KO)
    tok.save(tok_path)

    print("✅ Prepared sample data + tokenizer")
    print(f"- raw: {raw_path}")
    print(f"- train: {train_path}")
    print(f"- tokenizer: {tok_path} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()


