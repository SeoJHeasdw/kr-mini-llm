from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.data import CharTokenizer
from src.model import TinyGPT


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/tiny_gpt.pt")
    ap.add_argument("--tokenizer", type=str, default=None, help="defaults to <checkpoint>.tokenizer.json")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--config", type=str, default="configs/training_tiny.yaml", help="optional YAML to fill defaults")
    args = ap.parse_args()

    prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k: Optional[int] = args.top_k

    # allow pulling defaults from the same yaml used for training
    cfg_path = Path(args.config)
    if cfg_path.exists():
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        gen = payload.get("generate", {}) or {}
        if prompt is None:
            prompt = gen.get("prompt", "안녕하세요")
        if max_new_tokens is None:
            max_new_tokens = int(gen.get("max_new_tokens", 120))
        if temperature is None:
            temperature = float(gen.get("temperature", 0.9))
        if top_k is None and gen.get("top_k", None) is not None:
            top_k = int(gen.get("top_k"))
    else:
        prompt = prompt or "안녕하세요"
        max_new_tokens = max_new_tokens or 120
        temperature = temperature or 0.9

    ckpt = Path(args.checkpoint)
    tok_path = Path(args.tokenizer) if args.tokenizer else ckpt.with_suffix(".tokenizer.json")
    device = pick_device(args.device)

    tok = CharTokenizer.load(tok_path)
    model = TinyGPT.load(ckpt, map_location=device).to(device)

    x = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out = tok.decode(y[0].tolist())
    print(out)


if __name__ == "__main__":
    main()


