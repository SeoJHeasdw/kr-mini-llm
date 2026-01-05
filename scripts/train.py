from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import trange

from src.data import CharTokenizer
from src.model import TinyGPT, TinyGPTConfig


@dataclass
class TrainCfg:
    seed: int = 1337
    device: str = "auto"  # auto|cpu|mps|cuda

    train_path: str = "data/processed/train.txt"
    tokenizer_path: str = "models/tokenizer.json"

    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    batch_size: int = 16
    max_steps: int = 300
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 25

    save_path: str = "checkpoints/tiny_gpt.pt"


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: Optional[str]) -> TrainCfg:
    cfg = TrainCfg()
    if not path:
        return cfg

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.seed = int(payload.get("seed", cfg.seed))
    cfg.device = str(payload.get("device", cfg.device))

    data = payload.get("data", {}) or {}
    cfg.train_path = str(data.get("train_path", cfg.train_path))
    cfg.tokenizer_path = str(data.get("tokenizer_path", cfg.tokenizer_path))

    model = payload.get("model", {}) or {}
    cfg.block_size = int(model.get("block_size", cfg.block_size))
    cfg.n_layer = int(model.get("n_layer", cfg.n_layer))
    cfg.n_head = int(model.get("n_head", cfg.n_head))
    cfg.n_embd = int(model.get("n_embd", cfg.n_embd))
    cfg.dropout = float(model.get("dropout", cfg.dropout))

    train = payload.get("train", {}) or {}
    cfg.batch_size = int(train.get("batch_size", cfg.batch_size))
    cfg.max_steps = int(train.get("max_steps", cfg.max_steps))
    cfg.learning_rate = float(train.get("learning_rate", cfg.learning_rate))
    cfg.weight_decay = float(train.get("weight_decay", cfg.weight_decay))
    cfg.grad_clip = float(train.get("grad_clip", cfg.grad_clip))
    cfg.log_every = int(train.get("log_every", cfg.log_every))
    cfg.save_path = str(train.get("save_path", cfg.save_path))
    return cfg


def get_batch(data: torch.Tensor, *, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # sample random starting positions
    max_start = data.size(0) - block_size - 1
    ix = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/training_tiny.yaml", help="YAML config path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    train_text = Path(cfg.train_path).read_text(encoding="utf-8")
    tok = CharTokenizer.load(cfg.tokenizer_path)
    ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    if ids.numel() < cfg.block_size + 2:
        raise ValueError("Training text is too small; add more lines to data/processed/train.txt")

    model_cfg = TinyGPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = TinyGPT(model_cfg).to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    running = 0.0
    pbar = trange(cfg.max_steps, desc=f"train ({device})")
    for step in pbar:
        x, y = get_batch(ids, block_size=cfg.block_size, batch_size=cfg.batch_size, device=device)
        _, loss = model(x, y)
        assert loss is not None

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        running += float(loss.item())
        if (step + 1) % cfg.log_every == 0:
            avg = running / cfg.log_every
            running = 0.0
            pbar.set_postfix({"loss": f"{avg:.4f}"})

    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)

    # also copy tokenizer next to checkpoint for convenience
    tok_copy = save_path.with_suffix(".tokenizer.json")
    tok.save(tok_copy)

    print("✅ Training finished")
    print(f"- checkpoint: {save_path}")
    print(f"- tokenizer: {tok_copy}")


if __name__ == "__main__":
    main()


