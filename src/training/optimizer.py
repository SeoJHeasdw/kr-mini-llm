"""
Optimizer 설정 및 생성

AdamW optimizer with weight decay 지원
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    AdamW optimizer 생성
    
    Args:
        model: 학습할 모델
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Adam beta parameters
        eps: Adam epsilon
    
    Returns:
        torch.optim.AdamW optimizer
    """
    # Weight decay를 적용하지 않을 파라미터 그룹
    no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=betas,
        eps=eps
    )
    
    return optimizer


# Made with Bob
