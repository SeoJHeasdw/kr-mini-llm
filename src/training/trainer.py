"""
학습 루프 구현 (Phase 4)

M4 Max 최적화:
- Mixed precision training (FP16)
- Gradient checkpointing 지원
- MPS 백엔드 최적화
- 효율적인 체크포인트 관리
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    LLM 학습 트레이너
    
    Features:
    - 자동 체크포인트 저장/로드
    - Learning rate scheduling (cosine with warmup)
    - Gradient clipping
    - Mixed precision training
    - 검증 루프
    - 학습 메트릭 로깅
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "mps",
        output_dir: str = "checkpoints",
        # 학습 설정
        max_steps: int = 100000,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        logging_steps: int = 100,
        # 최적화 설정
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,  # Mixed precision
        # Learning rate scheduling
        warmup_steps: int = 2000,
        lr_scheduler_type: str = "cosine",
        # 기타
        resume_from: Optional[str] = None,
        seed: int = 42,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        
        # 학습 설정
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        # 최적화 설정
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device == "mps"  # MPS에서만 AMP 사용
        
        # Learning rate scheduling
        self.warmup_steps = warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시드 설정
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 모델을 디바이스로 이동
        self.model.to(device)
        
        # Mixed precision scaler (MPS용)
        self.scaler: Optional[torch.cuda.amp.GradScaler] = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Learning rate scheduler 생성
        self.scheduler = self._create_scheduler()
        
        # 학습 상태
        self.global_step = 0
        self.epoch = 0
        self.best_valid_loss = float('inf')
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"✅ Trainer 초기화 완료")
        print(f"   - Device: {device}")
        print(f"   - Mixed Precision: {self.use_amp}")
        print(f"   - Gradient Accumulation: {gradient_accumulation_steps}")
        print(f"   - Max Steps: {max_steps:,}")
        print(f"   - Warmup Steps: {warmup_steps:,}")
    
    def _create_scheduler(self):
        """Learning rate scheduler 생성"""
        if self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps - self.warmup_steps,
                eta_min=0
            )
        elif self.lr_scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.max_steps - self.warmup_steps
            )
        else:
            return None
    
    def _get_lr(self) -> float:
        """현재 learning rate 반환"""
        return self.optimizer.param_groups[0]['lr']
    
    def _warmup_lr(self, step: int) -> float:
        """Warmup learning rate 계산"""
        if step < self.warmup_steps:
            return self._get_lr() * (step / self.warmup_steps)
        return self._get_lr()
    
    def train(self) -> None:
        """메인 학습 루프"""
        print(f"\n🚀 학습 시작!")
        print(f"   - 시작 Step: {self.global_step}")
        print(f"   - 목표 Step: {self.max_steps:,}")
        
        self.model.train()
        
        # 학습 메트릭
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        # 무한 데이터 로더 (에폭 개념 없음)
        train_iter = iter(self.train_loader)
        
        while self.global_step < self.max_steps:
            # 배치 가져오기
            try:
                batch = next(train_iter)
            except StopIteration:
                # 데이터로더 재시작
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.epoch += 1
            
            # Forward pass
            loss, num_tokens = self._training_step(batch)
            
            # 메트릭 누적
            total_loss += loss
            total_tokens += num_tokens
            
            # Gradient accumulation
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Learning rate scheduling
                if self.global_step < self.warmup_steps:
                    # Warmup
                    lr = self._warmup_lr(self.global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                elif self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # 로깅
            if self.global_step % self.logging_steps == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / self.logging_steps
                tokens_per_sec = total_tokens / elapsed
                
                print(
                    f"Step {self.global_step:,}/{self.max_steps:,} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {self._get_lr():.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )
                
                # 메트릭 리셋
                total_loss = 0.0
                total_tokens = 0
                start_time = time.time()
            
            # 검증
            if self.global_step % self.eval_steps == 0:
                valid_loss = self.evaluate()
                print(f"📊 Validation Loss: {valid_loss:.4f}")
                
                # Best model 저장
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_checkpoint("best_model")
                    print(f"   ✨ New best model saved!")
                
                self.model.train()
            
            # 체크포인트 저장
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
                print(f"💾 Checkpoint saved at step {self.global_step}")
        
        print(f"\n✅ 학습 완료!")
        print(f"   - 최종 Step: {self.global_step:,}")
        print(f"   - Best Validation Loss: {self.best_valid_loss:.4f}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, int]:
        """단일 학습 스텝"""
        # 배치를 디바이스로 이동
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = self.model(input_ids, attention_mask=attention_mask)
                loss = self._compute_loss(logits, labels)
        else:
            logits, _ = self.model(input_ids, attention_mask=attention_mask)
            loss = self._compute_loss(logits, labels)
        
        # Backward pass
        loss = loss / self.gradient_accumulation_steps
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 토큰 수 계산 (패딩 제외)
        num_tokens = int((labels != -100).sum().item())
        
        return float(loss.item() * self.gradient_accumulation_steps), num_tokens
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss 계산"""
        # logits: [batch_size, seq_len, vocab_size]
        # labels: [batch_size, seq_len]
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """검증 루프"""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.valid_loader, desc="Evaluating", leave=False):
            # 배치를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            logits, _ = self.model(input_ids, attention_mask=attention_mask)
            loss = self._compute_loss(logits, labels)
            
            # 메트릭 누적
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        print(f"   - Perplexity: {perplexity:.2f}")
        
        return avg_loss
    
    def save_checkpoint(self, name: str) -> None:
        """체크포인트 저장"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt"
        )
        
        # Optimizer 저장
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt"
        )
        
        # Scheduler 저장
        if self.scheduler:
            torch.save(
                self.scheduler.state_dict(),
                checkpoint_dir / "scheduler.pt"
            )
        
        # 학습 상태 저장
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_valid_loss': self.best_valid_loss,
        }
        with open(checkpoint_dir / "trainer_state.json", 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """체크포인트 로드"""
        checkpoint_dir = Path(checkpoint_path)
        
        print(f"📂 체크포인트 로딩: {checkpoint_dir}")
        
        # 모델 로드
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"   ✅ 모델 로드 완료")
        
        # Optimizer 로드
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print(f"   ✅ Optimizer 로드 완료")
        
        # Scheduler 로드
        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists() and self.scheduler:
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            print(f"   ✅ Scheduler 로드 완료")
        
        # 학습 상태 로드
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_valid_loss = state['best_valid_loss']
            print(f"   ✅ 학습 상태 로드 완료 (Step: {self.global_step:,})")


# Made with Bob
