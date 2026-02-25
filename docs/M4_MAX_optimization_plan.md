# M4 Max 최적화 계획

## 🖥️ 하드웨어 스펙
- **Chip**: Apple M4 Max
- **CPU**: 14-core (10 Performance + 4 Efficiency)
- **GPU**: 32-core
- **통합 메모리**: 36GB
- **메모리 대역폭**: ~400GB/s (예상)

## 🎯 최적화된 모델 크기 제안

### 권장: Medium 모델 (300M-500M 파라미터)
36GB 메모리로 안정적으로 학습 가능한 크기

#### 모델 설정 (Medium)
```yaml
# configs/model_medium.yaml
vocab_size: 32000
hidden_size: 1024          # 768 → 1024 (33% 증가)
num_layers: 24             # 12 → 24 (2배)
num_heads: 16              # 12 → 16
num_kv_heads: 4            # GQA 유지
intermediate_size: 4096    # 2048 → 4096 (2배, SwiGLU)
max_seq_length: 2048       # 1024 → 2048 (2배)
rope_theta: 10000.0
dropout: 0.1

# 예상 파라미터: ~350M
```

#### 학습 설정 (Medium)
```yaml
# configs/training_m4max.yaml
seed: 1337
device: mps  # M4 Max Metal Performance Shaders

data:
  train_path: data/processed/train.txt
  valid_path: data/processed/valid.txt
  tokenizer_model: models/tokenizer.model

train:
  batch_size: 16           # 4 → 16 (4배)
  gradient_accumulation_steps: 4  # 8 → 4
  effective_batch_size: 64  # 16 * 4 = 64
  max_seq_length: 2048     # 512 → 2048 (4배)
  learning_rate: 2.0e-4    # 3e-4 → 2e-4 (큰 모델은 낮은 LR)
  warmup_steps: 4000       # 2000 → 4000
  max_steps: 200000        # 100k → 200k (더 긴 학습)
  save_steps: 5000
  eval_steps: 1000
  mixed_precision: true    # FP16
  gradient_checkpointing: false  # 메모리 충분하면 끄기 (속도 향상)
  max_grad_norm: 1.0
  weight_decay: 0.1
  output_dir: checkpoints/medium/

optimizer:
  type: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8

scheduler:
  type: cosine
  min_lr: 2.0e-5  # learning_rate의 10%
```

### 선택: Large 모델 (700M-1B 파라미터)
메모리를 최대한 활용하는 공격적 설정

#### 모델 설정 (Large)
```yaml
# configs/model_large.yaml
vocab_size: 32000
hidden_size: 1536          # 1.5배 증가
num_layers: 24
num_heads: 24
num_kv_heads: 6            # GQA
intermediate_size: 6144    # 4 * hidden_size
max_seq_length: 2048
rope_theta: 10000.0
dropout: 0.1

# 예상 파라미터: ~800M
```

#### 학습 설정 (Large)
```yaml
# configs/training_m4max_large.yaml
train:
  batch_size: 8            # 메모리 제약
  gradient_accumulation_steps: 8
  effective_batch_size: 64
  max_seq_length: 1536     # 2048보다 약간 작게
  gradient_checkpointing: true  # 메모리 절약 필요
```

## 📊 메모리 사용량 추정

### Medium 모델 (350M)
```
모델 파라미터: 350M * 4 bytes (FP32) = 1.4GB
  → FP16 학습: 350M * 2 bytes = 0.7GB

옵티마이저 상태 (AdamW):
  - 파라미터 복사본: 0.7GB
  - Momentum: 0.7GB
  - Variance: 0.7GB
  → 총 2.1GB

Gradient: 0.7GB

배치 데이터 (batch=16, seq=2048):
  - Activations: ~4-6GB (gradient checkpointing 없이)
  - 입력 데이터: ~0.5GB

총 예상: ~8-10GB (학습 시)
→ 36GB 메모리에서 여유롭게 실행 가능
```

### Large 모델 (800M)
```
모델 파라미터: 800M * 2 bytes (FP16) = 1.6GB
옵티마이저 상태: ~4.8GB
Gradient: 1.6GB
배치 데이터: ~8-12GB (gradient checkpointing 사용)

총 예상: ~18-22GB
→ 36GB 메모리에서 실행 가능하나 여유 적음
```

## 🚀 성능 최적화 전략

### 1. Metal Performance Shaders (MPS) 활용
```python
# PyTorch에서 MPS 사용
device = torch.device("mps")
model = model.to(device)

# MPS 최적화 팁
torch.mps.set_per_process_memory_fraction(0.8)  # 메모리 80% 사용
```

### 2. 데이터 로딩 최적화
```yaml
dataloader:
  num_workers: 8           # CPU 코어 활용 (14코어 중 8개)
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4       # 미리 로드
```

### 3. 컴파일 최적화 (PyTorch 2.0+)
```python
# torch.compile로 속도 향상
model = torch.compile(model, mode="reduce-overhead")
```

### 4. Flash Attention (선택)
```python
# xformers 또는 flash-attention 사용
# MPS에서 지원 여부 확인 필요
from xformers.ops import memory_efficient_attention
```

## 📈 예상 학습 시간

### Medium 모델 (350M)
- **Steps**: 200,000
- **Effective batch size**: 64
- **Tokens per step**: 64 * 2048 = 131,072
- **Total tokens**: 200k * 131k ≈ 26B tokens

**예상 속도** (M4 Max 32-core GPU):
- ~2-3 steps/sec (FP16, no gradient checkpointing)
- **총 학습 시간**: 200k / 2.5 ≈ 80,000초 ≈ **22시간**

### Large 모델 (800M)
- **예상 속도**: ~1-1.5 steps/sec
- **총 학습 시간**: 200k / 1.25 ≈ 160,000초 ≈ **44시간**

## 🎯 데이터 요구사항

### 권장 데이터 크기
- **Minimum**: 10GB 원본 텍스트 (토큰화 후 ~30B tokens)
- **Optimal**: 20-50GB 원본 텍스트 (토큰화 후 ~60-150B tokens)

### 데이터 소스 (한국어)
1. **AI Hub** (우선순위 최상)
   - 일상대화 데이터
   - 문서요약 데이터
   - 뉴스 기사

2. **Korean Wikipedia** (10GB+)
   - 위키미디어 덤프

3. **나무위키** (20GB+)
   - 크롤링 후 전처리

4. **모두의 말뭉치** (국립국어원)
   - 신문, 문어체, 구어체

5. **Common Crawl Korean**
   - 웹 크롤링 데이터

## 🔧 추가 최적화 옵션

### 1. Gradient Accumulation 동적 조정
```python
# 메모리 사용량에 따라 동적 조정
if memory_usage > 0.8:
    gradient_accumulation_steps *= 2
    batch_size //= 2
```

### 2. Mixed Precision 고급 설정
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
# Loss scaling으로 FP16 안정성 향상
```

### 3. Checkpoint Averaging
```python
# 마지막 N개 체크포인트 평균으로 성능 향상
# Stochastic Weight Averaging (SWA)
```

## 📝 구현 우선순위

### Phase 2A: 모델 아키텍처 (3-4일)
1. RoPE 구현
2. RMSNorm 구현
3. GQA 구현
4. SwiGLU FFN 구현
5. TransformerBlock 통합
6. 전체 모델 통합

### Phase 2B: 테스트 및 검증 (1-2일)
1. 단위 테스트
2. Forward/Backward pass 검증
3. 메모리 프로파일링
4. 더미 데이터로 overfitting 테스트

### Phase 3: 데이터 파이프라인 (3-5일)
1. 데이터 수집 (10-20GB)
2. 전처리 파이프라인
3. Tokenizer 학습 (32k vocab)
4. Dataset 클래스 구현
5. DataLoader 최적화

### Phase 4: 학습 (2-3주)
1. Trainer 구현
2. Sanity check (작은 데이터)
3. 본격 학습 (200k steps)
4. 모니터링 및 조정

## 🎯 성공 기준

### Medium 모델
- **Validation Perplexity**: < 15
- **생성 품질**: 문법적으로 올바른 한국어
- **추론 속도**: > 50 tokens/sec
- **학습 시간**: < 30시간

### Large 모델
- **Validation Perplexity**: < 12
- **생성 품질**: 문맥 일관성 + 창의성
- **추론 속도**: > 30 tokens/sec
- **학습 시간**: < 50시간

## 🚨 주의사항

### M4 Max 특화 고려사항
1. **열 관리**: 장시간 학습 시 쿨링 패드 권장
2. **전원 연결**: 고성능 모드 유지
3. **백그라운드 앱**: 학습 중 최소화
4. **MPS 안정성**: PyTorch 2.1+ 사용 권장

### 메모리 관리
- 36GB 중 ~30GB까지 사용 가능
- 시스템 예약 메모리 고려
- OOM 발생 시 batch size 감소

## 📚 참고 자료

### M4 Max 최적화
- Apple Metal Performance Shaders 문서
- PyTorch MPS Backend 가이드

### 모델 아키텍처
- LLaMA 2 논문 및 구현
- Mistral 7B 아키텍처
- GPT-NeoX 구현

## 🎉 결론

**권장 설정**: Medium 모델 (350M 파라미터)
- 36GB 메모리에서 안정적
- 22시간 내 학습 완료 가능
- 실용적인 추론 속도
- 한국어 텍스트 생성에 충분한 성능

**도전 과제**: Large 모델 (800M 파라미터)
- 메모리 최대 활용
- 더 나은 성능 기대
- 44시간 학습 시간
- Gradient checkpointing 필수