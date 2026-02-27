# 한국어 LLM 학습 가이드

## 📋 개요

이 문서는 kr-mini-llm 프로젝트의 학습 파이프라인을 설명합니다. M4 Max 36GB 환경에 최적화되어 있습니다.

## 🎯 학습 전 체크리스트

### 1. 환경 준비
- [ ] Python 3.9+ 설치
- [ ] PyTorch 2.0+ (MPS 지원) 설치
- [ ] 필수 패키지 설치 (`pip install -r requirements.txt`)
- [ ] 36GB 메모리 확보 (다른 앱 종료)

### 2. 데이터 준비
- [ ] 한국어 텍스트 데이터 수집 (최소 1GB 권장)
- [ ] 토크나이저 학습 완료
- [ ] 학습/검증 데이터 분할 완료

### 3. 설정 파일 준비
- [ ] 모델 설정 파일 (`configs/model_*.yaml`)
- [ ] 학습 설정 파일 (`configs/training*.yaml`)

## 📊 Phase 3: 데이터 준비

### 3.1 데이터 다운로드

```bash
# KOREAN-WEBTEXT 데이터셋 다운로드 (예시)
python scripts/prepare_data.py --download

# 샘플 수 제한 (테스트용)
python scripts/prepare_data.py --download --max_samples 10000
```

### 3.2 데이터 전처리 및 분할

```bash
# 학습/검증 데이터 분할 (기본 1% 검증)
python scripts/prepare_data.py --prepare

# 커스텀 분할 비율
python scripts/prepare_data.py --prepare --valid_ratio 0.02
```

**출력 파일:**
- `data/processed/train.txt` - 학습 데이터
- `data/processed/valid.txt` - 검증 데이터

### 3.3 토크나이저 학습

```bash
python scripts/train_tokenizer.py \
    --input data/processed/train.txt \
    --vocab_size 32000 \
    --output models/tokenizer.model
```

## 🏋️ Phase 4: 모델 학습

### 4.1 학습 설정 선택

프로젝트는 3가지 모델 크기를 지원합니다:

| 모델 | 파라미터 | 메모리 | 학습 시간 (M4 Max) | 설정 파일 |
|------|----------|--------|-------------------|-----------|
| **Small** | 150M | ~8GB | 2-3일 | `model_small.yaml` |
| **Medium** | 350M | ~16GB | 5-7일 | `model_medium.yaml` |
| **Large** | 800M | ~32GB | 10-14일 | `model_large.yaml` |

### 4.2 기본 학습 실행

```bash
# Small 모델 학습
python scripts/train.py \
    --model_config configs/model_small.yaml \
    --training_config configs/training.yaml \
    --train_data data/processed/train.txt \
    --valid_data data/processed/valid.txt \
    --tokenizer models/tokenizer.model \
    --output_dir checkpoints/small
```

### 4.3 M4 Max 최적화 학습

```bash
# Medium 모델 (M4 Max 최적화)
python scripts/train.py \
    --model_config configs/model_medium.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/processed/train.txt \
    --valid_data data/processed/valid.txt \
    --tokenizer models/tokenizer.model \
    --output_dir checkpoints/medium \
    --device mps
```

### 4.4 대용량 데이터 학습 (스트리밍)

```bash
# 스트리밍 모드 (메모리 효율적)
python scripts/train.py \
    --model_config configs/model_large.yaml \
    --training_config configs/training_m4max_large.yaml \
    --train_data data/processed/train.txt \
    --valid_data data/processed/valid.txt \
    --tokenizer models/tokenizer.model \
    --output_dir checkpoints/large \
    --device mps \
    --streaming
```

### 4.5 체크포인트에서 재개

```bash
python scripts/train.py \
    --model_config configs/model_medium.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/processed/train.txt \
    --valid_data data/processed/valid.txt \
    --tokenizer models/tokenizer.model \
    --output_dir checkpoints/medium \
    --resume_from checkpoints/medium/checkpoint-5000
```

## ⚙️ 학습 설정 상세

### 주요 하이퍼파라미터

```yaml
# configs/training_m4max.yaml 예시

# 배치 설정
batch_size: 16                    # M4 Max 36GB에 최적화
gradient_accumulation_steps: 4    # Effective batch = 64
max_seq_length: 2048              # 긴 컨텍스트

# 학습 설정
learning_rate: 2.0e-4             # 큰 모델은 낮은 LR
warmup_steps: 4000                # Warmup 단계
max_steps: 200000                 # 총 학습 스텝
lr_scheduler_type: "cosine"       # Cosine annealing

# 최적화
max_grad_norm: 1.0                # Gradient clipping
weight_decay: 0.01                # L2 regularization
adam_betas: [0.9, 0.95]           # Adam beta
adam_epsilon: 1.0e-8              # Adam epsilon

# Mixed Precision
fp16: true                        # FP16 학습 (MPS)

# 체크포인트
save_steps: 5000                  # 체크포인트 저장 주기
eval_steps: 1000                  # 검증 주기
logging_steps: 100                # 로깅 주기

# 기타
num_workers: 4                    # 데이터 로더 워커
seed: 42                          # 랜덤 시드
```

### 메모리 최적화 팁

**OOM (Out of Memory) 발생 시:**

1. **배치 크기 감소**
   ```yaml
   batch_size: 8  # 16 → 8
   gradient_accumulation_steps: 8  # 4 → 8 (effective batch 유지)
   ```

2. **시퀀스 길이 감소**
   ```yaml
   max_seq_length: 1024  # 2048 → 1024
   ```

3. **스트리밍 모드 사용**
   ```bash
   --streaming
   ```

4. **워커 수 감소**
   ```yaml
   num_workers: 2  # 4 → 2
   ```

## 📈 학습 모니터링

### 학습 중 출력 예시

```
Step 100/200,000 | Loss: 3.4521 | LR: 5.00e-05 | Tokens/s: 2,450 | Time: 12.3s
Step 200/200,000 | Loss: 3.2134 | LR: 1.00e-04 | Tokens/s: 2,480 | Time: 11.8s
...
Step 1,000/200,000 | Loss: 2.8765 | LR: 2.00e-04 | Tokens/s: 2,520 | Time: 11.5s
📊 Validation Loss: 2.7543
   - Perplexity: 15.73
   ✨ New best model saved!
💾 Checkpoint saved at step 5,000
```

### 주요 메트릭

- **Loss**: 낮을수록 좋음 (일반적으로 2.0-3.5 범위)
- **Perplexity**: 낮을수록 좋음 (exp(loss))
- **Tokens/s**: 높을수록 좋음 (M4 Max에서 2,000-3,000 목표)
- **Learning Rate**: Warmup 후 점진적 감소

### 학습 진행 확인

```bash
# 체크포인트 디렉토리 확인
ls -lh checkpoints/medium/

# 출력:
# checkpoint-5000/
# checkpoint-10000/
# checkpoint-15000/
# best_model/
```

## 🎓 학습 후 작업

### 1. Best Model 확인

```bash
# Best model 위치
checkpoints/medium/best_model/
├── model.pt           # 모델 가중치
├── optimizer.pt       # Optimizer 상태
├── scheduler.pt       # Scheduler 상태
└── trainer_state.json # 학습 상태
```

### 2. 텍스트 생성 테스트

```bash
python scripts/generate.py \
    --checkpoint checkpoints/medium/best_model/model.pt \
    --config configs/model_medium.yaml \
    --tokenizer models/tokenizer.model \
    --prompt "안녕하세요" \
    --temperature 0.8 \
    --max_new_tokens 100
```

### 3. 모델 평가

```python
# 검증 데이터로 Perplexity 측정
from src.training.trainer import Trainer

trainer = Trainer(...)
trainer.load_checkpoint("checkpoints/medium/best_model")
valid_loss = trainer.evaluate()
perplexity = math.exp(valid_loss)
print(f"Perplexity: {perplexity:.2f}")
```

## 🔧 문제 해결

### 일반적인 문제

#### 1. MPS 백엔드 오류

**증상**: `RuntimeError: MPS backend not available`

**해결**:
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# MPS 지원 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# CPU로 대체
--device cpu
```

#### 2. 학습 속도 느림

**원인**: 
- 백그라운드 앱이 메모리/CPU 사용
- 배터리 모드 (전원 연결 필요)
- 데이터 로딩 병목

**해결**:
```yaml
# 워커 수 증가
num_workers: 8  # 4 → 8

# 배치 크기 증가 (메모리 허용 시)
batch_size: 24  # 16 → 24
```

#### 3. Loss가 발산 (NaN)

**원인**: Learning rate가 너무 높음

**해결**:
```yaml
# Learning rate 감소
learning_rate: 1.0e-4  # 2.0e-4 → 1.0e-4

# Gradient clipping 강화
max_grad_norm: 0.5  # 1.0 → 0.5
```

#### 4. Validation Loss가 개선되지 않음

**원인**: 
- 데이터 부족
- 모델 크기 부적절
- Overfitting

**해결**:
- 더 많은 데이터 수집 (5-10GB 이상)
- 모델 크기 조정
- Weight decay 증가
- Dropout 추가

## 📊 예상 학습 시간 (M4 Max 36GB)

| 모델 | 데이터 크기 | 학습 스텝 | 예상 시간 | 비고 |
|------|------------|----------|----------|------|
| Small (150M) | 1GB | 50k | 1-2일 | 빠른 프로토타입 |
| Small (150M) | 5GB | 100k | 2-3일 | 기본 품질 |
| Medium (350M) | 5GB | 100k | 4-5일 | 권장 설정 |
| Medium (350M) | 10GB | 200k | 5-7일 | 고품질 |
| Large (800M) | 10GB | 200k | 10-14일 | 최고 품질 |

**참고**: 
- 실제 시간은 데이터 복잡도와 시스템 상태에 따라 달라질 수 있습니다
- 전원 연결 필수 (배터리 모드는 성능 저하)
- 쿨링 패드 권장 (장시간 학습 시)

## 🎯 다음 단계

학습 완료 후:

1. **텍스트 생성 테스트** → `docs/text_generation.md` 참조
2. **모델 평가** → Perplexity, 생성 품질 확인
3. **하이퍼파라미터 튜닝** → 더 나은 결과를 위한 조정
4. **Instruction Tuning** (선택) → 특정 태스크 최적화

---

**Made with Bob** 🤖