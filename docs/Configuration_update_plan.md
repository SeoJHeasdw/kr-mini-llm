# 설정 파일 업데이트 계획

## 📋 업데이트 필요 파일 목록

### 1. 모델 설정 파일

#### 생성: `configs/model_medium.yaml`
```yaml
# M4 Max 최적화 중형 모델 (권장)
vocab_size: 32000
hidden_size: 1024
num_layers: 24
num_heads: 16
num_kv_heads: 4  # GQA
intermediate_size: 4096  # SwiGLU (4 * hidden_size)
max_seq_length: 2048
rope_theta: 10000.0
dropout: 0.1

# 예상 파라미터: ~350M
# 메모리 사용량: ~8-10GB (학습 시)
```

#### 생성: `configs/model_large.yaml`
```yaml
# M4 Max 대형 모델 (도전 과제)
vocab_size: 32000
hidden_size: 1536
num_layers: 24
num_heads: 24
num_kv_heads: 6  # GQA
intermediate_size: 6144  # SwiGLU (4 * hidden_size)
max_seq_length: 2048
rope_theta: 10000.0
dropout: 0.1

# 예상 파라미터: ~800M
# 메모리 사용량: ~18-22GB (학습 시)
```

#### 유지: `configs/model_small.yaml`
- 기존 파일 유지 (테스트/디버깅용)
- 주석 추가: "레거시 - 테스트용"

---

### 2. 학습 설정 파일

#### 생성: `configs/training_m4max.yaml`
```yaml
# M4 Max 최적화 학습 설정 (Medium 모델용)

seed: 1337
device: mps  # Metal Performance Shaders

data:
  train_path: data/processed/train.txt
  valid_path: data/processed/valid.txt
  tokenizer_model: models/tokenizer.model

train:
  # 배치 설정
  batch_size: 16
  gradient_accumulation_steps: 4
  effective_batch_size: 64  # 16 * 4
  max_seq_length: 2048
  
  # 옵티마이저
  learning_rate: 2.0e-4
  weight_decay: 0.1
  max_grad_norm: 1.0
  
  # 스케줄러
  warmup_steps: 4000
  max_steps: 200000
  lr_scheduler: cosine
  min_lr: 2.0e-5  # learning_rate의 10%
  
  # 체크포인트
  save_steps: 5000
  eval_steps: 1000
  logging_steps: 100
  output_dir: checkpoints/medium/
  
  # 최적화
  mixed_precision: true  # FP16
  gradient_checkpointing: false  # 메모리 충분하면 끄기
  compile_model: true  # torch.compile 사용
  
  # 데이터 로딩
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

optimizer:
  type: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8

# W&B 로깅 (선택)
wandb:
  enabled: true
  project: kr-mini-llm
  name: medium-m4max
  tags: [m4max, medium, 350m]
```

#### 생성: `configs/training_m4max_large.yaml`
```yaml
# M4 Max 최적화 학습 설정 (Large 모델용)

seed: 1337
device: mps

data:
  train_path: data/processed/train.txt
  valid_path: data/processed/valid.txt
  tokenizer_model: models/tokenizer.model

train:
  # 배치 설정 (메모리 제약)
  batch_size: 8
  gradient_accumulation_steps: 8
  effective_batch_size: 64
  max_seq_length: 1536  # 2048보다 약간 작게
  
  # 옵티마이저
  learning_rate: 1.5e-4  # 더 큰 모델은 더 낮은 LR
  weight_decay: 0.1
  max_grad_norm: 1.0
  
  # 스케줄러
  warmup_steps: 5000
  max_steps: 200000
  lr_scheduler: cosine
  min_lr: 1.5e-5
  
  # 체크포인트
  save_steps: 5000
  eval_steps: 1000
  logging_steps: 100
  output_dir: checkpoints/large/
  
  # 최적화
  mixed_precision: true
  gradient_checkpointing: true  # 메모리 절약 필요
  compile_model: true
  
  # 데이터 로딩
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4

optimizer:
  type: adamw
  betas: [0.9, 0.95]
  eps: 1.0e-8

wandb:
  enabled: true
  project: kr-mini-llm
  name: large-m4max
  tags: [m4max, large, 800m]
```

#### 수정: `configs/training.yaml`
- 주석 추가: "레거시 - MacBook Air용"
- 새 파일 참조 추가

---

### 3. 추론 설정 파일

#### 생성: `configs/inference.yaml`
```yaml
# 추론 설정

model:
  checkpoint_path: checkpoints/medium/final/
  device: mps
  compile: true  # torch.compile로 속도 향상

generation:
  max_length: 512
  temperature: 0.8
  top_p: 0.95
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true
  
  # 배치 추론
  batch_size: 4
  use_cache: true  # KV cache 사용

# 성능 목표
# - Medium 모델: > 50 tokens/sec
# - Large 모델: > 30 tokens/sec
```

---

## 🔧 코드 수정 필요 사항

### 1. `src/model/config.py`
```python
@dataclass
class TransformerConfig:
    """M4 Max 최적화 설정"""
    vocab_size: int = 32000
    hidden_size: int = 1024  # 768 → 1024
    num_layers: int = 24     # 12 → 24
    num_heads: int = 16      # 12 → 16
    num_kv_heads: int = 4
    intermediate_size: int = 4096  # 2048 → 4096
    max_seq_length: int = 2048     # 1024 → 2048
    rope_theta: float = 10000.0
    dropout: float = 0.1     # 0.0 → 0.1
    
    @classmethod
    def from_yaml(cls, path: str) -> "TransformerConfig":
        """YAML 파일에서 설정 로드"""
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    @classmethod
    def medium(cls) -> "TransformerConfig":
        """Medium 모델 프리셋"""
        return cls()
    
    @classmethod
    def large(cls) -> "TransformerConfig":
        """Large 모델 프리셋"""
        return cls(
            hidden_size=1536,
            num_heads=24,
            num_kv_heads=6,
            intermediate_size=6144
        )
```

### 2. `scripts/train.py` 수정 필요
```python
# MPS 디바이스 설정
if config.device == "auto":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(config.device)

# torch.compile 적용
if config.train.compile_model:
    model = torch.compile(model, mode="reduce-overhead")

# MPS 메모리 관리
if device.type == "mps":
    torch.mps.set_per_process_memory_fraction(0.8)
```

### 3. `scripts/generate.py` 수정 필요
```python
# KV cache 구현
# 추론 속도 최적화
# 배치 추론 지원
```

---

## 📊 성능 벤치마크 목표

### Medium 모델 (350M)
- **학습 속도**: 2-3 steps/sec
- **학습 시간**: ~22시간 (200k steps)
- **추론 속도**: > 50 tokens/sec
- **메모리 사용**: ~10GB
- **Validation PPL**: < 15

### Large 모델 (800M)
- **학습 속도**: 1-1.5 steps/sec
- **학습 시간**: ~44시간 (200k steps)
- **추론 속도**: > 30 tokens/sec
- **메모리 사용**: ~20GB
- **Validation PPL**: < 12

---

## 🚀 실행 순서

### Phase 2: 아키텍처 구현 (Code 모드 필요)
1. 설정 파일 생성
   - `configs/model_medium.yaml`
   - `configs/model_large.yaml`
   - `configs/training_m4max.yaml`
   - `configs/training_m4max_large.yaml`
   - `configs/inference.yaml`

2. 코드 수정
   - `src/model/config.py` - 설정 클래스 업데이트
   - `scripts/train.py` - MPS 지원 추가
   - `scripts/generate.py` - 추론 최적화

3. 테스트
   - 설정 파일 로드 테스트
   - 더미 데이터로 forward pass 테스트
   - 메모리 프로파일링

### Phase 3: 데이터 준비
1. 한국어 데이터 수집 (20-50GB)
2. 전처리 파이프라인 실행
3. Tokenizer 학습 (32k vocab)

### Phase 4: 학습
1. Medium 모델 학습 시작
2. 모니터링 및 조정
3. 체크포인트 평가

---

## ✅ 체크리스트

### 설정 파일 생성
- [ ] `configs/model_medium.yaml`
- [ ] `configs/model_large.yaml`
- [ ] `configs/training_m4max.yaml`
- [ ] `configs/training_m4max_large.yaml`
- [ ] `configs/inference.yaml`

### 코드 수정
- [ ] `src/model/config.py` - 프리셋 메서드 추가
- [ ] `scripts/train.py` - MPS 지원
- [ ] `scripts/generate.py` - 추론 최적화
- [ ] `src/training/trainer.py` - 학습 루프 구현

### 테스트
- [ ] 설정 파일 로드 테스트
- [ ] 모델 초기화 테스트
- [ ] Forward/Backward pass 테스트
- [ ] 메모리 프로파일링

---

## 🎯 다음 단계

**Plan 모드에서 완료**:
- ✅ M4 Max 최적화 계획 수립
- ✅ 문서 업데이트
- ✅ 설정 파일 계획 작성

**Code 모드로 전환 필요**:
- 실제 설정 파일 생성 (YAML)
- 코드 수정 (Python)
- 테스트 실행

Code 모드로 전환하여 실제 파일을 생성하고 코드를 수정하시겠습니까?