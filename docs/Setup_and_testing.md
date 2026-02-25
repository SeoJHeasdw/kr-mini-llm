# 설정 및 테스트 가이드

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd kr-mini-llm

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 설정 파일 확인

생성된 설정 파일들:

```
configs/
├── model_small.yaml          # 레거시 (테스트용, ~50M)
├── model_medium.yaml         # 권장 (M4 Max, ~350M)
├── model_large.yaml          # 도전 (M4 Max, ~800M)
├── training.yaml             # 레거시 (MacBook Air용)
├── training_m4max.yaml       # Medium 모델용
└── training_m4max_large.yaml # Large 모델용
```

## 🧪 테스트

### 설정 파일 테스트

```bash
# 설정 로드 테스트
python3 scripts/test_config.py
```

**예상 출력:**
```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 
kr-mini-llm 설정 테스트
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 

============================================================
🧪 설정 파일 로드 테스트
============================================================

📄 Small (레거시): configs/model_small.yaml
   ✅ 로드 성공
   - Hidden size: 768
   - Layers: 12
   - Heads: 12
   - Parameters: ~50.3M

📄 Medium (권장): configs/model_medium.yaml
   ✅ 로드 성공
   - Hidden size: 1024
   - Layers: 24
   - Heads: 16
   - Parameters: ~350.2M

📄 Large (도전): configs/model_large.yaml
   ✅ 로드 성공
   - Hidden size: 1536
   - Layers: 24
   - Heads: 24
   - Parameters: ~788.5M

============================================================
📊 테스트 결과 요약
============================================================
✅ 통과: 설정 파일 로드
✅ 통과: 프리셋
✅ 통과: 설정 메서드

============================================================
🎉 모든 테스트 통과!
============================================================
```

### 학습 스크립트 테스트

```bash
# Medium 모델 설정 확인
python3 scripts/train.py \
  --config configs/training_m4max.yaml \
  --model_config configs/model_medium.yaml

# Large 모델 설정 확인
python3 scripts/train.py \
  --config configs/training_m4max_large.yaml \
  --model_config configs/model_large.yaml
```

**예상 출력:**
```
📦 PyTorch 버전: 2.x.x
✅ MPS 백엔드 사용 가능
🚀 M4 Max Metal Performance Shaders (MPS) 사용
   MPS 메모리 제한: 80%

📄 설정 파일 로드 중...
   ✅ 학습 설정: configs/training_m4max.yaml
   ✅ 모델 설정: configs/model_medium.yaml

📂 출력 디렉토리: checkpoints/medium/

============================================================
📊 학습 설정 정보
============================================================

🤖 모델:
   - Hidden size: 1024
   - Layers: 24
   - Heads: 16
   - Vocab size: 32000

🏋️  학습:
   - Batch size: 16
   - Gradient accumulation: 4
   - Effective batch size: 64
   - Learning rate: 0.0002
   - Max steps: 200,000
   - Mixed precision: True
   - Gradient checkpointing: False
   - Compile model: True

💻 디바이스:
   - Device: mps
   - MPS 최적화: 활성화

📁 데이터:
   - Train: data/processed/train.txt
   - Valid: data/processed/valid.txt
   - Tokenizer: models/tokenizer.model

============================================================

🎲 Random seed: 1337

============================================================
⚠️  Phase 2-4 구현 필요
============================================================
다음 단계:
1. src/model/* - 모델 아키텍처 구현 (RoPE, GQA, SwiGLU)
2. src/data/* - 데이터 파이프라인 구현
3. src/training/* - 학습 루프 구현
4. 이 스크립트에서 학습 루프 연결
============================================================

💾 설정 저장: checkpoints/medium/config.yaml
💾 모델 설정 저장: checkpoints/medium/model_config.yaml
```

## 📋 설정 파일 상세

### Medium 모델 (권장)

**모델 설정** (`configs/model_medium.yaml`):
```yaml
vocab_size: 32000
hidden_size: 1024
num_layers: 24
num_heads: 16
num_kv_heads: 4
intermediate_size: 4096
max_seq_length: 2048
rope_theta: 10000.0
dropout: 0.1
```

**학습 설정** (`configs/training_m4max.yaml`):
```yaml
device: mps
train:
  batch_size: 16
  gradient_accumulation_steps: 4
  effective_batch_size: 64
  max_seq_length: 2048
  learning_rate: 2.0e-4
  max_steps: 200000
  mixed_precision: true
  gradient_checkpointing: false
  compile_model: true
```

**예상 성능:**
- 파라미터: ~350M
- 메모리 사용: ~10GB
- 학습 속도: 2-3 steps/sec
- 학습 시간: ~22시간
- 추론 속도: > 50 tokens/sec

### Large 모델 (도전)

**모델 설정** (`configs/model_large.yaml`):
```yaml
vocab_size: 32000
hidden_size: 1536
num_layers: 24
num_heads: 24
num_kv_heads: 6
intermediate_size: 6144
max_seq_length: 2048
rope_theta: 10000.0
dropout: 0.1
```

**학습 설정** (`configs/training_m4max_large.yaml`):
```yaml
device: mps
train:
  batch_size: 8
  gradient_accumulation_steps: 8
  effective_batch_size: 64
  max_seq_length: 1536
  learning_rate: 1.5e-4
  max_steps: 200000
  mixed_precision: true
  gradient_checkpointing: true  # 메모리 절약
  compile_model: true
```

**예상 성능:**
- 파라미터: ~800M
- 메모리 사용: ~20GB
- 학습 속도: 1-1.5 steps/sec
- 학습 시간: ~44시간
- 추론 속도: > 30 tokens/sec

## 🔧 Python API 사용

### 설정 로드

```python
from src.model.config import TransformerConfig

# YAML 파일에서 로드
config = TransformerConfig.from_yaml("configs/model_medium.yaml")

# 프리셋 사용
config = TransformerConfig.medium()  # 또는 .small(), .large()

# 설정 정보 출력
print(config)

# 파라미터 수 확인
print(f"Parameters: {config.num_parameters / 1e6:.1f}M")

# 딕셔너리로 변환
config_dict = config.to_dict()

# YAML로 저장
config.save_yaml("my_config.yaml")
```

### 설정 커스터마이징

```python
from src.model.config import TransformerConfig

# Medium 기반으로 커스터마이징
config = TransformerConfig.medium()
config.num_layers = 20  # 레이어 수 조정
config.dropout = 0.2    # Dropout 증가

print(f"Custom config: {config.num_parameters / 1e6:.1f}M parameters")
```

## 🐛 문제 해결

### PyTorch 설치 확인

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### MPS 사용 불가 시

`configs/training_m4max.yaml`에서 device 변경:
```yaml
device: cpu  # mps → cpu
```

### 메모리 부족 (OOM) 시

**Medium 모델:**
```yaml
train:
  batch_size: 8  # 16 → 8
  gradient_accumulation_steps: 8  # 4 → 8
  gradient_checkpointing: true  # false → true
```

**Large 모델:**
```yaml
train:
  batch_size: 4  # 8 → 4
  max_seq_length: 1024  # 1536 → 1024
```

## 📚 다음 단계

1. **Phase 2**: 모델 아키텍처 구현
   - `src/model/attention.py` - GQA 구현
   - `src/model/layers.py` - RoPE, RMSNorm 구현
   - `src/model/transformer.py` - 전체 모델 통합

2. **Phase 3**: 데이터 준비
   - 한국어 데이터 수집 (20-50GB)
   - `scripts/prepare_data.py` - 전처리 파이프라인
   - `scripts/train_tokenizer.py` - 토크나이저 학습

3. **Phase 4**: 학습
   - `src/training/trainer.py` - 학습 루프 구현
   - 실제 학습 실행

4. **Phase 5**: 추론
   - `src/inference/generator.py` - 텍스트 생성
   - KV cache 최적화

## 🎯 체크리스트

### 환경 설정
- [ ] Python 3.9+ 설치
- [ ] 가상환경 생성
- [ ] requirements.txt 설치
- [ ] PyTorch MPS 지원 확인

### 설정 확인
- [ ] `python3 scripts/test_config.py` 실행
- [ ] 모든 테스트 통과
- [ ] `python3 scripts/train.py` 실행
- [ ] 설정 정보 정상 출력

### 다음 단계 준비
- [ ] 모델 아키텍처 구현 계획 확인
- [ ] 데이터 소스 조사
- [ ] 학습 환경 최종 점검

---

**참고 문서:**
- [`M4_MAX_optimization_plan.md`](M4_MAX_optimization_plan.md) - 상세 최적화 가이드
- [`Korean_llm_project_roadmap.md`](Korean_llm_project_roadmap.md) - 전체 로드맵
- [`Configuration_update_plan.md`](Configuration_update_plan.md) - 설정 업데이트 계획