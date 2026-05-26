# 컴포넌트 상세 가이드

각 파일과 모듈의 역할을 상세히 설명합니다.

## 📦 src/model/ - 모델 아키텍처

### transformer.py - TransformerLM
**역할**: 메인 언어 모델 클래스

**주요 클래스**:
- `TransformerLM`: 전체 Transformer 언어 모델
  - `forward()`: 입력 토큰 → 로짓 출력
  - `generate()`: 텍스트 생성
  - `count_parameters()`: 파라미터 수 계산

**사용 예시**:
```python
from src.model.transformer import TransformerLM
from src.model.config import TransformerConfig

config = TransformerConfig.from_yaml("configs/model_medium.yaml")
model = TransformerLM(config)

# Forward pass
logits, loss = model(input_ids, labels=labels)

# 파라미터 수
params = model.count_parameters()
# {'total': 350000000, 'embedding': ..., 'attention': ..., 'ffn': ...}
```

### attention.py - Grouped Query Attention
**역할**: 효율적인 어텐션 메커니즘

**주요 클래스**:
- `GroupedQueryAttention`: GQA 구현
  - Query heads: 16개
  - Key/Value heads: 4개 (4:1 비율)
  - RoPE 적용

**특징**:
- 메모리 효율: KV 캐시 크기 75% 감소
- 속도: Multi-Head Attention 대비 1.5배 빠름

### layers.py - 핵심 레이어
**역할**: Transformer 블록의 구성 요소

**주요 클래스**:
- `RMSNorm`: Root Mean Square Normalization
  - LayerNorm보다 빠르고 안정적
  
- `SwiGLU`: Swish-Gated Linear Unit
  - FFN의 활성화 함수
  - GELU보다 성능 우수

**사용 예시**:
```python
from src.model.layers import RMSNorm, SwiGLU

# RMSNorm
norm = RMSNorm(hidden_size=1024)
normalized = norm(x)

# SwiGLU
ffn = SwiGLU(hidden_size=1024, intermediate_size=4096)
output = ffn(x)
```

### config.py - 설정 관리
**역할**: 모델 설정 로드 및 관리

**주요 클래스**:
- `TransformerConfig`: 모델 하이퍼파라미터
  - `from_yaml()`: YAML 파일에서 로드
  - `to_dict()`: 딕셔너리로 변환

## 📊 src/data/ - 데이터 처리

### tokenizer.py - 토크나이저
**역할**: 텍스트 ↔ 토큰 변환

**주요 클래스**:
- `KoreanTokenizer`: 한국어 토크나이저
  - `encode()`: 텍스트 → 토큰 IDs
  - `decode()`: 토큰 IDs → 텍스트
  - `train()`: 새 토크나이저 학습

**사용 예시**:
```python
from src.data.tokenizer import KoreanTokenizer

# 로드
tokenizer = KoreanTokenizer("tokenizer/korean.model")

# 인코딩
tokens = tokenizer.encode("안녕하세요")
# [101, 5234, 102]

# 디코딩
text = tokenizer.decode(tokens)
# "안녕하세요"
```

### dataset.py - 데이터셋
**역할**: 학습 데이터 로딩 및 배치 생성

**주요 함수**:
- `create_dataloaders()`: 학습/검증 데이터로더 생성
  - 토큰화된 파일 (.npy) 지원
  - 텍스트 파일 (.txt) 지원
  - 스트리밍 모드 지원

**사용 예시**:
```python
from src.data.dataset import create_dataloaders

train_loader, valid_loader = create_dataloaders(
    train_data="data/train.npy",
    valid_data="data/valid.npy",
    tokenizer=tokenizer,
    batch_size=8,
    max_length=2048,
    use_tokenized=True
)
```

## 🎓 src/training/ - 학습 시스템

### trainer.py - Trainer
**역할**: 학습 루프 관리

**주요 클래스**:
- `Trainer`: 학습 오케스트레이션
  - `train()`: 메인 학습 루프
  - `evaluate()`: 검증
  - `save_checkpoint()`: 체크포인트 저장
  - `load_checkpoint()`: 체크포인트 로드

**기능**:
- Mixed Precision (FP16)
- Gradient Accumulation
- Gradient Clipping
- Learning Rate Scheduling
- 체크포인트 관리
- 로깅 (W&B 지원)

**사용 예시**:
```python
from src.training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    device="mps",
    output_dir="checkpoints/",
    max_steps=200000,
    eval_steps=1000,
    save_steps=5000
)

trainer.train()
```

### optimizer.py - 옵티마이저
**역할**: 최적화 알고리즘 설정

**주요 함수**:
- `create_optimizer()`: AdamW 옵티마이저 생성
  - Weight decay
  - Learning rate
  - Betas, epsilon

**사용 예시**:
```python
from src.training.optimizer import create_optimizer

optimizer = create_optimizer(
    model=model,
    lr=2e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)
```

## 💬 src/inference/ - 텍스트 생성

### generator.py - Generator
**역할**: 텍스트 생성 및 샘플링

**주요 클래스**:
- `Generator`: 텍스트 생성기
  - `generate()`: 텍스트 생성
  - Temperature 조절
  - Top-K, Top-P 샘플링
  - Beam search
  - Repetition penalty

- `GenerationConfig`: 생성 설정
  - `max_new_tokens`: 최대 생성 토큰 수
  - `temperature`: 샘플링 온도
  - `top_k`, `top_p`: 샘플링 파라미터

**사용 예시**:
```python
from src.inference.generator import Generator, GenerationConfig

generator = Generator(model, tokenizer, device="mps")

config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1
)

text = generator.generate("안녕하세요", config)
```

## 🔧 scripts/ - 실행 스크립트

### train_simple.py
**역할**: 프로파일 기반 간편 학습

**사용법**:
```bash
python scripts/train_simple.py --profile medium_full
```

**특징**:
- 프로파일 하나로 모든 설정
- 오버라이드 가능 (--device, --resume 등)

### generate_simple.py
**역할**: 프로파일 기반 간편 생성

**사용법**:
```bash
python scripts/generate_simple.py --profile generate_default --prompt "안녕하세요"
```

### train.py
**역할**: 상세 제어 학습

**사용법**:
```bash
python scripts/train.py \
    --model_config configs/model_medium.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/train.npy \
    --valid_data data/valid.npy \
    --output_dir checkpoints/
```

### generate.py
**역할**: 상세 제어 생성

**사용법**:
```bash
python scripts/generate.py \
    --checkpoint checkpoints/final.pt \
    --config configs/model_medium.yaml \
    --tokenizer tokenizer/korean.model \
    --prompt "안녕하세요" \
    --temperature 0.8
```

### tokenize_data.py
**역할**: 텍스트 데이터를 토큰화하여 .npy 파일로 저장

**사용법**:
```bash
python scripts/tokenize_data.py \
    --input data/raw/train.txt \
    --output data/processed/train_tokens.npy \
    --tokenizer tokenizer/korean.model \
    --max_length 2048
```

**장점**:
- 학습 시 빠른 로딩
- 메모리 효율적
- 한 번만 토큰화

### train_tokenizer.py
**역할**: 새 토크나이저 학습

**사용법**:
```bash
python scripts/train_tokenizer.py \
    --input data/raw/corpus.txt \
    --vocab_size 32000 \
    --output tokenizer/korean.model
```

## ⚙️ configs/ - 설정 파일

### model_*.yaml
**역할**: 모델 아키텍처 설정

**주요 파라미터**:
- `vocab_size`: 어휘 크기 (32000)
- `hidden_size`: 은닉층 크기 (1024)
- `num_layers`: 레이어 수 (24)
- `num_heads`: 어텐션 헤드 수 (16)
- `num_kv_heads`: KV 헤드 수 (4)
- `intermediate_size`: FFN 크기 (4096)
- `max_seq_length`: 최대 시퀀스 길이 (2048)

### training_*.yaml
**역할**: 학습 하이퍼파라미터

**주요 파라미터**:
- `batch_size`: 배치 크기 (8)
- `learning_rate`: 학습률 (2e-4)
- `max_steps`: 최대 스텝 (200000)
- `warmup_steps`: 워밍업 스텝 (4000)
- `gradient_accumulation_steps`: 그래디언트 누적 (8)

### profiles/*.yaml
**역할**: 통합 설정 프로파일

**구조**:
```yaml
model:
  config: configs/model_medium.yaml
training:
  config: configs/training_m4max.yaml
data:
  train: data/train.npy
  valid: data/valid.npy
  tokenizer: tokenizer/korean.model
output:
  dir: checkpoints/medium
device: mps
```

## 📚 더 알아보기

- [ARCHITECTURE.md](ARCHITECTURE.md) - 전체 구조
- [USAGE.md](USAGE.md) - 실제 사용 예시
- [QUICKSTART.md](../QUICKSTART.md) - 빠른 시작
