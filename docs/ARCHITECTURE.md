# 프로젝트 구조

## 📁 디렉토리 구조

```
kr-mini-llm/
├── configs/                    # 설정 파일
│   ├── model_small.yaml       # 150M (테스트용)
│   ├── model_medium.yaml      # 350M (권장)
│   ├── model_large.yaml       # 800M (도전)
│   ├── training_m4max.yaml    # M4 Max 최적화 학습 설정
│   └── profiles/              # 프로파일 기반 설정
│       ├── quick_test.yaml
│       ├── medium_full.yaml
│       └── generate_default.yaml
│
├── src/                        # 소스 코드
│   ├── model/                 # 모델 아키텍처
│   │   ├── transformer.py     # TransformerLM (메인 모델)
│   │   ├── attention.py       # GQA, RoPE
│   │   ├── layers.py          # SwiGLU, RMSNorm
│   │   └── config.py          # TransformerConfig
│   │
│   ├── data/                  # 데이터 처리
│   │   ├── tokenizer.py       # KoreanTokenizer
│   │   └── dataset.py         # Dataset, DataLoader
│   │
│   ├── training/              # 학습 시스템
│   │   ├── trainer.py         # Trainer 클래스
│   │   └── optimizer.py       # AdamW, 스케줄러
│   │
│   └── inference/             # 텍스트 생성
│       └── generator.py       # Generator, GenerationConfig
│
├── scripts/                    # 실행 스크립트
│   ├── train_simple.py        # 간편 학습 (프로파일)
│   ├── generate_simple.py     # 간편 생성 (프로파일)
│   ├── train.py               # 상세 학습
│   ├── generate.py            # 상세 생성
│   ├── tokenize_data.py       # 데이터 토큰화
│   └── train_tokenizer.py     # 토크나이저 학습
│
├── docs/                       # 문서
│   ├── ARCHITECTURE.md        # 프로젝트 구조 (이 파일)
│   ├── COMPONENTS.md          # 컴포넌트 상세 설명
│   └── USAGE.md               # 사용 가이드
│
├── tests/                      # 테스트
│   └── test_model.py
│
├── QUICKSTART.md              # 빠른 실행 가이드
└── README.md                  # 프로젝트 개요
```

## 🏗️ 아키텍처 개요

### 모델 구조
```
TransformerLM
├── Embedding (vocab → hidden)
├── Transformer Blocks × N
│   ├── RMSNorm
│   ├── GroupedQueryAttention (GQA + RoPE)
│   ├── RMSNorm
│   └── SwiGLU FFN
└── Output Head (hidden → vocab)
```

### 데이터 흐름
```
텍스트 → Tokenizer → Token IDs → Model → Logits → 생성된 텍스트
```

## 🔧 주요 컴포넌트

### 1. 모델 (src/model/)
- **transformer.py**: 메인 언어 모델
- **attention.py**: Grouped Query Attention + RoPE
- **layers.py**: SwiGLU, RMSNorm
- **config.py**: 모델 설정 관리

### 2. 데이터 (src/data/)
- **tokenizer.py**: 한국어 토크나이저
- **dataset.py**: 데이터셋, 데이터로더

### 3. 학습 (src/training/)
- **trainer.py**: 학습 루프, 체크포인트, 로깅
- **optimizer.py**: AdamW, 스케줄러

### 4. 생성 (src/inference/)
- **generator.py**: 텍스트 생성, 샘플링

## 📝 설정 파일

### 모델 설정 (configs/model_*.yaml)
```yaml
vocab_size: 32000
hidden_size: 1024
num_layers: 24
num_heads: 16
num_kv_heads: 4
intermediate_size: 4096
max_seq_length: 2048
```

### 학습 설정 (configs/training_*.yaml)
```yaml
batch_size: 8
learning_rate: 2.0e-4
max_steps: 200000
warmup_steps: 4000
```

### 프로파일 (configs/profiles/*.yaml)
모든 설정을 하나의 파일로 통합
```yaml
model:
  config: configs/model_medium.yaml
training:
  config: configs/training_m4max.yaml
data:
  train: data/train.npy
  tokenizer: tokenizer/korean.model
output:
  dir: checkpoints/medium
```

## 🚀 실행 흐름

### 학습
```
1. 프로파일 로드 (configs/profiles/medium_full.yaml)
2. 모델 설정 로드 (configs/model_medium.yaml)
3. 학습 설정 로드 (configs/training_m4max.yaml)
4. 데이터 로드 (data/train.npy)
5. 모델 생성 (TransformerLM)
6. Trainer 초기화
7. 학습 시작
```

### 생성
```
1. 프로파일 로드 (configs/profiles/generate_default.yaml)
2. 체크포인트 로드 (checkpoints/final.pt)
3. 토크나이저 로드
4. Generator 생성
5. 텍스트 생성
```

## 💡 설계 원칙

1. **모듈화**: 각 컴포넌트는 독립적으로 테스트 가능
2. **설정 기반**: 코드 수정 없이 YAML로 제어
3. **프로파일 시스템**: 복잡한 설정을 간단하게
4. **M4 Max 최적화**: MPS, Mixed Precision, 메모리 효율

## 📚 더 알아보기

- [COMPONENTS.md](COMPONENTS.md) - 각 파일의 상세 역할
- [USAGE.md](USAGE.md) - 실제 사용 예시
- [QUICKSTART.md](../QUICKSTART.md) - 빠른 시작
