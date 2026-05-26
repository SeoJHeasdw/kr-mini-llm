# kr-mini-llm

MacBook Pro 16" M4 Max에서 **나만의 중형 한국어 LLM**을 개발하기 위한 프로젝트입니다.

## 🖥️ 하드웨어 스펙
- **Chip**: Apple M4 Max
- **CPU**: 14-core (10 Performance + 4 Efficiency)
- **GPU**: 32-core
- **통합 메모리**: 36GB

## 🎯 목표 모델
- **Small**: 150M 파라미터 (테스트용)
- **Medium**: 350M 파라미터 (권장) ⭐
- **Large**: 800M 파라미터 (도전)

## 📊 현재 상태

### ✅ 구현 완료
- **모델 아키텍처**: TransformerLM with RoPE, SwiGLU, RMSNorm, GQA
- **텍스트 생성**: Temperature, Top-K/P, Beam search, Repetition penalty
- **학습 시스템**: Trainer, Optimizer, 학습 스크립트
- **간소화된 CLI**: 프로파일 기반 실행 시스템

### ⚠️ 미학습 상태
- 모델 구조는 완성되었으나 **파라미터가 랜덤 초기화 상태**
- 실제 한국어 텍스트 생성을 위해서는 **학습 필요**
- 데이터 수집 및 학습 준비 완료

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -U pip
pip install -r requirements.txt
```

### 2. 간편 실행 (프로파일 기반) ⚡
```bash
# 빠른 테스트 학습
python scripts/train_simple.py --profile quick_test

# 실제 학습 (Medium 모델)
python scripts/train_simple.py --profile medium_full

# 텍스트 생성
python scripts/generate_simple.py --profile generate_default --prompt "안녕하세요"
```

**👉 자세한 사용법**: [QUICKSTART.md](QUICKSTART.md)

### 3. 기존 방식 (세밀한 제어)
```bash
# 학습
python scripts/train.py \
    --model_config configs/model_medium.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/kullm/tokenized.npy \
    --valid_data data/kullm/tokenized.npy \
    --output_dir checkpoints \
    --tokenizer tokenizer/korean_tokenizer.model

# 생성
python scripts/generate.py \
    --checkpoint checkpoints/final.pt \
    --config configs/model_medium.yaml \
    --tokenizer tokenizer/korean_tokenizer.model \
    --prompt "안녕하세요" \
    --temperature 0.8
```

## 📁 프로젝트 구조
```
.
├── configs/
│   ├── model_small.yaml          # 150M (테스트용)
│   ├── model_medium.yaml         # 350M (권장) ⭐
│   ├── model_large.yaml          # 800M (도전)
│   ├── training_m4max.yaml       # M4 Max 최적화 설정
│   └── profiles/                 # 프로파일 기반 설정
│       ├── quick_test.yaml
│       ├── medium_full.yaml
│       └── generate_default.yaml
├── src/
│   ├── model/                    # Transformer 모델
│   │   ├── transformer.py        # TransformerLM
│   │   ├── attention.py          # GQA, RoPE
│   │   ├── layers.py             # SwiGLU, RMSNorm
│   │   └── config.py
│   ├── data/                     # 데이터 처리
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── training/                 # 학습 시스템
│   │   ├── trainer.py
│   │   └── optimizer.py
│   └── inference/                # 텍스트 생성
│       └── generator.py
├── scripts/
│   ├── train_simple.py           # 간편 학습 (프로파일)
│   ├── generate_simple.py        # 간편 생성 (프로파일)
│   ├── train.py                  # 상세 학습
│   ├── generate.py               # 상세 생성
│   ├── tokenize_data.py          # 데이터 토큰화
│   └── train_tokenizer.py        # 토크나이저 학습
└── docs/                         # 문서
```

## 📚 문서

### 🚀 시작하기
- **[QUICKSTART.md](QUICKSTART.md)** ⚡ 빠른 실행 가이드 - 프로파일 기반 간편 실행
- **[Quick_start_guide.md](docs/Quick_start_guide.md)** ⭐ 5분 안에 시작
- **[Setup_and_testing.md](docs/Setup_and_testing.md)** - 상세 설정 및 테스트

### 📋 계획 및 상태
- **[model_status.md](docs/model_status.md)** ⚠️ 현재 모델 상태 (미학습)
- **[Korean_llm_project_roadmap.md](docs/Korean_llm_project_roadmap.md)** - 전체 로드맵 (6 Phases)
- **[Configuration_update_plan.md](docs/Configuration_update_plan.md)** - 설정 업데이트 계획

### 📖 가이드
- **[training_guide.md](docs/training_guide.md)** - 학습 가이드
- **[data_collection_guide.md](docs/data_collection_guide.md)** - 데이터 수집 가이드
- **[docs/README.md](docs/README.md)** - 문서 구조 및 읽기 순서

## 🎯 모델 스펙

### Small (150M) - 테스트용
- Hidden: 768, Layers: 12, Heads: 12, KV Heads: 4
- 메모리: ~3-5GB
- 학습 속도: 5-8 steps/sec

### Medium (350M) - 권장 ⭐
- Hidden: 1024, Layers: 24, Heads: 16, KV Heads: 4
- 메모리: ~8-10GB
- 학습 속도: 2-3 steps/sec
- 예상 학습 시간: ~22시간 (200k steps)

### Large (800M) - 도전
- Hidden: 1536, Layers: 24, Heads: 24, KV Heads: 6
- 메모리: ~18-22GB
- 학습 속도: 1-1.5 steps/sec
- 예상 학습 시간: ~44시간 (200k steps)

## 🔧 주요 기능

### 모델 아키텍처
- ✅ **RoPE** (Rotary Position Embedding)
- ✅ **GQA** (Grouped Query Attention)
- ✅ **SwiGLU** (Swish-Gated Linear Unit)
- ✅ **RMSNorm** (Root Mean Square Normalization)

### 텍스트 생성
- ✅ Temperature 조절 (0.1 ~ 2.0)
- ✅ Top-K, Top-P (nucleus) 샘플링
- ✅ Beam search
- ✅ Repetition penalty

### 학습 최적화
- ✅ Mixed Precision (FP16)
- ✅ Gradient Accumulation
- ✅ Gradient Checkpointing
- ✅ AdamW Optimizer
- ✅ Cosine LR Scheduler with Warmup

## 📦 데이터 준비

### 1. 데이터 수집
```bash
# KULLMv2 데이터셋 다운로드 (예시)
python scripts/download_kullm_data.py
```

### 2. 토크나이저 학습
```bash
python scripts/train_tokenizer.py \
    --input data/raw/korean_corpus.txt \
    --vocab_size 32000 \
    --output tokenizer/korean_tokenizer.model
```

### 3. 데이터 토큰화
```bash
python scripts/tokenize_data.py \
    --input data/raw/train.txt \
    --output data/processed/train_tokens.npy \
    --tokenizer tokenizer/korean_tokenizer.model \
    --max_length 2048
```

## 🚀 학습 시작

### 프로파일 기반 (간편)
```bash
# 빠른 테스트
python scripts/train_simple.py --profile quick_test

# 실제 학습
python scripts/train_simple.py --profile medium_full

# 체크포인트 재개
python scripts/train_simple.py --profile medium_full --resume checkpoints/medium/step_50000.pt
```

### 상세 제어
```bash
python scripts/train.py \
    --model_config configs/model_medium.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/processed/train_tokens.npy \
    --valid_data data/processed/valid_tokens.npy \
    --output_dir checkpoints/medium \
    --tokenizer tokenizer/korean_tokenizer.model
```

## 💬 텍스트 생성

### 프로파일 기반 (간편)
```bash
# 기본 생성
python scripts/generate_simple.py --profile generate_default --prompt "안녕하세요"

# 창의적 생성
python scripts/generate_simple.py --profile generate_default --prompt "옛날 옛적에" --temperature 1.5

# 긴 텍스트
python scripts/generate_simple.py --profile generate_default --prompt "한국의 역사는" --max-new-tokens 200
```

### 상세 제어
```bash
python scripts/generate.py \
    --checkpoint checkpoints/medium/final.pt \
    --config configs/model_medium.yaml \
    --tokenizer tokenizer/korean_tokenizer.model \
    --prompt "안녕하세요" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9 \
    --repetition-penalty 1.1
```

## 🧪 테스트

```bash
# 모델 구조 테스트
python tests/test_model.py

# 파라미터 수 확인
python scripts/count_params.py --config configs/model_medium.yaml

# 설정 파일 검증
python scripts/test_config.py
```

## 📈 예상 성능 (M4 Max 36GB)

| 모델 | 파라미터 | 메모리 | 학습 속도 | 학습 시간 | 추론 속도 |
|------|---------|--------|----------|----------|----------|
| Small | 150M | 3-5GB | 5-8 steps/s | ~11시간 | >80 tok/s |
| Medium | 350M | 8-10GB | 2-3 steps/s | ~22시간 | >50 tok/s |
| Large | 800M | 18-22GB | 1-1.5 steps/s | ~44시간 | >30 tok/s |

*200k steps 기준

## 🛠️ 개발 로드맵

- [x] Phase 1: 프로젝트 구조 및 환경 설정
- [x] Phase 2: 모델 아키텍처 구현
- [x] Phase 3: 데이터 파이프라인
- [x] Phase 4: 학습 시스템
- [x] Phase 5: 텍스트 생성 시스템
- [ ] Phase 6: 모델 학습 및 평가 (진행 중)

## 🤝 기여

이슈 및 PR 환영합니다!

## 📄 라이선스

MIT License

---
