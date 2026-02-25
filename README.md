# kr-mini-llm

MacBook Pro 16" M4 Max에서 **나만의 중형 한국어 LLM**을 개발하기 위한 프로젝트입니다.
이 레포는 `docs/`의 로드맵/가이드를 기반으로, **초기 구축(Phase 1) 스캐폴딩**을 제공합니다.

## 🖥️ 하드웨어 스펙
- **Chip**: Apple M4 Max
- **CPU**: 14-core (10 Performance + 4 Efficiency)
- **GPU**: 32-core
- **통합 메모리**: 36GB

## 🎯 목표 모델
- **Medium**: 350M 파라미터 (권장)
- **Large**: 800M 파라미터 (도전)
- **아키텍처**: Transformer with RoPE, SwiGLU, RMSNorm, GQA
- **예상 학습 시간**: 22-44시간

## Quick Start (가장 간단한 초기 구축)

아래 커맨드는 **레포 루트 디렉토리에서** 실행하세요.

```bash
# 1) 가상환경
python3 -m venv venv
source venv/bin/activate
pip install -U pip

# 2) 의존성 설치
pip install -r requirements.txt
```

## What you get
- **구조**: `src/`, `scripts/`, `configs/`, `tests/`, `docs/` 기반 프로젝트 레이아웃
- **M4 Max 최적화**: 36GB 메모리를 활용한 중형 모델 설정
- **상세 문서**: [`M4_MAX_optimization_plan.md`](docs/M4_MAX_optimization_plan.md) 참고
- **다음 단계 연결**: Phase 2+ (Tokenizer, 모델 아키텍처, 데이터 파이프라인, 학습/추론)로 확장하기 쉬운 틀

## Project Structure (핵심)
```
.
├── configs/                  # 설정 파일
├── docs/                     # 가이드/로드맵
├── scripts/                  # 데이터 준비/학습/추론 엔트리포인트
└── src/
    ├── data/                 # Tokenizer / Dataset
    └── model/                # Transformer 모델 컴포넌트
```

## 📚 문서

### 🚀 시작하기
- **[Quick_start_guide.md](docs/Quick_start_guide.md)** ⭐ 먼저 읽기 - 5분 안에 시작
- **[Setup_and_testing.md](docs/Setup_and_testing.md)** - 상세 설정 및 테스트 가이드

### 📋 계획 및 최적화
- **[M4_MAX_optimization_plan.md](docs/M4_MAX_optimization_plan.md)** ⭐ 상세 최적화 가이드
- **[Korean_llm_project_roadmap.md](docs/Korean_llm_project_roadmap.md)** - 전체 로드맵 (6 Phases)
- **[Configuration_update_plan.md](docs/Configuration_update_plan.md)** - 설정 업데이트 계획

### 📖 문서 가이드
- **[docs/README.md](docs/README.md)** - 문서 구조 및 읽기 순서 가이드

## 🎯 Next Steps

### 즉시 실행 가능
```bash
# 1. 환경 설정 확인
source venv/bin/activate
python3 scripts/test_config.py

# 2. 학습 스크립트 테스트
python3 scripts/train.py \
  --config configs/training_m4max.yaml \
  --model_config configs/model_medium.yaml
```

### Phase 2부터 시작
1. **모델 아키텍처 구현** (3-5일)
   - RoPE, RMSNorm, GQA, SwiGLU
   - [Quick_start_guide.md > Phase 2](docs/Quick_start_guide.md#phase-2-아키텍처-구현-🏗️-3-5일)

2. **데이터 준비** (2-3일)
   - 20-50GB 한국어 데이터 수집
   - Tokenizer 학습 (32k vocab)

3. **학습 시작** (2-3주)
   - Medium 모델: 200k steps (~22시간)
   - [M4_MAX_optimization_plan.md](docs/M4_MAX_optimization_plan.md) 참고

## 🚀 Quick Commands
```bash
# 환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 데이터 준비 (Phase 3)
python scripts/prepare_data.py

# Tokenizer 학습 (Phase 3)
python scripts/train_tokenizer.py --vocab_size 32000

# 모델 학습 (Phase 4)
python scripts/train.py --config configs/training_m4max.yaml

# 텍스트 생성 (Phase 5)
python scripts/generate.py --prompt "안녕하세요"
```
