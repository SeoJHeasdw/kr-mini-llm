# 📚 kr-mini-llm 문서 가이드

> MacBook Pro 16" M4 Max 기반 한국어 LLM 프로젝트 문서

## 🗂️ 문서 구조

### 📐 아키텍처 및 구조
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** ⭐ **프로젝트 구조**
   - 디렉토리 구조
   - 모델 아키텍처 개요
   - 데이터 흐름
   - 설정 파일 구조

2. **[COMPONENTS.md](COMPONENTS.md)** 
   - 각 컴포넌트 상세 설명
   - 모델 레이어 (Transformer, Attention, Layers)
   - 데이터 처리 (Tokenizer, Dataset)
   - 학습 시스템 (Trainer, Optimizer)
   - 추론 시스템 (Generator)

### 💻 사용 가이드
3. **[USAGE.md](USAGE.md)**
   - 환경 설정
   - 시나리오별 사용법
   - 학습 실행 방법
   - 텍스트 생성 방법
   - 문제 해결

4. **[text_generation.md](text_generation.md)**
   - 텍스트 생성 기술 가이드
   - Temperature 조절
   - 샘플링 전략 (Greedy, Top-K, Top-P, Beam Search)
   - 고급 설정 (Repetition Penalty)
   - 성능 최적화

### 📊 데이터 및 상태
5. **[data_collection_guide.md](data_collection_guide.md)**
   - 한국어 데이터 수집 방법
   - 추천 데이터 소스
   - 데이터 전처리
   - 법적 고려사항

6. **[model_status.md](model_status.md)**
   - 현재 모델 구현 상태
   - 파라미터 정보
   - 학습 전/후 비교

---

## 📊 권장 읽기 순서

### 처음 시작하는 경우
```
1. ARCHITECTURE.md (프로젝트 구조 이해)
   ↓
2. COMPONENTS.md (각 컴포넌트 역할 파악)
   ↓
3. USAGE.md (실제 사용 방법)
   ↓
4. data_collection_guide.md (데이터 준비)
```

### 모델 학습 시작 전
```
1. data_collection_guide.md (데이터 수집)
   ↓
2. USAGE.md > 학습 실행 (학습 시작)
   ↓
3. model_status.md (현재 상태 확인)
```

### 텍스트 생성 최적화
```
1. text_generation.md (생성 전략 이해)
   ↓
2. USAGE.md > 텍스트 생성 (실제 사용)
```

---

## 🎯 모델 크기별 가이드

### Medium 모델 (468M, 권장)
- **설정**: `configs/model_medium.yaml` + `configs/training_m4max.yaml`
- **메모리**: ~10-15GB
- **학습 시간**: ~22시간
- **추론 속도**: > 50 tokens/sec

### Large 모델 (1004M, 도전)
- **설정**: `configs/model_large.yaml` + `configs/training_m4max_large.yaml`
- **메모리**: ~20-25GB
- **학습 시간**: ~44시간
- **추론 속도**: > 30 tokens/sec

### Small 모델 (134M, 레거시)
- **용도**: 테스트 및 디버깅 전용
- **설정**: `configs/model_small.yaml` + `configs/training.yaml`

---

## 🔍 주요 주제별 문서

### 프로젝트 구조
- [ARCHITECTURE.md](ARCHITECTURE.md) - 전체 구조 및 아키텍처
- [COMPONENTS.md](COMPONENTS.md) - 각 파일의 역할

### 모델 아키텍처
- [COMPONENTS.md > 모델](COMPONENTS.md#-srcmodel---모델-아키텍처) - Transformer, Attention, Layers
- [ARCHITECTURE.md > 모델 구조](ARCHITECTURE.md#️-아키텍처-개요) - 전체 모델 구조

### 데이터 처리
- [data_collection_guide.md](data_collection_guide.md) - 데이터 수집 및 전처리
- [COMPONENTS.md > 데이터](COMPONENTS.md#-srcdata---데이터-처리) - Tokenizer, Dataset

### 학습
- [USAGE.md > 실제 학습](USAGE.md#2-실제-학습-medium-모델) - 학습 실행 방법
- [COMPONENTS.md > 학습](COMPONENTS.md#-srctraining---학습-시스템) - Trainer, Optimizer

### 텍스트 생성
- [text_generation.md](text_generation.md) - 생성 전략 및 최적화
- [COMPONENTS.md > 생성](COMPONENTS.md#-srcinference---텍스트-생성) - Generator

### 설정 파일
- [ARCHITECTURE.md > 설정 파일](ARCHITECTURE.md#-설정-파일) - 모델 및 학습 설정
- [USAGE.md > 커스텀 설정](USAGE.md#4-커스텀-설정) - 설정 커스터마이징

---

## 🆘 문제 해결

### 자주 묻는 질문

**Q: PyTorch MPS를 사용할 수 없다고 나옵니다.**
- A: [USAGE.md > 문제 해결](USAGE.md#-문제-해결) 참고

**Q: 메모리 부족 (OOM) 에러가 발생합니다.**
- A: [USAGE.md > 메모리 부족](USAGE.md#메모리-부족) 참고
- batch_size 감소 또는 gradient_checkpointing 활성화

**Q: 학습 속도가 너무 느립니다.**
- A: [USAGE.md > 학습 속도 느림](USAGE.md#학습-속도-느림) 참고
- torch.compile, num_workers 조정

**Q: 텍스트 생성 품질이 낮습니다.**
- A: [text_generation.md > 문제 해결](text_generation.md#문제-해결) 참고
- Temperature, Top-P 조정

---

## 📝 문서 업데이트 이력

### 2026-05-26
- ✅ 문서 구조 정리 및 단순화
- ✅ Phase별 가이드 제거 (중복 제거)
- ✅ 기술 문서 중심으로 재구성
- ✅ 네비게이션 개선

### 2026-02-25
- ✅ M4 Max 기반 전체 문서 업데이트
- ✅ Medium (468M) / Large (1004M) 모델 설정 추가
- ✅ MPS 최적화 가이드 추가

---

## 🔗 관련 링크

### 프로젝트 파일
- [../README.md](../README.md) - 프로젝트 메인 README
- [../QUICKSTART.md](../QUICKSTART.md) - 빠른 시작 가이드
- [../configs/](../configs/) - 설정 파일 디렉토리
- [../src/](../src/) - 소스 코드

### 외부 리소스
- [PyTorch MPS 문서](https://pytorch.org/docs/stable/notes/mps.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [AI Hub (한국어 데이터)](https://aihub.or.kr)

---

## 💡 기여 가이드

문서 개선 제안이나 오류 발견 시:
1. 이슈 생성
2. 수정 사항을 명확히 설명
3. 가능하면 수정된 내용 제안

---

**다음 단계**: [ARCHITECTURE.md](ARCHITECTURE.md)에서 프로젝트 구조를 파악하세요! 🚀