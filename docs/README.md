# 📚 kr-mini-llm 문서 가이드

> MacBook Pro 16" M4 Max 기반 한국어 LLM 프로젝트 문서

## 🗂️ 문서 구조

### 🚀 시작하기
1. **[Quick_start_guide.md](Quick_start_guide.md)** ⭐ **먼저 읽기**
   - 5분 안에 시작하기
   - 환경 설정 및 테스트
   - Phase별 체크리스트 (M4 Max 최적화)

2. **[Setup_and_testing.md](Setup_and_testing.md)**
   - 상세 설정 가이드
   - 테스트 실행 방법
   - 문제 해결 (Troubleshooting)

### 📋 프로젝트 계획
3. **[Korean_llm_project_roadmap.md](Korean_llm_project_roadmap.md)**
   - 전체 프로젝트 로드맵 (6 Phases)
   - M4 Max 기준 업데이트
   - 마일스톤 및 체크리스트

4. **[Configuration_update_plan.md](Configuration_update_plan.md)**
   - 설정 파일 업데이트 계획
   - 코드 수정 사항
   - 실행 순서

### 🔧 최적화 가이드
5. **[M4_MAX_optimization_plan.md](M4_MAX_optimization_plan.md)** ⭐ **상세 가이드**
   - M4 Max 하드웨어 스펙 분석
   - 모델 크기별 최적화 전략
   - 메모리 사용량 추정
   - 성능 벤치마크 목표

---

## 📊 권장 읽기 순서

### 처음 시작하는 경우
```
1. Quick_start_guide.md (5분 안에 시작)
   ↓
2. Setup_and_testing.md (환경 설정 및 테스트)
   ↓
3. M4_MAX_optimization_plan.md (최적화 전략 이해)
   ↓
4. Korean_llm_project_roadmap.md (전체 계획 파악)
```

### Phase 2 (모델 구현) 시작 전
```
1. M4_MAX_optimization_plan.md (모델 크기 결정)
   ↓
2. Configuration_update_plan.md (설정 확인)
   ↓
3. Korean_llm_project_roadmap.md > Phase 2 섹션
```

### 학습 시작 전 (Phase 4)
```
1. M4_MAX_optimization_plan.md > 학습 설정 섹션
   ↓
2. Setup_and_testing.md > 문제 해결 섹션
   ↓
3. Quick_start_guide.md > Phase 4 체크리스트
```

---

## 🎯 모델 크기별 가이드

### Medium 모델 (468M, 권장)
- **설정**: `configs/model_medium.yaml` + `configs/training_m4max.yaml`
- **메모리**: ~10-15GB
- **학습 시간**: ~22시간
- **추론 속도**: > 50 tokens/sec
- **문서**: [M4_MAX_optimization_plan.md](M4_MAX_optimization_plan.md#medium-모델-권장)

### Large 모델 (1004M, 도전)
- **설정**: `configs/model_large.yaml` + `configs/training_m4max_large.yaml`
- **메모리**: ~20-25GB
- **학습 시간**: ~44시간
- **추론 속도**: > 30 tokens/sec
- **문서**: [M4_MAX_optimization_plan.md](M4_MAX_optimization_plan.md#large-모델-도전)

### Small 모델 (134M, 레거시)
- **용도**: 테스트 및 디버깅 전용
- **설정**: `configs/model_small.yaml` + `configs/training.yaml`
- **문서**: 실제 학습에는 권장하지 않음

---

## 🔍 주요 주제별 문서

### 환경 설정
- [Quick_start_guide.md > 5분 안에 시작하기](Quick_start_guide.md#🚀-5분-안에-시작하기)
- [Setup_and_testing.md > 환경 설정](Setup_and_testing.md#🚀-빠른-시작)

### MPS (Metal Performance Shaders) 최적화
- [M4_MAX_optimization_plan.md > MPS 활용](M4_MAX_optimization_plan.md#1-metal-performance-shaders-mps-활용)
- [Setup_and_testing.md > MPS 사용 불가 시](Setup_and_testing.md#mps-사용-불가-시)

### 메모리 관리
- [M4_MAX_optimization_plan.md > 메모리 사용량 추정](M4_MAX_optimization_plan.md#📊-메모리-사용량-추정)
- [Setup_and_testing.md > 메모리 부족 (OOM) 시](Setup_and_testing.md#메모리-부족-oom-시)

### 데이터 준비
- [Quick_start_guide.md > Phase 3](Quick_start_guide.md#phase-3-데이터-준비-📊-2-3일)
- [Korean_llm_project_roadmap.md > Phase 3](Korean_llm_project_roadmap.md#📊-phase-3-데이터-준비-2-3일)

### 학습 최적화
- [M4_MAX_optimization_plan.md > 성능 최적화 전략](M4_MAX_optimization_plan.md#🚀-성능-최적화-전략)
- [Quick_start_guide.md > Phase 4](Quick_start_guide.md#phase-4-학습-🏋️-2-3주)

### 추론 최적화
- [Quick_start_guide.md > Phase 5](Quick_start_guide.md#phase-5-추론-최적화-🚀-2-3일)
- [M4_MAX_optimization_plan.md > 추가 최적화 옵션](M4_MAX_optimization_plan.md#🔧-추가-최적화-옵션)

---

## 🆘 문제 해결

### 자주 묻는 질문

**Q: PyTorch MPS를 사용할 수 없다고 나옵니다.**
- A: [Setup_and_testing.md > MPS 사용 불가 시](Setup_and_testing.md#mps-사용-불가-시) 참고

**Q: 메모리 부족 (OOM) 에러가 발생합니다.**
- A: [Setup_and_testing.md > 메모리 부족 시](Setup_and_testing.md#메모리-부족-oom-시) 참고
- batch_size 감소 또는 gradient_checkpointing 활성화

**Q: 학습 속도가 너무 느립니다.**
- A: [M4_MAX_optimization_plan.md > 성능 최적화](M4_MAX_optimization_plan.md#🚀-성능-최적화-전략) 참고
- torch.compile, num_workers 조정

**Q: Medium과 Large 중 어떤 모델을 선택해야 하나요?**
- A: [M4_MAX_optimization_plan.md > 결론](M4_MAX_optimization_plan.md#🎉-결론) 참고
- 36GB RAM에서는 Medium 권장

**Q: 레거시 설정 파일(model_small.yaml)은 언제 사용하나요?**
- A: 테스트 및 디버깅 전용, 실제 학습에는 권장하지 않음
- 빠른 프로토타입 테스트나 모델 아키텍처 검증용

---

## 📝 문서 업데이트 이력

### 2026-02-25
- ✅ M4 Max 기반 전체 문서 업데이트
- ✅ Medium (468M) / Large (1004M) 모델 설정 추가
- ✅ MPS 최적화 가이드 추가
- ✅ 레거시 파일 주석 추가
- ✅ Quick_start_guide.md M4 Max 기준 재작성

### 초기 버전
- MacBook Air 기반 설정 (50-150M 파라미터)

---

## 🔗 관련 링크

### 프로젝트 파일
- [../README.md](../README.md) - 프로젝트 메인 README
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

**다음 단계**: [Quick_start_guide.md](Quick_start_guide.md)에서 시작하세요! 🚀