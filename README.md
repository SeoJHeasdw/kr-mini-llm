# kr-mini-llm

MacBook Air에서 **나만의 작은 한국어 LLM**을 개발하기 위한 프로젝트입니다.  
이 레포는 `docs/`의 로드맵/가이드를 기반으로, **초기 구축(Phase 1) 스캐폴딩**을 제공합니다.

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

## Next
docs의 `Korean_llm_project_roadmap.md` / `Quick_start_guide.md`의 Phase 2부터 진행하면 됩니다.
