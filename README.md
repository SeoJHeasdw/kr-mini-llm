# kr-mini-llm

MacBook Air에서 **아주 작은 LLM(장난감 수준, 문자 단위 GPT)** 을 먼저 “끝까지 연결”해보는 프로젝트입니다.  
이 레포는 docs의 로드맵/가이드를 기반으로, **초기 구축(Phase 1) + 로컬에서 바로 돌아가는 스모크 테스트**까지 포함합니다.

## Quick Start (가장 간단한 초기 구축)

아래 커맨드는 **레포 루트 디렉토리에서** 실행하세요.

```bash
# 1) 가상환경
python3 -m venv venv
source venv/bin/activate
pip install -U pip

# 2) 최소 의존성 설치(권장: 빠름)
pip install -r requirements-minimal.txt

# 3) 샘플 데이터/토크나이저 생성 → 초간단 학습 → 생성
python scripts/prepare_data.py
python scripts/train.py --config configs/training_tiny.yaml
python scripts/generate.py --prompt "안녕하세요. 오늘은"
```

## What you get
- **완전 로컬 실행**: 데이터 다운로드/외부 모델 다운로드 없이 학습→저장→생성까지 한 번에 확인
- **Mac 최적화 포인트**: 가능하면 `mps`(Apple Silicon)로 자동 선택
- **다음 단계 연결**: docs의 Phase 2+ (SentencePiece, RoPE/RMSNorm/SwiGLU/GQA, 데이터 파이프라인)로 확장하기 쉬운 구조

## Project Structure (핵심)
```
.
├── configs/                  # tiny 실행용 설정
├── docs/                     # 가이드/로드맵
├── scripts/                  # prepare/train/generate 엔트리포인트
└── src/
    ├── data/                 # CharTokenizer (초기 스모크 테스트용)
    └── model/                # TinyGPT (초기 스모크 테스트용)
```

## Next
docs의 `Korean_llm_project_roadmap.md` / `Quick_start_guide.md`의 Phase 2부터 진행하면 됩니다.
