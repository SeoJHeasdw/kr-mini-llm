# 빠른 시작 가이드

## 🚀 5분 안에 시작하기

### 1. Repository 생성 및 클론
```bash
# GitHub에서 새 repository 생성
gh repo create korean-tiny-llm --public --clone

cd korean-tiny-llm
```

### 2. 프로젝트 구조 자동 생성
```bash
# 디렉토리 생성
mkdir -p docs src/{model,data,training,inference} scripts tests configs

# __init__.py 파일 생성
touch src/__init__.py
touch src/model/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
```

### 3. requirements.txt 생성
```bash
cat > requirements.txt << 'EOF'
# Core
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0

# NLP
transformers>=4.30.0
tokenizers>=0.13.0
sentencepiece>=0.1.99

# Data
datasets>=2.12.0
pyarrow>=12.0.0

# Training
wandb>=0.15.0
pyyaml>=6.0

# Development
pytest>=7.3.0
black>=23.3.0
flake8>=6.0.0

# Optional: Optimization
# onnx>=1.14.0
# onnxruntime>=1.15.0
EOF
```

### 4. 가상환경 설정 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. 기본 설정 파일 생성
```bash
# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/
data/processed/
*.bin
*.pkl

# Models
checkpoints/
models/
*.pth
*.pt

# Logs
logs/
wandb/
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
EOF
```

### 6. README.md 생성
```bash
cat > README.md << 'EOF'
# Korean Tiny LLM

MacBook Air에서 학습 가능한 소형 한국어 텍스트 생성 모델

## Features
- 50M-150M 파라미터 경량 모델
- 최신 Transformer 아키텍처 (RoPE, SwiGLU, GQA, RMSNorm)
- MacBook 최적화 (M1/M2/M3)
- 한국어 특화 토크나이저

## Quick Start
```bash
# 설치
pip install -r requirements.txt

# 데이터 준비
python scripts/prepare_data.py

# 학습
python scripts/train.py --config configs/training.yaml

# 추론
python scripts/generate.py --prompt "안녕하세요"
```

## Project Structure
See `docs/roadmap.md` for detailed information.

## License
MIT
EOF
```

### 7. 첫 커밋
```bash
git add .
git commit -m "Initial project setup with structure and dependencies"
git push -u origin main
```

---

## 📋 Phase별 상세 체크리스트

### Phase 1: 프로젝트 셋업 ✅

#### Day 1
- [ ] GitHub repository 생성
- [ ] 로컬 환경 클론
- [ ] 디렉토리 구조 생성
- [ ] requirements.txt 작성
- [ ] 가상환경 설정 및 패키지 설치
- [ ] .gitignore 설정
- [ ] README.md 작성
- [ ] 첫 커밋 및 푸시

#### Day 2
- [ ] `docs/roadmap.md` 작성
- [ ] `docs/architecture.md` 초안 작성
- [ ] 모델 config 파일 작성 (`configs/model_small.yaml`)
- [ ] 학습 config 파일 작성 (`configs/training.yaml`)

---

### Phase 2: 아키텍처 구현 🏗️

#### Week 1, Day 3-4: Tokenizer
```bash
# 작업 목록
- [ ] src/data/tokenizer.py 구현
  - [ ] SentencePiece 래퍼 클래스
  - [ ] 한국어 normalizer
  - [ ] Vocab 관리
- [ ] scripts/train_tokenizer.py 구현
- [ ] 샘플 데이터로 테스트
- [ ] tests/test_tokenizer.py 작성
```

**코드 스켈레톤**:
```python
# src/data/tokenizer.py
from sentencepiece import SentencePieceProcessor

class KoreanTokenizer:
    def __init__(self, model_path):
        self.sp = SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, text):
        # TODO: 구현
        pass
    
    def decode(self, ids):
        # TODO: 구현
        pass
```

#### Week 1, Day 5-7: Core Model Components

**Day 5: RoPE & RMSNorm**
```bash
- [ ] src/model/layers.py 구현
  - [ ] RotaryPositionEmbedding 클래스
  - [ ] RMSNorm 클래스
- [ ] 단위 테스트
```

**Day 6: Attention Mechanism**
```bash
- [ ] src/model/attention.py 구현
  - [ ] GroupedQueryAttention 클래스
  - [ ] KV cache 준비
- [ ] Shape 테스트
```

**Day 7: Transformer Block**
```bash
- [ ] src/model/transformer.py 구현
  - [ ] TransformerBlock
  - [ ] SwiGLU FFN
  - [ ] 전체 모델 통합
- [ ] Forward pass 테스트
```

#### Week 2, Day 8-9: 통합 및 테스트
```bash
- [ ] 모델 전체 통합
- [ ] 더미 데이터로 forward/backward 테스트
- [ ] 메모리 사용량 프로파일링
- [ ] 문서 업데이트 (architecture.md)
```

---

### Phase 3: 데이터 준비 📊

#### Week 2, Day 10-11: 데이터 수집
```bash
- [ ] 데이터 소스 리서치
  - [ ] AI Hub 계정 생성 및 데이터 다운로드
  - [ ] Korean Wikipedia 덤프 다운로드
  - [ ] 라이선스 확인
- [ ] data/raw/ 디렉토리에 저장
- [ ] 데이터 품질 간단히 확인
```

**추천 다운로드 스크립트**:
```bash
# scripts/download_data.sh
#!/bin/bash

# Korean Wikipedia
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
bzip2 -d kowiki-latest-pages-articles.xml.bz2

# 나머지 소스는 수동 다운로드 또는 API 활용
```

#### Week 2, Day 12: 전처리 파이프라인
```bash
- [ ] src/data/preprocessing.py 구현
  - [ ] HTML/XML 파싱
  - [ ] 텍스트 정제
  - [ ] 문장 분리
  - [ ] 중복 제거
- [ ] scripts/prepare_data.py 통합
```

#### Week 2, Day 13: Tokenizer 학습
```bash
- [ ] 전처리된 데이터로 tokenizer 학습
- [ ] vocab size 실험 (16k, 32k, 64k)
- [ ] 한국어 토큰 분석
- [ ] 최종 tokenizer 저장
```

#### Week 2, Day 14: Dataset 클래스
```bash
- [ ] src/data/dataset.py 구현
  - [ ] PyTorch Dataset 상속
  - [ ] 효율적 데이터 로딩
  - [ ] Collate function
- [ ] Train/Val split
- [ ] DataLoader 테스트
```

---

### Phase 4: 학습 🏋️

#### Week 3, Day 15-16: 학습 파이프라인 구축
```bash
- [ ] src/training/trainer.py 구현
  - [ ] Training loop
  - [ ] Validation loop
  - [ ] Checkpoint 저장/로드
  - [ ] Logging
- [ ] src/training/optimizer.py
  - [ ] AdamW with warmup
  - [ ] Learning rate scheduler
- [ ] scripts/train.py 완성
```

#### Week 3, Day 17: Sanity Check
```bash
- [ ] 작은 데이터셋(1MB)으로 overfitting 테스트
  - [ ] Loss가 0에 가까워지는지 확인
  - [ ] 생성 품질 확인
- [ ] 버그 수정
- [ ] 하이퍼파라미터 초기 튜닝
```

#### Week 3-4, Day 18-28: 본격 학습
```bash
# 매일
- [ ] 학습 진행 모니터링
- [ ] Loss 트렌드 확인
- [ ] 생성 샘플 평가
- [ ] 이상 현상 대응

# 주간
- [ ] Checkpoint 평가
- [ ] Validation perplexity 확인
- [ ] 하이퍼파라미터 조정
```

**학습 모니터링 체크리스트**:
```
일일 체크:
□ Training loss 감소 중?
□ Gradient norm 안정적?
□ GPU/Memory 사용률?
□ 생성 샘플 품질 개선?

주간 체크:
□ Validation loss plateau?
□ Learning rate 조정 필요?
□ 데이터 추가 필요?
□ Early stopping 고려?
```

---

### Phase 5: 추론 최적화 🚀

#### Week 5, Day 29-30: 기본 추론
```bash
- [ ] src/inference/generator.py 구현
  - [ ] Greedy decoding
  - [ ] Top-k/Top-p sampling
  - [ ] Temperature scaling
- [ ] scripts/generate.py CLI 도구
- [ ] 다양한 프롬프트로 테스트
```

#### Week 5, Day 31: 최적화
```bash
- [ ] KV cache 구현 (메모리 효율)
- [ ] Batch inference 지원
- [ ] 추론 속도 벤치마크
- [ ] 메모리 프로파일링
```

#### Week 5, Day 32: 추가 기능
```bash
- [ ] Interactive chat mode
- [ ] API 서버 (FastAPI, 선택)
- [ ] Web demo (Gradio, 선택)
- [ ] 문서화 완료
```

---

## 🎯 각 Phase 완료 기준

### Phase 1 완료 ✅
- 모든 파일 구조 존재
- requirements 설치 완료
- Git repository 정상 작동

### Phase 2 완료 ✅
- 모델 forward pass 작동
- 모든 단위 테스트 통과
- 더미 데이터로 학습 1 step 성공

### Phase 3 완료 ✅
- 최소 1GB 이상의 정제된 데이터
- Tokenizer 학습 완료
- DataLoader에서 배치 정상 출력

### Phase 4 완료 ✅
- 최소 50k steps 학습 완료
- Validation loss < 3.0
- 생성 텍스트가 문법적으로 유의미

### Phase 5 완료 ✅
- 추론 속도 > 10 tokens/sec
- Interactive mode 작동
- 문서화 완료

---

## 🆘 문제 해결 Quick Reference

### 메모리 부족 (OOM)
```yaml
# configs/training.yaml 수정
batch_size: 2  # 4 → 2
gradient_accumulation_steps: 16  # 8 → 16
max_seq_length: 256  # 512 → 256
gradient_checkpointing: true
```

### 학습 속도 느림
```python
# DataLoader 최적화
DataLoader(
    dataset,
    num_workers=2,  # CPU 코어 수에 맞게
    pin_memory=True,
    persistent_workers=True
)
```

### Loss가 감소하지 않음
```yaml
# Learning rate 감소
learning_rate: 1e-4  # 3e-4 → 1e-4

# Warmup 증가
warmup_steps: 5000  # 2000 → 5000

# Gradient clipping
max_grad_norm: 0.5  # 1.0 → 0.5
```

### 생성 품질 낮음
1. 더 많은 데이터 추가
2. 더 긴 시간 학습
3. Validation loss가 충분히 낮은지 확인
4. Temperature 조정 (0.7 ~ 1.0 실험)

---

## 📊 Progress Tracking

### 진행 상황 추적 템플릿
```markdown
## 주차별 진행 상황

### Week 1 (MM/DD - MM/DD)
- [x] Repository 셋업
- [x] 모델 아키텍처 설계
- [ ] Tokenizer 구현
- [ ] 데이터 수집 시작

**이슈**:
- 

**다음 주 계획**:
- 
```

---

## 🎓 학습 리소스

### 필독 자료
1. "Attention Is All You Need" 논문
2. Andrej Karpathy's nanoGPT
3. Hugging Face Transformers 코드

### 추천 강의
- Stanford CS224N (NLP)
- Fast.ai Deep Learning

### 커뮤니티
- Hugging Face Discord
- Reddit r/MachineLearning
- Papers With Code

---

**다음 단계**: Phase 1 체크리스트부터 시작하세요!