# 빠른 시작 가이드 (M4 Max 최적화)

> **하드웨어**: MacBook Pro 16" M4 Max (36GB RAM, 32-core GPU)
> **목표**: 350M-1B 파라미터 한국어 LLM 구축

## 🚀 5분 안에 시작하기

### 전제 조건
- ✅ MacBook Pro M4 Max (36GB RAM 이상)
- ✅ Python 3.13+
- ✅ 20GB+ 디스크 여유 공간

### 1. 프로젝트 클론 (이미 완료된 경우 스킵)
```bash
cd kr-mini-llm
```

### 2. 프로젝트 구조 확인
```bash
# 이미 생성된 구조 확인
ls -la configs/  # 설정 파일들
ls -la src/      # 소스 코드
ls -la docs/     # 문서
```

**주요 설정 파일:**
- `configs/model_medium.yaml` - 468M 파라미터 (권장)
- `configs/model_large.yaml` - 1004M 파라미터 (도전)
- `configs/training_m4max.yaml` - Medium 모델 학습 설정
- `configs/training_m4max_large.yaml` - Large 모델 학습 설정

### 3. requirements.txt 확인
```bash
cat requirements.txt
```

**이미 포함된 주요 패키지:**
- PyTorch 2.0+ (MPS 지원)
- Transformers, Tokenizers
- SentencePiece (한국어 토크나이저)
- Weights & Biases (학습 모니터링)

### 4. 가상환경 설정 및 패키지 설치
```bash
# 가상환경 생성
python3 -m venv venv

# 활성화
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# PyTorch MPS 지원 확인
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**예상 출력:**
```
PyTorch: 2.x.x
MPS available: True
```

### 5. 설정 테스트
```bash
# 설정 파일 로드 테스트
python3 scripts/test_config.py
```

**예상 출력:**
```
🎉 모든 테스트 통과!
- ✅ Small 모델: ~134M 파라미터
- ✅ Medium 모델: ~468M 파라미터
- ✅ Large 모델: ~1004M 파라미터
```

### 6. 학습 스크립트 테스트
```bash
# Medium 모델 설정 확인
python3 scripts/train.py \
  --config configs/training_m4max.yaml \
  --model_config configs/model_medium.yaml
```

**예상 출력:**
```
🚀 M4 Max Metal Performance Shaders (MPS) 사용
   MPS 메모리 제한: 80%

🤖 모델: 1024 hidden, 24 layers, 16 heads
🏋️  학습: batch=16, effective_batch=64
💻 디바이스: mps (MPS 최적화 활성화)
```

---

## 📋 Phase별 체크리스트 (M4 Max 기준)

### Phase 1: 프로젝트 셋업 ✅ (완료)

- [x] 프로젝트 구조 생성
- [x] M4 Max 최적화 설정 파일 생성
- [x] 문서 작성 (로드맵, 가이드)
- [x] 테스트 스크립트 작성
- [x] MPS 지원 추가

**생성된 파일:**
- `configs/model_medium.yaml`, `model_large.yaml`
- `configs/training_m4max.yaml`, `training_m4max_large.yaml`
- `src/model/config.py` (YAML 로드, 프리셋)
- `scripts/train.py` (MPS 지원)
- `scripts/test_config.py` (테스트)

---

### Phase 2: 아키텍처 구현 🏗️ (3-5일)

**목표**: Medium 모델 (468M) 아키텍처 완성

#### 우선순위 1: 핵심 레이어 (1-2일)
```bash
# src/model/layers.py 구현
- [ ] RMSNorm - Layer normalization 대체
- [ ] RotaryPositionEmbedding (RoPE) - 위치 인코딩
- [ ] SwiGLU - Feed-forward 활성화 함수

# 테스트
python3 -m pytest tests/test_layers.py -v
```

**M4 Max 최적화 포인트:**
- MPS 백엔드 호환성 확인
- FP16 mixed precision 지원
- torch.compile 적용 가능성

#### 우선순위 2: Attention (1-2일)
```bash
# src/model/attention.py 구현
- [ ] GroupedQueryAttention (GQA)
  - num_heads=16, num_kv_heads=4 (Medium)
  - num_heads=24, num_kv_heads=6 (Large)
- [ ] KV cache 구현 (추론 최적화)
- [ ] Causal masking

# 메모리 프로파일링
python3 scripts/profile_attention.py
```

**예상 메모리 (Medium):**
- Attention weights: ~2GB
- KV cache: ~1GB (추론 시)

#### 우선순위 3: Transformer 통합 (1일)
```bash
# src/model/transformer.py 구현
- [ ] TransformerBlock (Attention + FFN)
- [ ] TransformerLM (전체 모델)
- [ ] Forward/Backward pass 검증

# 더미 데이터 테스트
python3 scripts/test_model_forward.py
```

**검증 항목:**
- [ ] 468M 파라미터 확인
- [ ] MPS에서 forward pass 성공
- [ ] Gradient 계산 정상
- [ ] 메모리 사용량 ~10GB 이내

---

### Phase 3: 데이터 준비 📊 (2-3일)

**목표**: 20-50GB 한국어 텍스트 데이터 확보

#### 데이터 소스 (우선순위 순)

**1. AI Hub (최우선)**
```bash
# https://aihub.or.kr
- [ ] 일상대화 데이터 (~5GB)
- [ ] 문서요약 데이터 (~3GB)
- [ ] 뉴스 기사 데이터 (~10GB)
```

**2. Korean Wikipedia**
```bash
# 위키미디어 덤프
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
bzip2 -d kowiki-latest-pages-articles.xml.bz2

# 예상 크기: ~2GB (압축 해제 후 ~10GB)
```

**3. 나무위키 (선택)**
```bash
# 크롤링 필요 (robots.txt 확인)
# 예상 크기: ~20GB
```

**4. 모두의 말뭉치 (국립국어원)**
```bash
# https://corpus.korean.go.kr
- [ ] 신문 말뭉치
- [ ] 문어 말뭉치
```

#### 데이터 전처리 (1일)
```bash
# scripts/prepare_data.py 실행
python3 scripts/prepare_data.py \
  --input data/raw/*.txt \
  --output data/processed/ \
  --min_length 10 \
  --max_length 2048

# 예상 처리 시간: 2-4시간 (M4 Max)
```

**전처리 단계:**
1. HTML/XML 태그 제거
2. 특수문자 정규화
3. 중복 문장 제거
4. Train/Val/Test 분할 (98/1/1)

#### Tokenizer 학습 (1일)
```bash
# 32k vocab SentencePiece 학습
python3 scripts/train_tokenizer.py \
  --input data/processed/train.txt \
  --vocab_size 32000 \
  --model_type bpe \
  --output models/tokenizer

# 예상 학습 시간: 1-2시간
```

**검증:**
```python
from src.data.tokenizer import KoreanTokenizer

tokenizer = KoreanTokenizer("models/tokenizer.model")
text = "안녕하세요, M4 Max에서 학습하는 한국어 LLM입니다."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")
```
```bash
# scripts/download_data.sh
#!/bin/bash

# Korean Wikipedia
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
bzip2 -d kowiki-latest-pages-articles.xml.bz2

# 나머지 소스는 수동 다운로드 또는 API 활용
```

---

### Phase 4: 학습 🏋️ (2-3주)

**목표**: Medium 모델 200k steps 학습 완료 (~22시간)

#### 학습 파이프라인 구축 (1-2일)
```bash
# src/training/trainer.py 구현
- [ ] Training loop (MPS 최적화)
- [ ] Validation loop
- [ ] Checkpoint 저장/로드
- [ ] W&B 로깅 (선택)

# src/training/optimizer.py
- [ ] AdamW (betas=[0.9, 0.95])
- [ ] Cosine learning rate scheduler
- [ ] Gradient clipping (max_norm=1.0)
```

#### Sanity Check (1일)
```bash
# 작은 데이터셋으로 overfitting 테스트
python3 scripts/train.py \
  --config configs/training_m4max.yaml \
  --model_config configs/model_medium.yaml \
  --max_steps 1000 \
  --data_size 1MB

# 확인 사항:
- [ ] Loss가 0에 가까워지는가?
- [ ] 메모리 사용량 ~10GB 이내?
- [ ] 학습 속도 2-3 steps/sec?
```

#### 본격 학습 (Medium 모델)
```bash
# 학습 시작
python3 scripts/train.py \
  --config configs/training_m4max.yaml \
  --model_config configs/model_medium.yaml

# 예상 시간: ~22시간 (200k steps)
# 예상 메모리: ~10-15GB
# 학습 속도: 2-3 steps/sec
```

**M4 Max 최적화 설정 (이미 적용됨):**
- ✅ Batch size: 16
- ✅ Gradient accumulation: 4 (effective batch=64)
- ✅ Sequence length: 2048
- ✅ Mixed precision: FP16
- ✅ Gradient checkpointing: OFF (속도 우선)
- ✅ torch.compile: ON
- ✅ MPS 메모리 제한: 80%

#### 학습 모니터링
```bash
# W&B 대시보드 (선택)
# configs/training_m4max.yaml에서 wandb.enabled: true

# 로컬 모니터링
tail -f checkpoints/medium/train.log

# 체크포인트 확인
ls -lh checkpoints/medium/checkpoint-*
```

**일일 체크리스트:**
- [ ] Training loss 감소 추세?
- [ ] Validation perplexity < 20?
- [ ] MPS 메모리 사용률 < 80%?
- [ ] 생성 샘플 품질 개선?
- [ ] Gradient norm 안정적? (< 5.0)

**주간 체크리스트:**
- [ ] Checkpoint 평가 (5k steps마다)
- [ ] Learning rate 조정 필요?
- [ ] 데이터 추가 필요?
- [ ] Early stopping 고려?

#### 학습 재개 (중단 시)
```bash
python3 scripts/train.py \
  --config configs/training_m4max.yaml \
  --model_config configs/model_medium.yaml \
  --resume_from checkpoints/medium/checkpoint-50000
```

---

### Phase 5: 추론 최적화 🚀 (2-3일)

**목표**: 추론 속도 > 50 tokens/sec (Medium)

#### 기본 추론 구현 (1일)
```bash
# src/inference/generator.py 구현
- [ ] Greedy decoding
- [ ] Top-k/Top-p sampling
- [ ] Temperature scaling
- [ ] KV cache (메모리 효율)

# 테스트
python3 scripts/generate.py \
  --checkpoint checkpoints/medium/final \
  --prompt "안녕하세요, M4 Max에서" \
  --max_length 100
```

#### 추론 최적화 (1일)
```bash
# 최적화 기법
- [ ] KV cache 구현 (필수)
- [ ] torch.compile 적용
- [ ] Batch inference 지원
- [ ] MPS 최적화

# 벤치마크
python3 scripts/benchmark_inference.py
```

**예상 성능 (Medium, M4 Max):**
- Tokens/sec: 50-80
- Latency (첫 토큰): < 100ms
- Memory: ~5GB

#### Interactive Demo (선택)
```bash
# Gradio 웹 인터페이스
pip install gradio
python3 scripts/demo.py --port 7860

# 또는 CLI
python3 scripts/chat.py
```
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