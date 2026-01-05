# 한국어 LLM 프로젝트 로드맵

## 프로젝트 개요
MacBook Air에서 구동 가능한 소형 한국어 텍스트 생성 LLM 구축

**목표 스펙**
- 모델 크기: 50M-150M 파라미터
- 타겟 디바이스: MacBook Air (M1/M2/M3)
- 언어: 한국어 텍스트 생성
- 아키텍처: 최신 Transformer 변형 (RoPE, SwiGLU, RMSNorm, GQA)

---

## 📋 Phase 1: 프로젝트 셋업 (1-2일)

### 1.1 Repository 구조 생성
```
korean-tiny-llm/
├── docs/
│   ├── architecture.md
│   ├── training-guide.md
│   └── inference-guide.md
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   ├── attention.py
│   │   └── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── optimizer.py
│   └── inference/
│       ├── __init__.py
│       └── generator.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── generate.py
├── tests/
├── configs/
│   ├── model_small.yaml
│   └── training.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

### 1.2 환경 설정
```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate

# 필수 패키지 설치
pip install torch torchvision torchaudio
pip install transformers
pip install tokenizers
pip install datasets
pip install wandb  # 선택: 학습 모니터링
pip install tqdm numpy pyyaml
```

### 1.3 Git 초기화
```bash
git init
git add .
git commit -m "Initial project structure"
```

---

## 📐 Phase 2: 아키텍처 설계 및 구현 (3-5일)

### 2.1 모델 설계 문서 작성
- `docs/architecture.md` 작성
- 하이퍼파라미터 결정
- 메모리 사용량 계산

**추천 모델 설정 (Small)**
```yaml
vocab_size: 32000
hidden_size: 768
num_layers: 12
num_heads: 12
num_kv_heads: 4  # GQA
intermediate_size: 2048  # SwiGLU
max_seq_length: 1024
rope_theta: 10000.0
```

### 2.2 핵심 컴포넌트 구현

**우선순위 1: Tokenizer**
- SentencePiece 또는 BPE 기반
- 한국어 최적화 vocab 32k
- `src/data/tokenizer.py`

**우선순위 2: Model Architecture**
- RoPE Position Encoding
- Multi-Head Attention with GQA
- SwiGLU Feed-Forward
- RMSNorm
- `src/model/transformer.py`

**우선순위 3: Data Pipeline**
- 텍스트 전처리
- 데이터셋 로더
- `src/data/dataset.py`

### 2.3 유닛 테스트 작성
```python
# tests/test_model.py
def test_model_forward():
    pass

def test_attention_shape():
    pass

def test_rope_encoding():
    pass
```

---

## 📊 Phase 3: 데이터 준비 (2-3일)

### 3.1 데이터 소스 선정

**추천 한국어 데이터셋**
1. **AI Hub 공개 데이터** (우선순위 높음)
   - 일상대화 데이터
   - 문서요약 데이터
   
2. **나무위키 덤프** (백업용)
   - 크롤링 후 전처리
   
3. **모두의 말뭉치** (국립국어원)
   - 신문, 문어체 데이터

4. **Korean Wikipedia**
   - 위키미디어 덤프

**목표 데이터 크기**: 최소 1GB, 이상적으로 5-10GB

### 3.2 데이터 전처리 파이프라인

```python
# scripts/prepare_data.py의 단계

# 1. Raw 데이터 다운로드
# 2. 텍스트 정제
#    - HTML 태그 제거
#    - 특수문자 정규화
#    - 중복 제거
# 3. 문장 분리 및 필터링
# 4. Train/Val/Test 분할 (98/1/1)
# 5. Tokenization
# 6. Binary 형식 저장 (효율성)
```

### 3.3 Tokenizer 학습
```bash
python scripts/train_tokenizer.py \
  --input data/raw/*.txt \
  --vocab_size 32000 \
  --output models/tokenizer
```

---

## 🏋️ Phase 4: 학습 (5-10일)

### 4.1 학습 설정

**MacBook Air 최적화**
```yaml
# configs/training.yaml
batch_size: 4  # Gradient accumulation으로 증가
gradient_accumulation_steps: 8  # Effective batch = 32
max_seq_length: 512  # 초기에는 짧게
learning_rate: 3e-4
warmup_steps: 2000
max_steps: 100000
save_steps: 5000
eval_steps: 1000
fp16: true  # Mixed precision
gradient_checkpointing: true  # 메모리 절약
```

### 4.2 학습 실행
```bash
# 단일 GPU 학습
python scripts/train.py \
  --config configs/training.yaml \
  --model_config configs/model_small.yaml \
  --output_dir checkpoints/

# 학습 재개
python scripts/train.py \
  --resume_from checkpoints/checkpoint-5000/
```

### 4.3 모니터링
- Weights & Biases 대시보드
- Loss 트래킹
- Perplexity 측정
- 샘플 생성 확인

### 4.4 학습 팁
- **첫 24시간**: Overfitting 확인용 작은 데이터셋으로 테스트
- **중간 평가**: 5k steps마다 생성 품질 확인
- **학습률 조정**: Loss plateau 시 감소
- **Early stopping**: Validation loss 상승 시 중단

---

## 🚀 Phase 5: 추론 및 최적화 (2-3일)

### 5.1 기본 추론 인터페이스
```python
# scripts/generate.py
from src.inference import Generator

generator = Generator.from_pretrained("checkpoints/final")

text = generator.generate(
    prompt="오늘 날씨가",
    max_length=100,
    temperature=0.8,
    top_p=0.95
)
print(text)
```

### 5.2 추론 최적화
- **KV Cache 구현**: 반복 계산 제거
- **Dynamic Batching**: 효율적 처리
- **양자화 (선택)**: INT8 양자화로 메모리 절약
- **ONNX 변환 (선택)**: 추론 속도 향상

### 5.3 성능 벤치마크
```python
# 측정 항목
- Tokens/sec (생성 속도)
- Latency (첫 토큰까지 시간)
- Memory usage (Peak RAM)
- Model size on disk
```

---

## 📈 Phase 6: 평가 및 개선 (진행형)

### 6.1 정성 평가
- 다양한 프롬프트로 생성 테스트
- 문법 정확도
- 문맥 일관성
- 창의성

### 6.2 정량 평가 (선택)
- Perplexity on test set
- BLEU/ROUGE (특정 태스크)
- KoBERT 기반 유사도

### 6.3 개선 방향
- 더 많은 데이터로 재학습
- 하이퍼파라미터 튜닝
- Instruction tuning (추가 단계)
- RLHF (고급)

---

## 🎯 마일스톤 체크리스트

### Week 1
- [ ] Repository 구조 완성
- [ ] 기본 모델 아키텍처 구현
- [ ] 유닛 테스트 통과
- [ ] 데이터 소스 확보

### Week 2
- [ ] Tokenizer 학습 완료
- [ ] 데이터 전처리 완료
- [ ] 학습 파이프라인 구축
- [ ] 작은 데이터셋으로 Overfitting 확인

### Week 3-4
- [ ] 전체 데이터셋 학습 시작
- [ ] 중간 체크포인트 평가
- [ ] 학습 모니터링 및 조정

### Week 5
- [ ] 학습 완료
- [ ] 추론 인터페이스 구현
- [ ] 성능 최적화
- [ ] 문서화 완료

---

## 🛠 개발 환경 권장사항

### 필수 도구
- Python 3.9+
- PyTorch 2.0+
- Git
- VSCode or PyCharm

### 선택 도구
- Weights & Biases (학습 모니터링)
- Jupyter Notebook (데이터 탐색)
- tmux (장시간 학습용)
- htop (리소스 모니터링)

---

## 📚 참고 자료

### 논문
- "Attention Is All You Need" (Transformer)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- "GLU Variants Improve Transformer"
- "GQA: Training Generalized Multi-Query Transformer"

### 구현 레퍼런스
- Hugging Face Transformers
- nanoGPT (Andrej Karpathy)
- LLaMA implementation
- GPT-NeoX

### 한국어 NLP 리소스
- KorQuAD
- Korean Hate Speech Dataset
- AIHub

---

## ⚠️ 주의사항 및 팁

### MacBook Air 제약사항
- **열 관리**: 장시간 학습 시 쿨링 패드 권장
- **배치 크기**: 메모리 오류 시 줄이기
- **전원 연결**: 학습 중 반드시 전원 연결
- **백그라운드 앱**: 학습 중 다른 앱 최소화

### 시간 절약 팁
- 작은 모델부터 시작 (50M)
- 전체 파이프라인 먼저 검증
- Pretrained embedding 활용 고려
- 학습 데이터 점진적 증가

### 일반적 문제 해결
- **OOM Error**: batch_size 감소, gradient_checkpointing 활성화
- **Slow Training**: Mixed precision 사용, 데이터 로딩 최적화
- **Poor Quality**: 더 많은 데이터, 더 긴 학습
- **Divergence**: Learning rate 감소, Gradient clipping

---

## 📞 다음 단계

1. **GitHub Repository 생성**
   ```bash
   gh repo create korean-tiny-llm --public
   cd korean-tiny-llm
   ```

2. **이 문서를 `docs/roadmap.md`로 저장**

3. **Phase 1 시작**
   - 프로젝트 구조 생성
   - requirements.txt 작성
   - 첫 커밋

준비되면 본격적인 구현을 시작하시면 됩니다!