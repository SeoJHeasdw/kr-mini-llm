# 텍스트 생성 가이드

Korean LLM의 텍스트 생성 기능 사용 가이드입니다.

## 📋 목차

1. [개요](#개요)
2. [기본 사용법](#기본-사용법)
3. [Temperature 조절](#temperature-조절)
4. [샘플링 전략](#샘플링-전략)
5. [고급 설정](#고급-설정)
6. [예제](#예제)

## 개요

Korean LLM은 다양한 텍스트 생성 전략을 지원합니다:

- **Greedy Decoding**: 가장 확률 높은 토큰 선택
- **Temperature Sampling**: 확률 분포 조정으로 다양성 제어
- **Top-K Sampling**: 상위 K개 토큰 중에서 샘플링
- **Top-P (Nucleus) Sampling**: 누적 확률 기반 샘플링
- **Beam Search**: 여러 후보 동시 탐색
- **Repetition Penalty**: 반복 억제

## 기본 사용법

### 명령줄에서 실행

```bash
python scripts/generate.py \
    --prompt "안녕하세요" \
    --checkpoint checkpoints/final.pt \
    --config configs/model_small.yaml \
    --max-new-tokens 100
```

### Python 코드에서 사용

```python
from src.model.transformer import TransformerLM
from src.model.config import TransformerConfig
from src.inference.generator import Generator, GenerationConfig

# 모델 로드
cfg = TransformerConfig.from_yaml("configs/model_small.yaml")
model = TransformerLM(cfg)
model.load_state_dict(torch.load("checkpoints/final.pt"))

# Generator 생성
generator = Generator(model, tokenizer, device)

# 텍스트 생성
result = generator.generate(
    prompt="안녕하세요",
    max_new_tokens=100,
    temperature=1.0
)
print(result)
```

## Temperature 조절

Temperature는 생성 다양성을 제어하는 핵심 파라미터입니다.

### Temperature 값의 의미

| Temperature | 특징 | 사용 사례 |
|------------|------|----------|
| **0.1 - 0.3** | 거의 deterministic, 안정적 | 정확한 답변, 번역 |
| **0.5 - 0.7** | 약간 보수적, 일관성 높음 | 기술 문서, 요약 |
| **1.0** | 균형잡힌 샘플링 (기본값) | 일반적인 대화 |
| **1.2 - 1.5** | 더 다양하고 창의적 | 창작, 브레인스토밍 |
| **1.8 - 2.0** | 매우 다양, 예측 불가능 | 실험적 생성 |

### 예제

```bash
# 안정적인 생성 (낮은 temperature)
python scripts/generate.py \
    --prompt "한국의 수도는" \
    --temperature 0.3 \
    --max-new-tokens 50

# 창의적인 생성 (높은 temperature)
python scripts/generate.py \
    --prompt "옛날 옛적에" \
    --temperature 1.5 \
    --max-new-tokens 100
```

## 샘플링 전략

### 1. Greedy Decoding

항상 가장 확률 높은 토큰을 선택합니다.

```bash
python scripts/generate.py \
    --prompt "안녕하세요" \
    --no-sample \
    --max-new-tokens 50
```

**장점**: Deterministic, 재현 가능  
**단점**: 다양성 부족, 반복 가능성

### 2. Top-K Sampling

상위 K개 토큰 중에서만 샘플링합니다.

```bash
python scripts/generate.py \
    --prompt "오늘 날씨는" \
    --temperature 1.0 \
    --top-k 50 \
    --max-new-tokens 50
```

**장점**: 낮은 확률 토큰 제거, 품질 향상  
**단점**: K 값 선택이 중요

**권장 K 값**:
- `k=10-20`: 매우 보수적
- `k=40-50`: 균형잡힌 (권장)
- `k=100+`: 다양성 중시

### 3. Top-P (Nucleus) Sampling

누적 확률이 P를 초과할 때까지의 토큰 중에서 샘플링합니다.

```bash
python scripts/generate.py \
    --prompt "인공지능은" \
    --temperature 1.0 \
    --top-p 0.9 \
    --max-new-tokens 50
```

**장점**: 동적으로 후보 크기 조정  
**단점**: 계산 비용 약간 높음

**권장 P 값**:
- `p=0.8`: 보수적
- `p=0.9`: 균형잡힌 (권장)
- `p=0.95`: 다양성 중시

### 4. Top-K + Top-P 조합

두 전략을 함께 사용하여 최상의 결과를 얻습니다.

```bash
python scripts/generate.py \
    --prompt "미래의 기술은" \
    --temperature 1.0 \
    --top-k 50 \
    --top-p 0.9 \
    --max-new-tokens 100
```

**권장 조합**:
- 일반 대화: `temperature=1.0, top_k=50, top_p=0.9`
- 창의적 생성: `temperature=1.3, top_k=100, top_p=0.95`
- 안정적 생성: `temperature=0.7, top_k=40, top_p=0.85`

### 5. Beam Search

여러 후보를 동시에 탐색하여 최적 시퀀스를 찾습니다.

```bash
python scripts/generate.py \
    --prompt "한국의 역사는" \
    --num-beams 5 \
    --max-new-tokens 100
```

**장점**: 전역 최적화, 일관성 높음  
**단점**: 계산 비용 높음, 다양성 낮음

**권장 빔 개수**:
- `beams=1`: 샘플링 (기본)
- `beams=3-5`: 균형잡힌 품질
- `beams=10+`: 최고 품질 (느림)

## 고급 설정

### Repetition Penalty

반복을 억제하여 더 다양한 텍스트를 생성합니다.

```bash
python scripts/generate.py \
    --prompt "인공지능" \
    --temperature 1.0 \
    --repetition-penalty 1.2 \
    --max-new-tokens 100
```

**권장 값**:
- `1.0`: 페널티 없음 (기본)
- `1.1-1.3`: 약간 억제 (권장)
- `1.5+`: 강하게 억제

### Length Penalty (Beam Search)

생성 길이에 대한 페널티를 조정합니다.

```python
config = GenerationConfig(
    num_beams=5,
    length_penalty=1.2,  # >1.0: 긴 시퀀스 선호
    max_new_tokens=100
)
```

### 랜덤 시드

재현 가능한 생성을 위해 시드를 설정합니다.

```bash
python scripts/generate.py \
    --prompt "안녕하세요" \
    --seed 42 \
    --temperature 1.0 \
    --max-new-tokens 50
```

## 예제

### 예제 1: 질문 답변 (안정적)

```bash
python scripts/generate.py \
    --prompt "한국의 수도는 어디인가요?" \
    --temperature 0.3 \
    --top-k 40 \
    --top-p 0.85 \
    --max-new-tokens 50
```

### 예제 2: 창의적 글쓰기

```bash
python scripts/generate.py \
    --prompt "옛날 옛적에 한 마을에" \
    --temperature 1.3 \
    --top-k 100 \
    --top-p 0.95 \
    --repetition-penalty 1.2 \
    --max-new-tokens 200
```

### 예제 3: 기술 문서 생성

```bash
python scripts/generate.py \
    --prompt "Python에서 리스트를 정렬하는 방법:" \
    --temperature 0.7 \
    --num-beams 3 \
    --max-new-tokens 150
```

### 예제 4: 대화 생성

```bash
python scripts/generate.py \
    --prompt "사용자: 오늘 날씨 어때요?\n어시스턴트:" \
    --temperature 1.0 \
    --top-k 50 \
    --top-p 0.9 \
    --max-new-tokens 100
```

## 성능 최적화

### M4 Max 최적화

```bash
# MPS 디바이스 사용 (M4 Max)
python scripts/generate.py \
    --prompt "안녕하세요" \
    --device mps \
    --max-new-tokens 100
```

### 배치 생성

여러 프롬프트를 한 번에 처리:

```python
prompts = [
    "안녕하세요",
    "오늘 날씨는",
    "인공지능이란"
]

results = generator.generate_batch(
    prompts,
    config=GenerationConfig(
        temperature=1.0,
        max_new_tokens=50
    )
)
```

## 문제 해결

### 반복되는 텍스트

**해결책**: Repetition penalty 증가

```bash
--repetition-penalty 1.3
```

### 일관성 없는 텍스트

**해결책**: Temperature 감소 또는 beam search 사용

```bash
--temperature 0.7 --num-beams 3
```

### 너무 보수적인 생성

**해결책**: Temperature 증가 또는 top-p 증가

```bash
--temperature 1.2 --top-p 0.95
```

### 생성 속도 느림

**해결책**:
1. Beam search 대신 샘플링 사용
2. max_new_tokens 감소
3. 더 작은 모델 사용

## 참고 자료

- [Transformer 아키텍처](architecture.md)
- [모델 설정](Configuration_update_plan.md)
- [학습 가이드](Setup_and_testing.md)

---

**Made with Bob** 🤖
