# 사용 가이드

실제 사용 시나리오별 가이드입니다.

## 🚀 시작하기

### 환경 설정
```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 📝 시나리오별 가이드

### 1. 빠른 테스트 (5분)

```bash
# 프로파일 기반 간편 실행
python scripts/train_simple.py --profile quick_test
```

**설명**:
- Small 모델 (150M)
- 빠른 반복 테스트용
- 메모리: ~3-5GB

### 2. 실제 학습 (Medium 모델)

```bash
# 1단계: 데이터 토큰화 (한 번만)
python scripts/tokenize_data.py \
    --input data/raw/train.txt \
    --output data/processed/train_tokens.npy \
    --tokenizer tokenizer/korean_tokenizer.model \
    --max_length 2048

# 2단계: 학습 시작
python scripts/train_simple.py --profile medium_full

# 3단계: 체크포인트 재개 (중단 시)
python scripts/train_simple.py --profile medium_full \
    --resume checkpoints/medium/step_50000.pt
```

**예상 시간**:
- 토큰화: 10-30분 (데이터 크기에 따라)
- 학습: ~22시간 (200k steps)

### 3. 텍스트 생성

```bash
# 기본 생성
python scripts/generate_simple.py \
    --profile generate_default \
    --prompt "안녕하세요"

# 창의적 생성 (높은 temperature)
python scripts/generate_simple.py \
    --profile generate_default \
    --prompt "옛날 옛적에" \
    --temperature 1.5

# 긴 텍스트 생성
python scripts/generate_simple.py \
    --profile generate_default \
    --prompt "한국의 역사는" \
    --max-new-tokens 200
```

### 4. 커스텀 설정

```bash
# 1. 새 프로파일 생성
cp configs/profiles/medium_full.yaml configs/profiles/my_experiment.yaml

# 2. 프로파일 수정 (에디터에서)
# - 데이터 경로 변경
# - 하이퍼파라미터 조정
# - 출력 디렉토리 변경

# 3. 실행
python scripts/train_simple.py --profile my_experiment
```

## 🔧 고급 사용법

### 상세 제어 학습

```bash
python scripts/train.py \
    --model_config configs/model_medium.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/processed/train_tokens.npy \
    --valid_data data/processed/valid_tokens.npy \
    --output_dir checkpoints/medium \
    --tokenizer tokenizer/korean_tokenizer.model \
    --device mps
```

### 상세 제어 생성

```bash
python scripts/generate.py \
    --checkpoint checkpoints/medium/final.pt \
    --config configs/model_medium.yaml \
    --tokenizer tokenizer/korean_tokenizer.model \
    --prompt "안녕하세요" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9 \
    --top-k 50 \
    --repetition-penalty 1.1
```

## 📊 모니터링

### 학습 진행 확인

```bash
# 로그 확인
tail -f checkpoints/medium/train.log

# 체크포인트 확인
ls -lh checkpoints/medium/
```

### 모델 파라미터 수 확인

```bash
python scripts/count_params.py --config configs/model_medium.yaml
```

## 🐛 문제 해결

### 메모리 부족

```yaml
# configs/training_m4max.yaml 수정
train:
  batch_size: 4  # 8 → 4로 감소
  gradient_accumulation_steps: 16  # 8 → 16으로 증가
```

### 학습 속도 느림

```yaml
# configs/training_m4max.yaml 수정
train:
  num_workers: 2  # 4 → 2로 감소
  prefetch_factor: 2  # 메모리 절약
```

### 체크포인트 손상

```bash
# 이전 체크포인트로 재개
python scripts/train_simple.py --profile medium_full \
    --resume checkpoints/medium/step_45000.pt
```

## 💡 팁

### 1. 데이터 준비
- 토큰화된 파일 (.npy) 사용 권장
- 텍스트 파일은 느림 (매번 토큰화)

### 2. 학습 최적화
- Mixed Precision 활성화 (기본값)
- Gradient Accumulation으로 유효 배치 크기 증가
- 적절한 warmup_steps 설정

### 3. 생성 품질
- Temperature 0.7-0.9: 균형잡힌 생성
- Temperature 1.0-1.5: 창의적 생성
- Top-P 0.9: 다양성과 품질 균형

### 4. 체크포인트 관리
- 정기적으로 백업
- 디스크 공간 확인
- 최고 성능 체크포인트 별도 저장

## 📚 더 알아보기

- [ARCHITECTURE.md](ARCHITECTURE.md) - 프로젝트 구조
- [COMPONENTS.md](COMPONENTS.md) - 컴포넌트 상세
- [QUICKSTART.md](../QUICKSTART.md) - 빠른 시작
