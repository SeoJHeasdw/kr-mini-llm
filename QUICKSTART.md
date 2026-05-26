# 빠른 실행 가이드

## 기존 방식 (복잡함) ❌
```bash
python scripts/train.py \
    --model_config configs/model_small.yaml \
    --training_config configs/training_m4max.yaml \
    --train_data data/kullm/tokenized.npy \
    --valid_data data/kullm/tokenized.npy \
    --output_dir checkpoints \
    --tokenizer tokenizer/korean_tokenizer.model
```

## 새 방식 (간단함) ✅
```bash
# 학습
python scripts/train_simple.py --profile quick_test

# 생성
python scripts/generate_simple.py --profile generate_default --prompt "안녕하세요"
```

---

## 사용 가능한 프로파일

### 학습
- `quick_test` - Small 모델, 빠른 테스트용
- `medium_full` - Medium 모델, 실제 학습용 (권장)

### 생성
- `generate_default` - 기본 생성 설정

---

## 옵션 오버라이드

```bash
# 디바이스 변경
python scripts/train_simple.py --profile quick_test --device cpu

# 체크포인트 재개
python scripts/train_simple.py --profile medium_full --resume checkpoints/medium/step_10000.pt

# 생성 파라미터 조정
python scripts/generate_simple.py --profile generate_default --prompt "한국어는" --temperature 1.2
```

---

## 새 프로파일 만들기

`configs/profiles/` 디렉토리에 YAML 파일 생성:

```yaml
# configs/profiles/my_profile.yaml
model:
  config: configs/model_medium.yaml

training:
  config: configs/training_m4max.yaml

data:
  train: data/my_data/train.npy
  valid: data/my_data/valid.npy
  tokenizer: tokenizer/korean_tokenizer.model

output:
  dir: checkpoints/my_experiment

device: mps
```

사용:
```bash
python scripts/train_simple.py --profile my_profile