# ResNet18 콩잎 3클래스 Fine-tuning 결과보고서

> 학습 수치는 실제 `training_report.json`, 터미널 로그, `tools/cnn_predict.py` 출력에서 옮겨 적었다.

- 평가 대상: 콩잎 질병 이미지 3종 분류 (이미지 분류, Image Classification)
- 기준 모델: `microsoft/resnet-18` (ImageNet 1,000클래스 사전학습)
- 학습 방식: **Head-only Fine-tuning** (Backbone 고정, 3클래스 분류 헤드만 학습)
- 학습 코드: `training/finetune.py` — **제공 원본 그대로 사용, 코드 수정 없음**
- 입력 데이터: `dataset/train`, `dataset/val` / 보관 데이터: `dataset/test` (학습·모델 선택에 미사용)
- **서비스 연결 모델: 실행 B (30 epoch)** → `artifacts/beans_resnet18`

---

## 1. 데이터 구성

| 클래스 | Train | Validation | Test |
|---|---:|---:|---:|
| angular_leaf_spot | 345 | 44 | 43 |
| bean_rust | 348 | 45 | 43 |
| healthy | 341 | 44 | 42 |
| **합계** | **1,034** | **133** | **128** |

- Train·Validation 합계는 `training_report.json`의 `train_count`, `val_count`와 일치한다.
- 원본: Hugging Face `AI-Lab-Makerere/beans` (`train.zip`, `validation.zip`, `test.zip`)

| 원본 폴더 | 코드 실행용 폴더 | 용도 |
|---|---|---|
| `train/` | `dataset/train/` | 가중치 학습 |
| `validation/` | `dataset/val/` | epoch별 성능 확인, 최고 모델 선택 |
| `test/` | `dataset/test/` | 학습 미사용 (최종 평가용 보관) |

- 클래스 라벨: 폴더 이름을 정답으로 사용 (코드가 알파벳 순으로 0, 1, 2 번호 부여)
- 데이터 누출 점검: `dataset/` 전체 파일의 MD5를 비교한 결과 **중복 파일 없음** (`find dataset -type f -exec md5sum {} + | sort | uniq -w32 -D` 출력 없음)

```bash
# Test 개수 확인
for c in angular_leaf_spot bean_rust healthy; do echo "test/$c: $(find dataset/test/$c -type f | wc -l)"; done
# split 간 중복 파일 확인 (출력이 없으면 중복 없음)
find dataset -type f -exec md5sum {} + | sort | uniq -w32 -d
```

---

## 2. 실행 설정

두 번 학습하였다. 두 실행의 차이는 **실행 옵션 `--epochs`뿐**이며 학습 코드는 수정하지 않았다. 평가가이드 기준으로 epoch 변경은 "실행 옵션 차이이며 기본 코드 수정이 아니다".

| 파라미터 | 실행 A | 실행 B (서비스 모델) |
|---|---|---|
| base_model | microsoft/resnet-18 | microsoft/resnet-18 |
| epochs | 3 | **30** |
| batch_size | 8 | 8 |
| learning_rate | 0.0002 | 0.0002 |
| freeze_backbone | True | True (가중치 비교로 확인 ※) |
| seed | 42 (코드 고정값) | 42 (코드 고정값) |
| 실행 장치 | GPU (NVIDIA GeForce RTX 3060 12GB) | GPU (NVIDIA GeForce RTX 3060 12GB) |
| 학습 당시 저장 경로 | `artifacts/beans_resnet18` → 정리 후 `artifacts/beans_resnet18_e3` | `artifacts/beans_resnet18_e30` → 정리 후 `artifacts/beans_resnet18`로 복사 |

```bash
# 실행 A — 01 문서 기본 명령
python -m training.finetune --data dataset --output artifacts/beans_resnet18 \
  --epochs 3 --batch-size 8 --freeze-backbone

# 실행 B — epoch만 변경, 별도 경로 저장 (실제 명령은 history | grep finetune 으로 확인)
python -m training.finetune --data dataset --output artifacts/beans_resnet18_e30 \
  --epochs 30 --batch-size 8 --freeze-backbone
```

※ 실행 B의 명령 기록이 남지 않아, 저장된 모델의 Backbone 가중치를 원본 `microsoft/resnet-18`과 비교하였다. BatchNorm 이동 통계를 제외한 모든 Backbone 가중치가 **원본과 완전히 같아**(`torch.equal` = True) Backbone이 고정(`--freeze-backbone`)되었음을 확인하였다.

```bash
python -c "
from transformers import AutoModelForImageClassification as M; import torch
a=M.from_pretrained('microsoft/resnet-18').resnet.state_dict()
b=M.from_pretrained('artifacts/beans_resnet18').resnet.state_dict()
print(all(torch.equal(a[k],b[k]) for k in a if 'num_batches' not in k and 'running' not in k))"
# 출력: True
```

**epoch 변경 이유:** 실행 A의 3 epoch 종료 시점까지 Train Loss가 계속 감소하고 Validation Accuracy가 계속 상승하여, 학습이 충분히 수렴하지 않은 상태(underfitting)로 판단하였다.

---

## 3. 학습 결과

### 3-1. 요약

| 항목 | 실행 A (3 epoch) | 실행 B (30 epoch) |
|---|---:|---:|
| Best Validation Accuracy | 0.8421 (112/133) | **0.9549 (127/133)** |
| Best 모델 저장 epoch | 3 | **27** |
| 마지막 epoch Train Loss | 0.7247 | 0.3716 |
| Test Accuracy | **미측정** | **미측정** |

- 실행 B는 27·29 epoch에서 같은 최고값(0.9549)을 기록하였다. 제공 코드는 이전 최고값보다 **클 때만** 저장하므로 27 epoch 모델이 저장되었다.

### 3-2. 실행 A 기록

| Epoch | Train Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | 1.0089 | 0.7218 |
| 2 | 0.8075 | 0.7519 |
| 3 | 0.7247 | 0.8421 |

### 3-3. 실행 B 기록

| Epoch | Loss | Val Acc | Epoch | Loss | Val Acc | Epoch | Loss | Val Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9924 | 0.6617 | 11 | 0.4895 | 0.8722 | 21 | 0.3975 | 0.9248 |
| 2 | 0.8108 | 0.7669 | 12 | 0.4764 | 0.9023 | 22 | 0.3986 | 0.9398 |
| 3 | 0.7078 | 0.7669 | 13 | 0.4884 | 0.9248 | 23 | 0.3687 | 0.9398 |
| 4 | 0.6454 | 0.8421 | 14 | 0.4456 | 0.9098 | 24 | 0.3672 | 0.9098 |
| 5 | 0.6012 | 0.8421 | 15 | 0.4202 | 0.9173 | 25 | 0.3873 | 0.9173 |
| 6 | 0.5554 | 0.8797 | 16 | 0.4202 | 0.8872 | 26 | 0.3729 | 0.9398 |
| 7 | 0.5520 | 0.8797 | 17 | 0.4177 | 0.9098 | **27** | **0.3752** | **0.9549** |
| 8 | 0.5107 | 0.8947 | 18 | 0.4265 | 0.9023 | 28 | 0.3538 | 0.9323 |
| 9 | 0.5184 | 0.8872 | 19 | 0.3897 | 0.9248 | 29 | 0.3567 | 0.9549 |
| 10 | 0.4942 | 0.8722 | 20 | 0.4069 | 0.9098 | 30 | 0.3716 | 0.9474 |

| 구간 | Validation Accuracy | 해석 |
|---|---|---|
| 1 ~ 8 epoch | 0.6617 → 0.8947 | 빠른 상승 |
| 9 ~ 21 epoch | 0.8722 ~ 0.9248 | 완만한 상승, 12 epoch에 처음 0.90 도달 |
| 22 ~ 30 epoch | 0.9098 ~ 0.9549 (평균 약 0.937) | **정체.** 변동폭은 검증 이미지 2~5장 수준 |

- Validation 1장 = 약 0.75%p. 22 epoch 이후의 차이는 의미 있는 개선으로 보기 어렵다.
- Validation Accuracy의 뚜렷한 하락(과적합 신호)은 관찰되지 않았다.

### 3-4. 실행 B 모델의 분류 상세 (Validation)

`tools/cnn_predict.py`로 저장된 모델을 다시 평가하여 얻은 혼동행렬이다 (학습 코드 수정 없이 별도 스크립트로 측정).

| 정답 \ 예측 | angular_leaf_spot | bean_rust | healthy |
|---|---:|---:|---:|
| **angular_leaf_spot** | **40** | 3 | 1 |
| **bean_rust** | 0 | **45** | 0 |
| **healthy** | 0 | 2 | **42** |

| 지표 | 값 |
|---|---:|
| Accuracy | 0.9549 (127/133) — `training_report.json`의 best와 일치 |
| 정상 vs 질병 2분류 | 0.977 |
| 질병 종류 구분 (질병 89장) | 0.955 |
| 오답 6건 | 반점병→녹병 3, 반점병→정상 1, 정상→녹병 2 |
| Train 정확도 (참고) | 0.9313 — Backbone 고정으로 Train·Val 차이가 작음 |

- 녹병 45장은 모두 맞혔다. 오답은 반점병과 정상 잎에서 나왔다.

---

## 4. 생성 파일 확인

| 파일 | 실행 A (`_e3`) | 실행 B (서비스 경로 `artifacts/beans_resnet18`) |
|---|---|---|
| config.json | 미확인 (비교용) | [x] 808 B |
| model.safetensors | 미확인 (비교용) | [x] 43 MB |
| preprocessor_config.json | 미확인 (비교용) | [x] 352 B |
| training_report.json | [x] | [x] |

```bash
ls -lh artifacts/beans_resnet18*/
grep -A4 id2label artifacts/beans_resnet18/config.json     # 3개 클래스인지 확인
```

---

## 5. 서비스 연결 확인

```bash
mv artifacts/beans_resnet18 artifacts/beans_resnet18_e3       # 실행 A 보존
cp -r artifacts/beans_resnet18_e30 artifacts/beans_resnet18   # 실행 B를 서비스 경로로
echo "MODEL_PATH=artifacts/beans_resnet18" > .env
```

| 항목 | 결과 |
|---|---|
| MODEL_PATH | `artifacts/beans_resnet18` (실행 B 모델) |
| 응답 `model_id` | `artifacts/beans_resnet18` (동기·비동기 모두) |
| 정답 angular_leaf_spot 이미지 → 예측 / 점수 | angular_leaf_spot / 0.9256 (동기, `evidence/ui_sync.png`), angular_leaf_spot / 0.8494 (비동기, `evidence/ui_async.png`) |
| 정답 bean_rust 이미지 → 예측 / 점수 | 클래스별 기록 미실시 — 성능 측정에서 bean_rust 라벨 응답 확인 (`evidence/db_rows.png` job 9~12) |
| 정답 healthy 이미지 → 예측 / 점수 | 클래스별 기록 미실시 — 성능 측정에서 healthy 라벨 응답 확인 (`evidence/db_rows.png` job 13~18) |

> 위 결과는 **API 동작 확인**이며 Test Accuracy가 아니다. 화면 시연 이미지(`angular_leaf_spot_train.*`)는 학습에 사용한 Train 이미지이므로 정확도 근거로 쓰지 않는다.

---

## 6. 결과 판정

**완료**
- Head-only Fine-tuning 2회 실행(3 epoch, 30 epoch) 및 모델 저장
- 동일 Validation 133장 기준 Best Validation Accuracy: 실행 A 0.8421, 실행 B 0.9549

**한계**
1. **재현성:** 동일 seed(42)에서도 3 epoch 시점 정확도가 실행 A 0.8421, 실행 B 0.7669로 달랐다. 학습이 GPU(CUDA)에서 실행되었고, 제공 코드는 `torch.manual_seed`만 고정할 뿐 cuDNN 연산의 비결정성(`torch.backends.cudnn.deterministic`)은 고정하지 않으므로 이것이 원인으로 추정된다 (미검증). 두 결과의 차이를 **epoch 증가만의 효과로 단정하지 않는다.**
2. **Validation 선택 편향:** Validation으로 최고 epoch을 골랐으므로 0.9549는 낙관적일 수 있다.
3. **Test 미측정:** 제공 코드에 Test 평가 절차가 없다.
4. **학습 범위:** Backbone 고정. Full Fine-tuning은 수행하지 않았다.

**추가 실험** (선택, 별도 보고서)
- 배경을 지운 이미지로 같은 조건 학습 → Val 0.9398. CNN이 배경에 의존하지 않음을 확인 ([보고서 10](10_실험_CNN_잎마스크_융합.md))
- OpenCV 색상 특징·융합 비교 ([보고서 09](09_실험_선명도마스크_색상특징_RandomForest.md), [보고서 10](10_실험_CNN_잎마스크_융합.md))

**증빙**
- `reports/evidence/training_report_headonly_e3.json`, `reports/evidence/training_report_headonly_e30.json`
- `reports/data/cnn_probs_beans_resnet18_e30.json` (혼동행렬)
- `reports/evidence/training_report_e30.png`, `reports/evidence/model_folder.png`
