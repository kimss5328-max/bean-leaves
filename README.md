# FastAPI Beans ResNet18 — 동기·비동기 콩잎 분류 서비스

콩잎 사진 1장을 `angular_leaf_spot`(각진잎반점병) / `bean_rust`(녹병) / `healthy`(정상)로 분류한다.
`microsoft/resnet-18`을 Beans 데이터로 Fine-tuning하고, FastAPI 동기 API와 Celery 비동기 작업 큐로 서빙한다.

| 항목 | 결과 |
|---|---|
| 서비스 모델 | Head-only Fine-tuning 30 epoch — Validation Accuracy **0.9549** (127/133) |
| Test Accuracy (1회 평가) | **0.8984** (115/128), Macro F1 0.899 |
| 선택 실험 최고값 | Full Fine-tuning 10 epoch — Validation 0.9925 (Best), 0.967 (안정 구간 평균) |

---

## 1. 환경

| 구분 | 내용 |
|---|---|
| OS | Windows + WSL2 Ubuntu |
| Python | 3.11.17 (pyenv) + `.venv` |
| Docker | Docker Desktop (WSL integration 활성화) — Redis 6379, PostgreSQL 5432 |
| Python 프로세스 | FastAPI(uvicorn), Celery Worker — WSL `.venv`에서 실행 |
| 학습 장치 | GPU (NVIDIA GeForce RTX 3060 12GB) — CUDA 사용 가능 시 자동 |
| 추론 장치 | CPU (`ai/hf_model.py`의 `device=-1` 고정) |

## 2. 설치

```bash
cd ~/fastapi_resnet18
pyenv local 3.11.17
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. 데이터 준비

Hugging Face [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/main/data)의 `train.zip`, `validation.zip`, `test.zip`을 받아 아래처럼 배치한다. **원본 `validation/`은 `dataset/val/`에 배치**한다.

```text
dataset/
├── train/{angular_leaf_spot,bean_rust,healthy}/   # 345 / 348 / 341
├── val/{angular_leaf_spot,bean_rust,healthy}/     # 44 / 45 / 44
└── test/{angular_leaf_spot,bean_rust,healthy}/    # 학습·모델 선택에 사용하지 않음
```

데이터셋은 용량과 이용 조건(MIT, 데이터셋 페이지 기준) 때문에 저장소에 포함하지 않는다.

## 4. 모델 준비 (Fine-tuning)

제공 `training/finetune.py`를 **수정 없이** 실행한다.

```bash
python -m training.finetune --data dataset --output artifacts/beans_resnet18 \
  --epochs 30 --batch-size 8 --freeze-backbone
cat artifacts/beans_resnet18/training_report.json      # best_val_accuracy 확인
```

`.env`:

```dotenv
MODEL_PATH=artifacts/beans_resnet18
```

모델 파일(`artifacts/`)은 용량 때문에 저장소에 포함하지 않으며, 허용된 경로로 별도 전달한다.

## 5. 실행 순서

Docker Desktop이 **Engine running** 상태인지 먼저 확인한다.

```bash
# 터미널 1 — Redis·PostgreSQL
source .venv/bin/activate
docker compose up -d redis postgres
docker compose exec redis redis-cli ping                               # PONG
docker compose exec postgres pg_isready -U vision_user -d vision_db    # accepting connections (시작 직후엔 수 초 대기)

# 터미널 2 — Celery Worker
source .venv/bin/activate
celery -A ai.celery_app:celery_app worker --loglevel=info --pool=solo --concurrency=1

# 터미널 3 — FastAPI
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

| 주소 | 내용 |
|---|---|
| http://127.0.0.1:8000/vision/sync/ | 동기 분류 화면 |
| http://127.0.0.1:8000/vision/async/ | 비동기 분류 화면 |
| http://127.0.0.1:8000/docs | Swagger API 문서 |

**모델을 바꾸면** `.env`를 수정하고 **FastAPI와 Worker를 모두 재시작**한다 (각 프로세스가 모델을 메모리에 유지).

## 6. 확인할 응답

```bash
IMG=$(find dataset/test/bean_rust -type f | head -1)
curl http://127.0.0.1:8000/health                                                  # {"status":"ok"}
curl -s -X POST -F "image=@$IMG" http://127.0.0.1:8000/api/sync/classify/          # top1, all, inference_ms, cache_hit
curl -i -X POST -F "image=@$IMG" http://127.0.0.1:8000/api/async/classify/         # 202, job_id
curl -s http://127.0.0.1:8000/api/async/jobs/1/                                    # status: done
docker compose exec postgres psql -U vision_user -d vision_db -c \
  'SELECT id, status, top1_label, cache_hit, inference_ms FROM image_jobs ORDER BY id DESC LIMIT 10;'
```

- 같은 이미지를 다시 보내면 `cache_hit: true`, `inference_ms: null` (Redis 캐시, 600초)
- 서버·Worker의 첫 요청은 모델 로딩 때문에 약 3.4초 걸린다

## 7. 테스트

```bash
python -m pytest -q tests/test_api.py      # 모의 통합 테스트 (가짜 모델·SQLite, Docker 불필요)
python -m tools.measure_timings --repeat 5 # 실환경 성능 측정 (서버·Worker·Docker 실행 중)
```

## 8. 결과물 위치

| 경로 | 내용 |
|---|---|
| `reports/01_FineTuning_결과보고서.md` ~ `08_최종기술보고서.md` | 산출물 1~8 |
| `reports/09_실험_선명도마스크_색상특징_RandomForest.md` | 선택 실험: OpenCV 마스크 + 색상 특징 |
| `reports/10_실험_CNN_잎마스크_융합.md` | 선택 실험: 잎 마스크 CNN, CNN+색상 융합, Full Fine-tuning |
| `reports/data/` | 측정 원본 (`timings.csv`, `cnn_probs_*.csv`, `color_features_*.csv`, 각종 리포트 JSON) |
| `reports/evidence/` | 실행 캡처, `training_report` 백업 |
| `reports/figures/` | 마스크 시각화, 오답 이미지 (`errors/test_test/` = Test 오답 13장) |
| `artifacts/beans_resnet18/training_report.json` | 서비스 모델 학습 기록 |
| `tools/` | 선택 실험·측정 스크립트 (제공 코드 아님) |
| `training/finetune_fusion.py` | 원본 `finetune.py` + 융합 평가 (선택 실험용, 원본 코드 구역은 동일) |

> **Git 포함 범위:** 저장소 전체에서 `*.csv`, `*.json`, 이미지는 제외하되, `reports/` 아래의 증빙·측정 원본·도표는 예외로 포함하였다 (`.gitignore`의 `!reports/**` 규칙).

## 9. 제공 코드 사용 원칙

- `training/finetune.py`, `ai/`, `main.py`, `templates/`, `static/`, `compose.yaml`은 **01·02 제공 코드를 수정 없이** 사용하였다.
- 학습 설정 변경(epoch, 학습률, Backbone 고정 여부)은 **실행 옵션**으로만 하였다.
- 선택 실험 코드는 모두 `tools/`와 `training/finetune_fusion.py`에 **별도 파일**로 두었다.

## 10. 제출 및 보안

- `.env`, 비밀번호, API 키는 저장소에 올리지 않는다 (`.gitignore` 적용).
- `compose.yaml`의 DB 비밀번호는 실습용 값이다. 실제 운영에서는 환경변수로 분리한다.
- `dataset/`, `dataset_leaf/`, `artifacts/`, `media/jobs/`는 저장소에서 제외하고 별도 전달한다.
