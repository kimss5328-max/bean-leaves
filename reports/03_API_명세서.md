# API 명세서

> 근거: `main.py`, `ai/router_sync.py`, `ai/router_async.py`, `ai/router_web.py` (02 제공 코드) 및 실제 응답

## 1. API 목록

| API ID | Method | URL | 요청 형태 | 성공 HTTP | 주요 응답 | 실패 HTTP |
|---|---|---|---|---:|---|---|
| API-01 | GET | `/health` | 없음 | 200 | `status` | 서버 미실행 시 응답 없음 |
| API-02 | POST | `/api/sync/classify/` | `multipart/form-data`, 필드 `image` | 200 | `model_id`, `top1`, `all`, `inference_ms`, `cache_hit` | 400, 413 |
| API-03 | POST | `/api/async/classify/` | `multipart/form-data`, 필드 `image` | **202** | `job_id`, `status`, `check_url` | 400, 413, 503 |
| API-04 | GET | `/api/async/jobs/{job_id}/` | 경로 정수 `job_id` | 200 | `status`, `top1`, `all`, `inference_ms`, `cache_hit`, `error_message` | 404 |
| WEB-01 | GET | `/` | 없음 | 307 | `/vision/sync/`로 이동 | — |
| WEB-02 | GET | `/vision/sync/` | 없음 | 200 | HTML | — |
| WEB-03 | GET | `/vision/async/` | 없음 | 200 | HTML | — |
| DOC | GET | `/docs` | 없음 | 200 | Swagger UI | — |

## 2. 공통 입력 제약 (`ai/config.py`, `ai/services.py`)

| 항목 | 제한 | 위반 시 |
|---|---|---|
| 파일 크기 | 1바이트 이상, 5MB(5,242,880바이트) 이하 | 413 (초과) / 400 (빈 파일) |
| 형식 | JPEG, PNG, WEBP (실제 디코딩 기준, 확장자 무관) | 400 |
| 해상도 | 1,200만 픽셀 이하 | 400 |
| 전처리 | EXIF 회전 보정 → RGB 변환 | — |

## 3. API-02 동기 분류

| 항목 | 내용 |
|---|---|
| 처리 | `classify_sync()` → `classify_with_cache()` → (MISS) `classify_image_bytes()` → ResNet18 |
| 캐시 키 | `vision:classify:<모델경로 해시 16자>:<이미지 SHA-256>`, TTL 600초 |
| `inference_ms` | MISS: 모델 추론 시간 / HIT: `null` (추론 생략, 0ms가 아님) |

**실제 성공 응답 — 캐시 MISS**

```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "bean_rust", "score": 0.6425225734710693},
  "all": [
    {"label": "bean_rust", "score": 0.6425225734710693},
    {"label": "angular_leaf_spot", "score": 0.34892964363098145},
    {"label": "healthy", "score": 0.008547825738787651}
  ],
  "inference_ms": 25.29,
  "cache_hit": false
}
```

**실제 성공 응답 — 캐시 HIT**

```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "angular_leaf_spot", "score": 0.9398528933525085},
  "all": [
    {"label": "angular_leaf_spot", "score": 0.9398528933525085},
    {"label": "bean_rust", "score": 0.03711671009659767},
    {"label": "healthy", "score": 0.0230304803699255}
  ],
  "inference_ms": null,
  "cache_hit": true
}
```

**오류 응답**

| 상황 | HTTP | 응답 |
|---|---:|---|
| 이미지가 아닌 파일 | 400 | `{"detail":"올바른 이미지 파일이 아닙니다."}` (실제 응답) |
| 5MB 초과 | 413 | `{"detail":"5MB 이하 파일만 허용합니다."}` (실제 응답) |

## 4. API-03 비동기 접수

| 항목 | 내용 |
|---|---|
| 처리 | 이미지 검증 → `media/jobs/<uuid>.bin` 저장 (사용자 파일명 미사용) → `image_jobs` 행(`queued`) 생성 → `classify_job.delay(job_id)` |
| 의미 | **202는 접수 완료이지 분석 완료가 아니다** |
| 503 | Redis 브로커 연결 실패 시. DB 행은 `failed`로 기록 |

**응답 형식** (코드 기준. 웹 화면에서 접수 42~113 ms, `job_id` 수신 확인 — `evidence/ui_async.png`)

```json
{"job_id": 7, "status": "queued", "check_url": "/api/async/jobs/7/"}
```

## 5. API-04 상태 조회

| `status` | 의미 | `top1`, `all` |
|---|---|---|
| `queued` | 접수됨, Worker 대기 | `null` |
| `running` | Worker 처리 중 | `null` |
| `done` | 완료 | 결과 포함 |
| `failed` | 실패 | `null`, `error_message` 포함 |

**실제 완료 응답 — Worker가 직접 추론 (job 7)**

```json
{
  "job_id": 7,
  "status": "done",
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "angular_leaf_spot", "score": 0.8439902067184448},
  "all": [
    {"label": "angular_leaf_spot", "score": 0.8439902067184448},
    {"label": "bean_rust", "score": 0.10489878803491592},
    {"label": "healthy", "score": 0.05111097916960716}
  ],
  "inference_ms": 3505.12,
  "cache_hit": false,
  "error_message": null
}
```

> `inference_ms` 3505.12는 Worker 프로세스의 **첫 추론이라 모델 로딩 시간이 포함**된 값이다 (`services.py`가 시간 측정 시작 후 모델을 처음 불러옴).

**실제 완료 응답 — 캐시 재사용 (job 6)**: 위와 같은 결과, `"inference_ms": null`, `"cache_hit": true`

**오류 응답:** 없는 작업 → 404 `{"detail":"해당 작업이 없습니다."}` (실제 응답)

## 6. 응답 필드 정의

| 필드 | 타입 | 설명 |
|---|---|---|
| `model_id` | string | `.env`의 `MODEL_PATH` 문자열 (실제 로드된 가중치를 보장하지는 않음) |
| `top1.label` | string | `angular_leaf_spot` / `bean_rust` / `healthy` |
| `top1.score` | float | 소프트맥스 확률 (0~1) |
| `all` | list | 클래스별 확률, 내림차순 (DB 컬럼명은 `all_scores`) |
| `inference_ms` | float \| null | 모델 추론 시간. 캐시 HIT이면 `null` |
| `cache_hit` | bool | Redis 캐시에서 가져왔는지 |
| `job_id` | int | `image_jobs.id` |

## 7. 실행 근거표

| 검증 ID | 입력 | 기대 | 실제 | 증빙 |
|---|---|---:|---|---|
| API-01-T01 | GET `/health` | 200 | 200 (`measure_timings.py` 사전 점검 통과) | `evidence/measure_timings.png` |
| API-02-T01 | Test 이미지 (MISS) | 200 | 200, `cache_hit: false`, 25.29ms | `evidence/ui_sync.png` |
| API-02-T02 | 같은 이미지 재전송 | 200 + HIT | 200, `cache_hit: true`, `inference_ms: null` | `evidence/cache_hit.png` |
| API-02-T03 | `.txt` 파일 | 400 | **400** | `evidence/api_errors.png` |
| API-02-T04 | 5MB+1바이트 파일 | 413 | **413** | `evidence/api_errors.png` |
| API-03-T01 | 이미지 (웹 화면) | 202 | 접수 107ms, job 7 | `evidence/ui_async.png` |
| API-04-T01 | job 7 조회 | 200 / done | done, `cache_hit: false` | `evidence/ui_async.png` |
| API-04-T02 | 없는 job (999999) | 404 | **404** | `evidence/api_errors.png` |

```bash
# 오류 응답 확인 명령
echo bad > /tmp/bad.txt
curl -s -o /dev/null -w "%{http_code}\n" -X POST -F "image=@/tmp/bad.txt" http://127.0.0.1:8000/api/sync/classify/
head -c 5242881 /dev/urandom > /tmp/big.jpg
curl -s -o /dev/null -w "%{http_code}\n" -X POST -F "image=@/tmp/big.jpg" http://127.0.0.1:8000/api/sync/classify/
curl -s -w "\n%{http_code}\n" http://127.0.0.1:8000/api/async/jobs/999999/
```
