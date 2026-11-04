"""동기·비동기·캐시 HIT/MISS 응답시간을 같은 조건으로 반복 측정한다 (산출물 6).

[측정 보조 도구] 제공 코드는 수정하지 않는다. 실행 중인 서버에 HTTP 요청만 보낸다.
사전 조건: Redis·PostgreSQL(Docker), Celery Worker, uvicorn이 모두 실행 중이어야 한다.

    python -m tools.measure_timings                 # 조건별 5회
    python -m tools.measure_timings --repeat 10

측정 설계 (평가가이드 산출물 6의 조건)
    - 워밍업: 서버·Worker 모두 첫 요청에 모델을 불러오므로, 측정 전에 워밍업 요청을 보내고 비교에서 제외
    - MISS: 매 회 서로 다른 Test 이미지 사용 → 캐시에 없는 이미지 (실행 전 Redis 캐시를 비움)
    - HIT : 직전에 동기로 분류한 같은 이미지를 다시 보냄
    - 동기 시간  = 요청 → 결과 응답까지 (HTTP 200)
    - 비동기 접수 = 요청 → 202 응답까지
    - 비동기 완료 = 요청 → 상태조회에서 done을 확인한 시점까지 (폴링 간격 포함, --poll 로 조정)
    - Test 이미지는 응답시간 측정에만 쓰며 정확도 평가에는 쓰지 않는다

출력: reports/data/timings.csv (원본), reports/data/timings_summary.json (중앙값)
"""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def sync_call(client, base, data: bytes, name: str) -> dict:
    t0 = time.perf_counter()
    r = client.post(f"{base}/api/sync/classify/", files={"image": (name, data, "image/jpeg")})
    http_ms = ms(t0)
    r.raise_for_status()
    j = r.json()
    return {"http_ms": http_ms, "final_ms": http_ms, "inference_ms": j.get("inference_ms"),
            "cache_hit": j.get("cache_hit"), "label": j["top1"]["label"], "job_id": ""}


def async_call(client, base, data: bytes, name: str, poll: float, timeout: float) -> dict:
    t0 = time.perf_counter()
    r = client.post(f"{base}/api/async/classify/", files={"image": (name, data, "image/jpeg")})
    http_ms = ms(t0)
    if r.status_code != 202:
        raise RuntimeError(f"비동기 접수 실패 HTTP {r.status_code}: {r.text}")
    job_id = r.json()["job_id"]
    while True:
        time.sleep(poll)
        j = client.get(f"{base}/api/async/jobs/{job_id}/").json()
        if j["status"] in ("done", "failed"):
            final_ms = ms(t0)
            break
        if time.perf_counter() - t0 > timeout:
            raise RuntimeError(f"job {job_id}가 {timeout}초 안에 끝나지 않았습니다 (Worker 실행 확인).")
    if j["status"] == "failed":
        raise RuntimeError(f"job {job_id} 실패: {j.get('error_message')}")
    return {"http_ms": http_ms, "final_ms": final_ms, "inference_ms": j.get("inference_ms"),
            "cache_hit": j.get("cache_hit"), "label": j["top1"]["label"], "job_id": job_id}


def flush_cache(redis_db: int) -> bool:
    """Docker의 Redis 캐시 DB만 비운다 (브로커 DB는 건드리지 않음)."""
    try:
        subprocess.run(["docker", "compose", "exec", "-T", "redis", "redis-cli", "-n", str(redis_db), "FLUSHDB"],
                       check=True, capture_output=True, timeout=20)
        return True
    except Exception as exc:
        print(f"경고: 캐시를 비우지 못했습니다 ({exc}). MISS 측정에 이전 캐시가 섞일 수 있습니다.")
        return False


def main():
    parser = argparse.ArgumentParser(description="동기·비동기·캐시 응답시간 측정")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--data", default="dataset/test", help="측정용 이미지 폴더 (클래스 폴더 포함)")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--poll", type=float, default=0.2, help="비동기 상태조회 간격(초)")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--cache-db", type=int, default=1, help="Redis 캐시 DB 번호 (config.py 기본 1)")
    parser.add_argument("--no-flush", action="store_true", help="측정 전 캐시를 비우지 않음")
    parser.add_argument("--output", default="reports/data/timings.csv")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.data).rglob("*") if p.suffix.lower() in EXTS)
    need = 2 + args.repeat * 2                      # 워밍업 2 + 동기 MISS n + 비동기 MISS n
    if len(files) < need:
        raise SystemExit(f"이미지가 부족합니다: {len(files)}장 < 필요 {need}장")
    # 클래스가 섞이도록 간격을 두고 선택
    step = max(1, len(files) // need)
    pool = files[::step][:need]
    warm, sync_imgs, async_imgs = pool[:2], pool[2:2 + args.repeat], pool[2 + args.repeat:need]

    flushed = False if args.no_flush else flush_cache(args.cache_db)
    rows = []
    with httpx.Client(timeout=args.timeout) as client:
        if client.get(f"{args.base}/health").status_code != 200:
            raise SystemExit("서버 /health 실패. uvicorn 실행을 확인하세요.")

        print("워밍업 (비교에서 제외): 서버·Worker 모델 로딩")
        w1 = sync_call(client, args.base, warm[0].read_bytes(), warm[0].name)
        w2 = async_call(client, args.base, warm[1].read_bytes(), warm[1].name, args.poll, args.timeout)
        rows.append({"run_id": 0, "path": "sync", "cache_status": "WARMUP", "image": str(warm[0]), **w1})
        rows.append({"run_id": 0, "path": "async", "cache_status": "WARMUP", "image": str(warm[1]), **w2})
        print(f"  sync {w1['http_ms']} ms (inference {w1['inference_ms']}) / async 완료 {w2['final_ms']} ms (inference {w2['inference_ms']})")

        for i in range(args.repeat):
            n = i + 1
            # 동기 MISS → 같은 이미지로 동기 HIT
            img = sync_imgs[i]
            data, name = img.read_bytes(), img.name
            r = sync_call(client, args.base, data, name)
            rows.append({"run_id": n, "path": "sync", "cache_status": "MISS", "image": str(img), **r})
            r = sync_call(client, args.base, data, name)
            rows.append({"run_id": n, "path": "sync", "cache_status": "HIT", "image": str(img), **r})
            # 비동기 MISS → 같은 이미지로 비동기 HIT
            img = async_imgs[i]
            data, name = img.read_bytes(), img.name
            r = async_call(client, args.base, data, name, args.poll, args.timeout)
            rows.append({"run_id": n, "path": "async", "cache_status": "MISS", "image": str(img), **r})
            r = async_call(client, args.base, data, name, args.poll, args.timeout)
            rows.append({"run_id": n, "path": "async", "cache_status": "HIT", "image": str(img), **r})
            print(f"  {n}/{args.repeat} 완료")

    for r in rows:
        r["image_sha256_12"] = hashlib.sha256(Path(r["image"]).read_bytes()).hexdigest()[:12]
        r["cache_hit_ok"] = (r["cache_status"] == "WARMUP") or \
                            (r["cache_hit"] is (r["cache_status"] == "HIT"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "path", "cache_status", "http_ms", "final_ms", "inference_ms", "cache_hit",
            "cache_hit_ok", "label", "job_id", "image", "image_sha256_12"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    def med(path, status, key):
        vals = [r[key] for r in rows if r["path"] == path and r["cache_status"] == status
                and isinstance(r[key], (int, float))]
        return round(statistics.median(vals), 2) if vals else None

    summary = {}
    print(f"\n중앙값 (ms, 조건별 {args.repeat}회, 워밍업 제외)")
    print(f"{'경로':<6} {'캐시':<5} {'HTTP 응답':>10} {'최종 확인':>10} {'추론':>8}")
    for path in ("sync", "async"):
        for status in ("MISS", "HIT"):
            s = {k: med(path, status, k) for k in ("http_ms", "final_ms", "inference_ms")}
            summary[f"{path}_{status}"] = s
            print(f"{path:<6} {status:<5} {str(s['http_ms']):>10} {str(s['final_ms']):>10} {str(s['inference_ms']):>8}")

    bad = [r for r in rows if not r["cache_hit_ok"]]
    if bad:
        print(f"\n경고: 기대와 다른 cache_hit {len(bad)}건 (csv의 cache_hit_ok=False 확인)")
    sm, am = summary["sync_MISS"]["http_ms"], summary["async_MISS"]["http_ms"]
    if sm and am:
        print(f"\nHTTP 응답시간 감소율(동기 MISS 대비 비동기 접수) = ({sm} - {am}) / {sm} × 100 = {(sm - am) / sm * 100:.2f}%")
        print("※ 추론이 빨라졌다는 뜻이 아니라, 접수와 처리를 분리한 효과이다.")

    meta = {"repeat": args.repeat, "poll_interval_s": args.poll, "cache_flushed_before": flushed,
            "warmup_excluded": True, "image_source": args.data, "summary_median_ms": summary,
            "unexpected_cache_hit_rows": len(bad)}
    out.with_name("timings_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}\n저장: {out.with_name('timings_summary.json')}")


if __name__ == "__main__":
    main()
