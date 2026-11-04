"""선명도(초점) 기반으로 가장 초점이 잘 맞은 물체만 색을 바꾼다.

[선택 제안 실험용 스크립트] 제공 코드(finetune.py, services.py)와 독립적으로 동작한다.
학습 데이터 전처리로 쓰지 않는다. 콩잎 질병은 색이 핵심 단서이기 때문이다.

처리 순서
    1. 회색조 변환
    2. 라플라시안(Laplacian)으로 경계 강도 계산
    3. 주변 window 영역의 국소 분산 = 선명도 지도
    4. Otsu 이진화로 "선명한 영역" 마스크 생성
    5. Morphology로 노이즈 정리 → 가장 큰 윤곽(Contour)만 남기고 내부 채움
    6. 마스크 영역만 HSV 색상(Hue)을 이동

사용 예
    python -m tools.focus_recolor --input dataset/test/healthy
    python -m tools.focus_recolor --input sample.jpg --hue-shift 90 --window 21
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_image(path: Path) -> np.ndarray:
    # cv2.imread는 한글 경로를 읽지 못하므로 바이트로 읽은 뒤 디코딩한다.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    buf.tofile(str(path))


def sharpness_map(bgr: np.ndarray, window: int) -> np.ndarray:
    """픽셀별 국소 라플라시안 분산(0~1 정규화). 값이 클수록 초점이 맞은 영역."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 센서 노이즈가 선명도로 잡히는 것을 줄인다
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mean = cv2.blur(lap, (window, window))
    mean_sq = cv2.blur(lap * lap, (window, window))
    var = np.maximum(mean_sq - mean * mean, 0)
    var = np.log1p(var)  # 값 분포가 한쪽으로 치우치므로 로그로 펴 준다
    return cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)


def focus_mask(sharp: np.ndarray, min_area_ratio: float, window: int) -> tuple[np.ndarray, dict]:
    """선명도 지도 → 가장 큰 선명 영역 하나의 마스크(0/255)."""
    sharp_u8 = (sharp * 255).astype(np.uint8)
    otsu_t, binary = cv2.threshold(sharp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    k = max(3, (min(h, w) // 50) | 1)  # 이미지 크기에 비례하는 홀수 커널
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)              # 작은 점 제거
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # 틈 메우기

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    info = {"otsu_threshold": int(otsu_t), "contours": len(contours), "area_ratio": 0.0}
    if not contours:
        return mask, info

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (h * w)
    info["area_ratio"] = round(float(area_ratio), 4)
    if area_ratio < min_area_ratio:
        return mask, info  # 너무 작으면 "초점 물체 없음"으로 처리

    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)  # 내부 구멍까지 채움
    # window로 평균을 내면 선명 영역이 경계 밖으로 약 window/2만큼 번진다 → 그만큼 깎아 보정
    shrink = window // 2
    if shrink > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink | 1, shrink | 1))
        mask = cv2.erode(mask, se)
    return mask, info


def recolor(bgr: np.ndarray, mask: np.ndarray, hue_shift: int, feather: int) -> np.ndarray:
    """마스크 영역만 Hue를 이동한다. 경계는 feather만큼 부드럽게 섞는다."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_shift) % 180).astype(np.uint8)
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    alpha = mask.astype(np.float32) / 255
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (feather | 1, feather | 1), 0)
    alpha = alpha[..., None]
    out = bgr.astype(np.float32) * (1 - alpha) + shifted.astype(np.float32) * alpha
    return out.astype(np.uint8)


def process(path: Path, out_dir: Path, args) -> dict:
    bgr = read_image(path)
    sharp = sharpness_map(bgr, args.window)
    mask, info = focus_mask(sharp, args.min_area, args.window)
    result = recolor(bgr, mask, args.hue_shift, args.feather) if mask.any() else bgr.copy()

    stem = path.stem
    write_image(out_dir / f"{stem}_recolor.png", result)
    write_image(out_dir / f"{stem}_mask.png", mask)
    heat = cv2.applyColorMap((sharp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    write_image(out_dir / f"{stem}_sharpness.png", heat)

    # 원본 | 선명도 | 마스크 | 결과를 한 장으로 비교 (보고서 첨부용)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    write_image(out_dir / f"{stem}_compare.png", np.hstack([bgr, heat, mask_bgr, result]))

    return {"file": str(path), "focus_found": bool(mask.any()), **info}


def main():
    parser = argparse.ArgumentParser(description="선명도 기반 초점 물체 색상 변경")
    parser.add_argument("--input", required=True, help="이미지 파일 또는 폴더")
    parser.add_argument("--output-dir", default="reports/figures/focus_recolor")
    parser.add_argument("--window", type=int, default=15, help="선명도 계산 영역 크기(px). 클수록 덩어리가 큼")
    parser.add_argument("--hue-shift", type=int, default=60, help="OpenCV Hue 이동량(0~179, 60=120도)")
    parser.add_argument("--feather", type=int, default=15, help="경계 부드럽게(px), 0이면 끔")
    parser.add_argument("--min-area", type=float, default=0.02, help="초점 물체 최소 면적 비율")
    parser.add_argument("--limit", type=int, default=0, help="폴더 입력 시 처리할 최대 장수(0=전체)")
    args = parser.parse_args()

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"처리할 이미지가 없습니다: {src}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [process(p, out_dir, args) for p in files]
    for row in rows:
        print(row)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    found = sum(r["focus_found"] for r in rows)
    print(f"완료: {len(rows)}장 중 초점 물체 검출 {found}장 → {out_dir}")


if __name__ == "__main__":
    main()