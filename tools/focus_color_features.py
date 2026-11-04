"""[1] 선명도(초점) 마스크 안에서만 색상 특징을 추출해 CSV로 저장한다.

[선택 개선 실험] 제공 코드(finetune.py 등)는 수정하지 않는다.
이미지 색은 바꾸지 않고 "읽기만" 한다. Test 폴더는 기본적으로 사용하지 않는다.

처리 순서
    이미지 → 선명도 지도 → 초점 마스크(focus_recolor.py 재사용)
          → 마스크 안 픽셀만 HSV로 변환 → 색 통계 → reports/data/color_features.csv

사용 예
    python -m tools.focus_color_features
    python -m tools.focus_color_features --splits train val --limit 20   # 빠른 확인
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from tools.focus_recolor import EXTS, focus_mask, read_image, sharpness_map
from tools.leaf_mask import leaf_mask

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
HUE_BINS = 12            # OpenCV Hue(0~179)를 15단위 12구간으로 나눔
MIN_SAT, MIN_VAL = 40, 40  # 회색·검정에 가까운 픽셀은 Hue가 무의미하므로 색 비율 계산에서 제외

# OpenCV Hue 기준 색 구간 (실제 콩잎 사진으로 범위 점검 필요)
HUE_RANGES = {
    "brown_red": [(0, 20), (170, 180)],  # 갈색·적갈색 (녹병 반점 후보)
    "yellow": [(20, 35)],                # 황색 (반점 주변 황변 후보)
    "green": [(35, 90)],                 # 녹색 (정상 잎 조직)
}


def feature_names() -> list[str]:
    names = []
    for ch in ("h", "s", "v"):
        names += [f"{ch}_mean", f"{ch}_std"]
    names += [f"hue_bin_{i:02d}" for i in range(HUE_BINS)]
    names += [f"ratio_{k}" for k in HUE_RANGES]
    names += ["ratio_low_saturation", "mask_area_ratio", "focus_sharpness_mean"]
    return names


def make_mask(bgr: np.ndarray, sharp: np.ndarray, mode: str, window: int,
              min_area: float) -> tuple[np.ndarray, bool]:
    """mode별 마스크와 '의도한 대상을 찾았는지' 여부를 반환한다."""
    full = np.full(sharp.shape, 255, np.uint8)
    if mode == "none":
        return full, True
    if mode == "leaf":
        mask, info = leaf_mask(bgr)
        ok = bool(mask.any()) and not info["fallback_full"]
        return (mask if mask.any() else full), ok
    mask, _ = focus_mask(sharp, min_area, window)          # mode == "focus"
    return (mask if mask.any() else full), bool(mask.any())


def extract(bgr: np.ndarray, window: int, min_area: float, mode: str = "focus") -> tuple[list[float], bool]:
    sharp = sharpness_map(bgr, window)
    mask, focus_found = make_mask(bgr, sharp, mode, window, min_area)  # 못 찾으면 전체 이미지로 대체
    sel = mask > 0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][sel].astype(np.float32)
    s = hsv[..., 1][sel].astype(np.float32)
    v = hsv[..., 2][sel].astype(np.float32)

    feats = []
    for ch in (h, s, v):
        feats += [float(ch.mean()), float(ch.std())]

    colored = (s >= MIN_SAT) & (v >= MIN_VAL)
    hue_c = h[colored]
    n_col = max(len(hue_c), 1)
    hist, _ = np.histogram(hue_c, bins=HUE_BINS, range=(0, 180))
    feats += (hist / n_col).tolist()

    for ranges in HUE_RANGES.values():
        hit = np.zeros(len(hue_c), bool)
        for lo, hi in ranges:
            hit |= (hue_c >= lo) & (hue_c < hi)
        feats.append(float(hit.sum() / n_col))

    feats.append(float(1 - colored.mean()))      # 채도 낮은 픽셀 비율
    feats.append(float(sel.mean()))              # 마스크가 차지하는 면적
    feats.append(float(sharp[sel].mean()))       # 마스크 내부 평균 선명도
    return feats, focus_found


def main():
    parser = argparse.ArgumentParser(description="초점 영역 색상 특징 추출")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="추출할 분할. Test는 최종 1회 평가 전까지 넣지 않는다")
    parser.add_argument("--mask", choices=["focus", "leaf", "none"], default="leaf",
                        help="focus=선명도 마스크(기존), leaf=잎 마스크, none=전체 이미지")
    parser.add_argument("--output", default=None,
                        help="기본값: reports/data/color_features_<mask>.csv")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--min-area", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=0, help="클래스당 최대 장수(0=전체)")
    args = parser.parse_args()

    out = Path(args.output or f"reports/data/color_features_{args.mask}.csv")
    print(f"마스크 방식: {args.mask}")
    out.parent.mkdir(parents=True, exist_ok=True)
    names = feature_names()
    stats = {}

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "file", "focus_found"] + names)
        for split in args.splits:
            for label in CLASSES:
                folder = Path(args.data) / split / label
                if not folder.is_dir():
                    raise SystemExit(f"폴더가 없습니다: {folder}")
                files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
                if args.limit:
                    files = files[: args.limit]
                found = 0
                for path in files:
                    feats, ok = extract(read_image(path), args.window, args.min_area, args.mask)
                    found += ok
                    writer.writerow([split, label, str(path), int(ok)] + [round(x, 6) for x in feats])
                stats[f"{split}/{label}"] = (len(files), found)
                print(f"{split}/{label}: {len(files)}장, 대상 검출 {found}장")

    total = sum(n for n, _ in stats.values())
    found = sum(k for _, k in stats.values())
    print(f"완료: {total}장 (대상 검출 {found}장, 미검출은 전체 이미지로 대체) → {out}")


if __name__ == "__main__":
    main()
