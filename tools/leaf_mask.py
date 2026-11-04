"""잎 영역만 꺼내는 마스크.

[선택 개선 실험] 제공 코드는 수정하지 않는다. 이미지 색은 바꾸지 않는다.

두 가지 방법을 제공한다. 기본값은 grabcut.

1) grabcut (기본) — 색 사전 정보 + 중앙 위치 사전 정보 + OpenCV GrabCut
    실제 사진 8장(밭 전경, 손에 든 잎, 흐린 녹색 배경, 황변 잎, 그림자)에서 모두 잎만 선택됨.
    처리 순서
      1. Lab 변환. a*(녹색-↔적색+) 기준 Otsu로 "잎 색" 픽셀 결정 (황변 잎: b* 높고 a* 중간도 포함)
      2. GrabCut 초기 라벨
         - 사진 중앙 30% 영역의 잎 색 픽셀 → 확실한 전경 (촬영자가 대상 잎을 가운데 둔다는 가정)
         - 가장자리 12%를 뺀 안쪽의 잎 색 픽셀 → 아마도 전경
         - 크기가 큰 붉은 덩어리(흙·피부, 이미지의 0.2% 이상) → 확실한 배경
           (작은 붉은 덩어리는 잎 속 병반일 수 있으므로 배경으로 고정하지 않음)
         - 나머지 → 아마도 배경
      3. GrabCut 반복(색 분포 모델을 사진마다 학습)으로 전경/배경 경계 결정
      4. 중앙과 가장 많이 겹치는 덩어리를 대표 잎으로 선택
         + 크고(5% 이상) 대표 잎과 색이 비슷한(Lab 거리 30 미만) 덩어리 추가 (같은 포기의 다른 잎)
      5. 윤곽 내부 채우기 → 잎 속 병반 포함
    속도를 위해 긴 변 320px로 줄여 계산한 뒤 원래 크기로 확대한다.

2) color — Lab a* Otsu + (흐린 녹색 배경일 때) 선명도. 이전 버전 방식, 비교 기록용.
    밭 사진·손에 든 잎에서는 동작하지만, 흐린 큰 잎·황변 잎·잡초가 많은 사진에서 실패함.

사용 예 (시각화)
    python -m tools.leaf_mask --input dataset/val/bean_rust --limit 10
    python -m tools.leaf_mask --input dataset/val/bean_rust --limit 10 --method color
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.focus_recolor import EXTS, read_image, write_image


# ---------------------------------------------------------------- 공통
def leaf_color(lab: np.ndarray, dark_l: int = 35) -> tuple[np.ndarray, int]:
    """잎 색 픽셀(녹색~황록, 황변 포함)과 a* Otsu 기준값."""
    a = cv2.GaussianBlur(lab[..., 1], (5, 5), 0).astype(np.int16)
    b = lab[..., 2].astype(np.int16)
    t, _ = cv2.threshold(a.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col = ((a < t) | ((b > 150) & (a < 140))) & (lab[..., 0] >= dark_l)
    return col, int(t)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


# ---------------------------------------------------------------- 1) grabcut
def leaf_mask_grabcut(bgr: np.ndarray, margin: float = 0.12, core: float = 0.30,
                      iters: int = 5, red_min_area: float = 0.002, extra_area: float = 0.05,
                      max_color_dist: float = 30.0, work_size: int = 320) -> tuple[np.ndarray, dict]:
    H, W = bgr.shape[:2]
    scale = min(1.0, work_size / max(H, W))
    img = cv2.resize(bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    col, t = leaf_color(lab)
    a = lab[..., 1].astype(np.int16)

    inner = np.zeros((h, w), bool)
    inner[int(h * margin):int(h * (1 - margin)), int(w * margin):int(w * (1 - margin))] = True
    c0, c1 = (1 - core) / 2, (1 + core) / 2
    center = np.zeros((h, w), bool)
    center[int(h * c0):int(h * c1), int(w * c0):int(w * c1)] = True

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[inner & col] = cv2.GC_PR_FGD
    gc[center & col] = cv2.GC_FGD
    red = (~col & (a > t + 8)).astype(np.uint8)
    rn, rl, rs, _ = cv2.connectedComponentsWithStats(red)
    for i in range(1, rn):
        if rs[i, cv2.CC_STAT_AREA] >= red_min_area * h * w:
            gc[rl == i] = cv2.GC_BGD

    info = {"method": "grabcut", "otsu_a": t, "fallback_full": False}
    if not (gc == cv2.GC_FGD).any() and not (gc == cv2.GC_PR_FGD).any():
        info.update(area_ratio=0.0, leaves=0, note="잎 색 픽셀 없음")
        return np.zeros((H, W), np.uint8), info

    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(img, gc, None, bgm, fgm, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:   # 전경/배경 한쪽이 비면 실패할 수 있다
        info["note"] = f"grabCut 실패: {exc.err}"
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        info.update(area_ratio=0.0, leaves=0)
        return np.zeros((H, W), np.uint8), info

    best = max(range(1, n), key=lambda i: ((lbl == i) & center).sum() * 10 + st[i, cv2.CC_STAT_AREA] * 0.01)
    labf = lab.astype(np.float32)
    ref = labf[lbl == best].mean(0)
    keep = [best]
    for i in range(1, n):
        if i == best or st[i, cv2.CC_STAT_AREA] < extra_area * h * w or not ((lbl == i) & inner).any():
            continue
        if np.linalg.norm(labf[lbl == i].mean(0) - ref) < max_color_dist:
            keep.append(i)

    mask = fill_holes((np.isin(lbl, keep) * 255).astype(np.uint8))
    if scale < 1:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    info.update(area_ratio=round(float(mask.mean() / 255), 4), leaves=len(keep))
    return mask, info


# ---------------------------------------------------------------- 2) color (이전 방식)
def raw_sharpness(bgr: np.ndarray, window: int = 15) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mean = cv2.blur(lap, (window, window))
    return np.maximum(cv2.blur(lap * lap, (window, window)) - mean * mean, 0)


def leaf_mask_color(bgr: np.ndarray, min_area: float = 0.01, min_sep: float = 10.0,
                    dark_l: int = 35, blur_ratio: float = 0.25) -> tuple[np.ndarray, dict]:
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[..., 0]
    a_ch = cv2.GaussianBlur(lab[..., 1], (5, 5), 0)
    otsu_t, _ = cv2.threshold(a_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw = ((a_ch < otsu_t) & (l_ch >= dark_l)).astype(np.uint8) * 255

    inside, outside = a_ch[raw > 0], a_ch[raw == 0]
    sep = float(outside.mean() - inside.mean()) if inside.size and outside.size else 0.0
    info = {"method": "color", "otsu_a": int(otsu_t), "a_separation": round(sep, 2), "fallback_full": False}
    if sep < min_sep:
        mask = np.full((h, w), 255, np.uint8)
        mask[l_ch < dark_l] = 0
        info.update(fallback_full=True, area_ratio=round(float(mask.mean() / 255), 4), leaves=1)
        return mask, info

    var = raw_sharpness(bgr)
    gvals = var[raw > 0]
    ratio, use_sharp, sharp_t = 1.0, False, 0.0
    if gvals.size > 100:
        u = cv2.normalize(np.log1p(gvals), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        st, _ = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = u.ravel() >= st
        if hi.any() and (~hi).any():
            ratio = float(np.median(gvals[~hi]) / max(np.median(gvals[hi]), 1e-6))
            sharp_t = float(np.min(gvals[hi]))
            use_sharp = ratio < blur_ratio
    info.update(blur_ratio=round(ratio, 3), mode="green+sharp" if use_sharp else "green")

    k = max(5, (min(h, w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if use_sharp:
        raw = ((raw > 0) & (var >= sharp_t)).astype(np.uint8) * 255
        big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 3 | 1, k * 3 | 1))
        clean = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, big, iterations=2)
    else:
        clean = cv2.morphologyEx(cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in contours if cv2.contourArea(c) >= min_area * h * w]
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, keep, -1, 255, thickness=cv2.FILLED)
    info.update(area_ratio=round(float(mask.mean() / 255), 4), leaves=len(keep))
    return mask, info


# ---------------------------------------------------------------- 진입점
def leaf_mask(bgr: np.ndarray, method: str = "grabcut") -> tuple[np.ndarray, dict]:
    """잎 마스크(0/255)와 진단 정보. method = 'grabcut'(기본) 또는 'color'."""
    return leaf_mask_color(bgr) if method == "color" else leaf_mask_grabcut(bgr)


def visualize(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """원본 | 마스크 | 잎만 남긴 이미지(배경 어둡게) | 윤곽 표시"""
    leaf_only = bgr.copy()
    leaf_only[mask == 0] = (leaf_only[mask == 0] * 0.15).astype(np.uint8)
    outline = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, contours, -1, (255, 255, 0), max(2, bgr.shape[1] // 250))
    return np.hstack([bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), leaf_only, outline])


def main():
    parser = argparse.ArgumentParser(description="잎 마스크 생성·시각화")
    parser.add_argument("--input", required=True, help="이미지 파일 또는 폴더")
    parser.add_argument("--output-dir", default="reports/figures/leaf_mask")
    parser.add_argument("--method", choices=["grabcut", "color"], default="grabcut")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"처리할 이미지가 없습니다: {src}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        bgr = read_image(path)
        mask, info = leaf_mask(bgr, args.method)
        write_image(out / f"{path.stem}_leaf_compare.png", visualize(bgr, mask))
        write_image(out / f"{path.stem}_leaf_mask.png", mask)
        rows.append({"file": str(path), **info})
        print(rows[-1])
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    empty = sum(r.get("leaves", 0) == 0 for r in rows)
    print(f"완료: {len(rows)}장 (잎 미검출 {empty}장) → {out}")


if __name__ == "__main__":
    main()
