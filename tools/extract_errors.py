"""CNN이 틀린 이미지를 뽑아 정답·예측·확률을 적은 이미지로 저장한다.

[선택 실험 보조 도구] tools.cnn_predict 결과 CSV를 읽기만 한다.

    python -m tools.extract_errors --cnn reports/data/cnn_probs_beans_resnet18_full.csv
    python -m tools.extract_errors --cnn reports/data/cnn_probs_beans_resnet18_e30.csv

출력: reports/figures/errors/<모델이름>_<split>/
    - 오답 이미지마다: 원본 복사본 + 아래에 정답·예측·클래스별 확률을 적은 PNG
    - errors.csv : 오답 목록
    - 오답 이미지마다: 잎 마스크 비교 (원본 | 마스크 | 잎만 | 윤곽) — tools.leaf_mask GrabCut
    - errors_grid.png : 오답 전체를 한 장으로 모은 이미지 (확률 표시 + 잎만 + 윤곽)
    마스크를 빼려면 --no-mask
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.focus_recolor import read_image, write_image
from tools.leaf_mask import leaf_mask, visualize


def mask_panels(path: Path, out_stem: Path, height: int) -> tuple[list[Image.Image], dict]:
    """잎 마스크 비교 이미지를 저장하고, 격자용 패널(잎만, 윤곽)을 반환한다."""
    bgr = read_image(path)
    mask, info = leaf_mask(bgr)
    vis = visualize(bgr, mask)                      # 원본 | 마스크 | 잎만 | 윤곽 (가로 4칸)
    write_image(out_stem.with_name(out_stem.name + "_mask_compare.png"), vis)
    w = bgr.shape[1]
    panels = []
    for k in (2, 3):                                # 잎만, 윤곽
        rgb = cv2.cvtColor(vis[:, k * w:(k + 1) * w], cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        panels.append(pil.resize((int(pil.width * height / pil.height), height)))
    return panels, info


def annotate(path: Path, label: str, pred: str, probs: dict, width: int = 400) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)))
    lines = [f"TRUE: {label}", f"PRED: {pred}"] + [f"  {k}: {v:.3f}" for k, v in probs.items()]
    pad, lh = 8, 18
    canvas = Image.new("RGB", (width, im.height + pad * 2 + lh * len(lines)), "white")
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = im.height + pad
    for i, text in enumerate(lines):
        color = (0, 120, 0) if i == 0 else (200, 0, 0) if i == 1 else (40, 40, 40)
        draw.text((pad, y), text, fill=color, font=font)
        y += lh
    draw.rectangle([0, 0, width - 1, im.height - 1], outline=(200, 0, 0), width=4)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="CNN 오답 이미지 추출")
    parser.add_argument("--cnn", required=True, help="tools.cnn_predict 결과 CSV")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-mask", action="store_true", help="잎 마스크 이미지를 만들지 않음")
    args = parser.parse_args()

    csv_path = Path(args.cnn)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prob_cols = [c for c in rows[0] if c.startswith("prob_")]
    errors = [r for r in rows if r["split"] == args.split and r["pred"] != r["label"]]

    model_name = csv_path.stem.replace("cnn_probs_", "")
    out = Path(args.output_dir or f"reports/figures/errors/{model_name}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{model_name} / {args.split}: 오답 {len(errors)}개 → {out}")

    tiles = []
    with (out / "errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "pred", "confidence"] + prob_cols)
        for r in errors:
            src = Path(r["file"])
            probs = {c[5:]: float(r[c]) for c in prob_cols}
            conf = max(probs.values())
            writer.writerow([str(src), r["label"], r["pred"], round(conf, 4)] + [r[c] for c in prob_cols])
            print(f"  {src.name}: 정답 {r['label']} → 예측 {r['pred']} (확신도 {conf:.2f})")
            if not src.exists():
                print(f"    경고: 원본 파일이 없습니다 ({src})")
                continue
            shutil.copy2(src, out / f"{r['label']}__{src.name}")
            stem = out / f"{r['label']}__{src.stem}"
            tile = annotate(src, r["label"], r["pred"], probs)
            tile.save(stem.with_name(stem.name + "_annotated.png"))
            if not args.no_mask:
                img_h = tile.height - 8 * 2 - 18 * (2 + len(prob_cols))   # annotate()의 사진 부분 높이
                panels, info = mask_panels(src, stem, img_h)
                print(f"    잎 마스크: 면적 {info.get('area_ratio')}, 잎 {info.get('leaves')}장")
                row = Image.new("RGB", (tile.width + sum(p.width for p in panels), tile.height), "white")
                row.paste(tile, (0, 0))
                x = tile.width
                for p in panels:
                    row.paste(p, (x, 0))
                    x += p.width
                tile = row
            tiles.append(tile)

    if tiles:
        cols = min(4 if args.no_mask else 2, len(tiles))
        rows_n = (len(tiles) + cols - 1) // cols
        w, h = max(t.width for t in tiles), max(t.height for t in tiles)
        grid = Image.new("RGB", (w * cols, h * rows_n), "white")
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * w, (i // cols) * h))
        grid.save(out / "errors_grid.png")
        print(f"모아 보기: {out / 'errors_grid.png'}")


if __name__ == "__main__":
    main()
