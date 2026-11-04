"""GrabCut 잎 마스크로 배경을 지운 데이터셋(dataset_leaf/)을 만든다.

[선택 개선 실험] 원본 dataset/은 건드리지 않는다. finetune.py도 수정하지 않는다.
만든 폴더를 원본 학습 코드에 --data 옵션으로 넘겨 그대로 학습한다.

    python -m tools.make_masked_dataset                      # train, val 생성
    python -m training.finetune --data dataset_leaf --output artifacts/beans_resnet18_leaf \
        --epochs 30 --batch-size 8 --freeze-backbone

배경 처리
    --fill black : 배경을 검정으로 (기본)
    --fill mean  : 배경을 잎 평균색으로 (검정 경계 자체가 단서가 되는 것을 줄이고 싶을 때)
잎이 검출되지 않은 사진은 원본을 그대로 복사하고 summary.json에 기록한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.focus_recolor import EXTS, read_image, write_image
from tools.leaf_mask import leaf_mask

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def main():
    parser = argparse.ArgumentParser(description="잎 마스크 데이터셋 생성")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="dataset_leaf")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Test는 최종 평가 직전까지 만들지 않는다")
    parser.add_argument("--fill", choices=["black", "mean"], default="black")
    parser.add_argument("--limit", type=int, default=0, help="클래스당 최대 장수(0=전체)")
    args = parser.parse_args()

    src_root, dst_root = Path(args.data), Path(args.output)
    summary = {"fill": args.fill, "splits": {}, "no_leaf": []}
    for split in args.splits:
        for label in CLASSES:
            src = src_root / split / label
            if not src.is_dir():
                raise SystemExit(f"폴더가 없습니다: {src}")
            dst = dst_root / split / label
            dst.mkdir(parents=True, exist_ok=True)
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
            if args.limit:
                files = files[: args.limit]
            areas = []
            for path in files:
                bgr = read_image(path)
                mask, info = leaf_mask(bgr)
                out = bgr.copy()
                if mask.any():
                    bg = mask == 0
                    out[bg] = 0 if args.fill == "black" else bgr[~bg].mean(0).astype(np.uint8)
                    areas.append(info["area_ratio"])
                else:
                    summary["no_leaf"].append(str(path))
                # 확장자를 PNG로 통일하면 JPEG 재압축 손실이 없다 (finetune.py는 .png를 읽는다)
                write_image(dst / f"{path.stem}.png", out)
            key = f"{split}/{label}"
            summary["splits"][key] = {"count": len(files),
                                      "mean_leaf_area": round(float(np.mean(areas)), 4) if areas else None}
            print(f"{key}: {len(files)}장, 평균 잎 면적 {summary['splits'][key]['mean_leaf_area']}")

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료 → {dst_root} (잎 미검출로 원본 복사 {len(summary['no_leaf'])}장)")


if __name__ == "__main__":
    main()
