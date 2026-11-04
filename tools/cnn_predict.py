"""학습된 CNN 모델로 이미지별 클래스 확률을 CSV로 저장하고, 분할별 혼동행렬을 출력한다.

[선택 개선 실험] 학습은 하지 않는다. finetune.py·서빙 코드는 수정하지 않는다.
전처리는 모델 폴더의 preprocessor_config.json을 그대로 사용한다 (학습·서빙과 동일).

    python -m tools.cnn_predict --model artifacts/beans_resnet18
    python -m tools.cnn_predict --model artifacts/beans_resnet18_leaf --data dataset_leaf

출력: reports/data/cnn_probs_<모델폴더명>.csv
    split, label, file, pred, prob_<class> ...
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def confusion(labels, preds, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for y, p in zip(labels, preds):
        cm[idx[y], idx[p]] += 1
    return cm


def print_report(name, labels, preds, classes):
    cm = confusion(labels, preds, classes)
    acc = np.trace(cm) / max(cm.sum(), 1)
    print(f"\n[{name}] Accuracy = {acc:.4f} ({np.trace(cm)}/{cm.sum()})")
    print(" " * 20 + "".join(f"{c[:12]:>14}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>20}" + "".join(f"{v:>14}" for v in row))
    return acc, cm


def main():
    parser = argparse.ArgumentParser(description="CNN 이미지별 확률 추출")
    parser.add_argument("--model", default="artifacts/beans_resnet18")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_dir = Path(args.model)
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    classes = [id2label[i] for i in range(len(id2label))]
    print(f"모델: {model_dir} / 클래스: {classes}")

    out = Path(args.output or f"reports/data/cnn_probs_{model_dir.name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with out.open("w", newline="", encoding="utf-8") as f, torch.no_grad():
        writer = csv.writer(f)
        writer.writerow(["split", "label", "file", "pred"] + [f"prob_{c}" for c in classes])
        for split in args.splits:
            items = []
            for c in classes:
                folder = Path(args.data) / split / c
                if not folder.is_dir():
                    raise SystemExit(f"폴더가 없습니다: {folder}")
                items += [(p, c) for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
            labels, preds = [], []
            for i in range(0, len(items), args.batch_size):
                chunk = items[i:i + args.batch_size]
                images = []
                for p, _ in chunk:
                    with Image.open(p) as im:
                        images.append(im.convert("RGB"))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                probs = torch.softmax(model(pixel_values=pixels).logits, dim=-1).numpy()
                for (p, y), pr in zip(chunk, probs):
                    pred = classes[int(pr.argmax())]
                    labels.append(y)
                    preds.append(pred)
                    writer.writerow([split, y, str(p), pred] + [round(float(x), 6) for x in pr])
            acc, cm = print_report(f"CNN {split}", labels, preds, classes)
            results[split] = {"accuracy": round(float(acc), 4), "confusion_matrix": cm.tolist()}

    report = out.with_suffix(".json")
    report.write_text(json.dumps({"model": str(model_dir), "data": args.data, "classes": classes,
                                  "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}\n저장: {report}")
    if "train" in results:
        print("참고: train 정확도는 학습에 쓴 이미지라 실제 성능보다 높게 나온다.")


if __name__ == "__main__":
    main()
