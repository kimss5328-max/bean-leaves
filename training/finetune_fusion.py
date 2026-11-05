"""예: python -m training.finetune --data dataset --epochs 3 --output artifacts/beans_resnet18"""
# =============================================================================
# finetune_fusion.py
#   [원본 구역]  1행 ~ run() 끝까지: 제공 training/finetune.py와 동일 (수정 없음)
#   [추가 구역]  아래 "융합 실험" 구역: tools/train_fusion.py를 학습 직후 자동 실행하도록 이식
#
#   원본과 동일하게 학습만 하려면 --fusion 없이 실행한다 (결과도 원본과 같다).
#   python -m training.finetune_fusion --data dataset --output artifacts/beans_resnet18_full \
#       --epochs 10 --batch-size 8 --lr 0.00005 --fusion
# =============================================================================
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_MODEL = "microsoft/resnet-18"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

class FolderImages(Dataset):
    def __init__(self, root: Path, classes: list[str], processor):
        self.records = []
        self.processor = processor
        for idx, name in enumerate(classes):
            folder = root / name
            if not folder.is_dir():
                raise ValueError(f"필요한 클래스 폴더가 없습니다: {folder}")
            self.records.extend((path, idx) for path in sorted(folder.iterdir())
                                if path.suffix.lower() in EXTS and path.is_file())
        if not self.records:
            raise ValueError(f"이미지가 없습니다: {root}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": torch.tensor(label, dtype=torch.long)}

def run(args):
    random.seed(42)
    torch.manual_seed(42)

    data = Path(args.data)
    train_root = data / "train"
    if not train_root.is_dir():
        raise ValueError("dataset/train/ 폴더를 먼저 준비하세요.")

    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다.")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_set = FolderImages(train_root, classes, processor)
    val_set = FolderImages(data / "val", classes, processor)

    train_counts = {n: sum(label == i for _, label in train_set.records) for i, n in enumerate(classes)}
    val_counts = {n: sum(label == i for _, label in val_set.records) for i, n in enumerate(classes)}

    if any(n < 2 for n in train_counts.values()) or any(n < 1 for n in val_counts.values()):
        raise ValueError("각 클래스에 학습 최소 2장, 검증 최소 1장이 필요합니다.")

    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)}")

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(classes),
        id2label={i: name for i, name in enumerate(classes)},
        label2id={name: i for i, name in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )

    # 기존 1000-class 분류 헤드는 크기가 맞지 않아 새롭게 초기화됩니다.
    # 초보자 첫 학습은 Backbone을 고정하고 마지막 분류 헤드만 학습합니다.
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    history = []
    best_acc = -1.0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            result = model(**batch)
            result.loss.backward()
            optim.step()
            losses.append(float(result.loss.item()))

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixel_values"].to(device)
                logits = model(pixel_values=pixels).logits
                correct += (logits.argmax(dim=-1).cpu() == batch["labels"]).sum().item()
                total += len(batch["labels"])

        acc = correct / total
        row = {
            "epoch": epoch + 1,
            "loss": round(sum(losses) / len(losses), 4),
            "val_accuracy": round(acc, 4)
        }
        history.append(row)
        print(row)

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(str(output), safe_serialization=True)
            processor.save_pretrained(str(output))

    (output / "training_report.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "classes": classes,
                "train_count": len(train_set),
                "val_count": len(val_set),
                "best_val_accuracy": best_acc,
                "history": history
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"저장 완료: {output} / best val accuracy={best_acc:.4f}")


# =============================================================================
# [추가 구역] 융합 실험 — tools/train_fusion.py 이식
#   학습이 끝난 뒤, 저장된 최고 모델(output 폴더)로만 동작한다. 위 run()에는 영향이 없다.
#
#   비교하는 방법 (모두 같은 Validation)
#     A.  CNN 단독              : CNN 확률의 최댓값 클래스
#     B.  색 단독 RF            : 색 특징 22개 → RandomForest (Train 학습)
#     C.  CNN + 색 규칙          : 정상/질병 판단이 엇갈리면 '확인 필요'로 보류
#     D.  CNN + 색 모델 확률 융합 : CNN 확률 3 + B의 확률 3 → RandomForest (stacking)
#     D2. CNN + 색 특징 원본 융합 : CNN 확률 3 + 색 특징 22 → RandomForest
#   융합 모델(D, D2) 학습 방식
#     cv    (기본) Validation 5겹 교차검증 — CNN이 학습하지 않은 이미지의 확률만 사용
#     train Train 학습 — Train의 CNN 확률은 과신되어 있어 참고용
#
#   색 특징 CSV는 미리 만들어 둔다:  python -m tools.focus_color_features --mask leaf
# =============================================================================
import csv

FUSION_META = {"split", "label", "file", "focus_found", "pred"}
FUSION_DEFAULT_DROP = ["focus_sharpness_mean", "mask_area_ratio"]


def predict_probs(model_dir: Path, data: Path, splits=("train", "val"), batch_size: int = 32) -> list[dict]:
    """저장된 모델로 이미지별 클래스 확률을 계산한다 (tools/cnn_predict.py와 같은 방식)."""
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    classes = [id2label[i] for i in range(len(id2label))]
    rows = []
    with torch.no_grad():
        for split in splits:
            items = []
            for c in classes:
                folder = data / split / c
                items += [(p, c) for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
            for i in range(0, len(items), batch_size):
                chunk = items[i:i + batch_size]
                images = []
                for p, _ in chunk:
                    with Image.open(p) as im:
                        images.append(im.convert("RGB"))
                pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                probs = torch.softmax(model(pixel_values=pixels).logits, dim=-1).numpy()
                for (p, y), pr in zip(chunk, probs):
                    row = {"split": split, "label": y, "file": str(p), "pred": classes[int(pr.argmax())]}
                    row.update({f"prob_{c}": float(v) for c, v in zip(classes, pr)})
                    rows.append(row)
    return rows


def _key(row: dict) -> tuple:
    return row["split"], row["label"], Path(row["file"]).stem


def _summarize(name, y, p, classes, healthy="healthy"):
    import numpy as np
    y, p = np.asarray(y), np.asarray(p)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes),) * 2, int)
    for a, b in zip(y, p):
        cm[idx[a], idx[b]] += 1
    sick_y, sick_p = y != healthy, p != healthy
    out = {"accuracy": round(float((y == p).mean()), 4), "correct": int((y == p).sum()), "total": int(len(y)),
           "binary_accuracy": round(float((sick_y == sick_p).mean()), 4),
           "disease_type_accuracy": round(float((y[sick_y] == p[sick_y]).mean()), 4) if sick_y.any() else 0.0,
           "disease_type_confusions": int((sick_y & sick_p & (y != p)).sum()),
           "sick_to_healthy": int((sick_y & ~sick_p).sum()),
           "healthy_to_sick": int((~sick_y & sick_p).sum()),
           "confusion_matrix": cm.tolist()}
    print(f"\n[{name}] Accuracy {out['accuracy']:.4f} ({out['correct']}/{out['total']})  "
          f"2분류 {out['binary_accuracy']:.3f}  질병종류 {out['disease_type_accuracy']:.3f}  "
          f"병↔병 {out['disease_type_confusions']}  병→정상 {out['sick_to_healthy']}  정상→병 {out['healthy_to_sick']}")
    for c, row in zip(classes, cm):
        print(f"   {c:>18} " + " ".join(f"{v:>4}" for v in row))
    return out


def _rf(seed=42):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)


def run_fusion(cnn_rows: list[dict], color_csv: Path, drop=FUSION_DEFAULT_DROP,
               scheme: str = "cv", folds: int = 5) -> dict:
    """tools/train_fusion.py의 main()과 같은 비교를 수행하고 결과 dict를 반환한다."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    with color_csv.open(encoding="utf-8") as f:
        color_rows = list(csv.DictReader(f))
    prob_cols = [c for c in cnn_rows[0] if c.startswith("prob_")]
    classes = [c[5:] for c in prob_cols]
    color_cols = [c for c in color_rows[0] if c not in FUSION_META and c not in drop]

    cnn_by = {_key(r): r for r in cnn_rows}
    joined = [(cnn_by[_key(r)], r) for r in color_rows if _key(r) in cnn_by]
    missing = len(color_rows) - len(joined)
    print(f"\n[융합] 클래스 {classes} / 색 특징 {len(color_cols)}개 / 결합 {len(joined)}행 (누락 {missing})")
    if missing:
        print("  경고: CNN 결과와 색 CSV의 이미지가 일부 맞지 않습니다 (split/label/파일명 기준).")

    def arrays(split):
        part = [(c, r) for c, r in joined if r["split"] == split]
        y = np.array([r["label"] for _, r in part])
        pc = np.array([[float(c[k]) for k in prob_cols] for c, _ in part])
        xc = np.array([[float(r[k]) for k in color_cols] for _, r in part])
        return y, pc, xc

    y_va, p_va, x_va = arrays("val")
    y_tr, p_tr, x_tr = arrays("train")
    cnn_pred = np.array([classes[i] for i in p_va.argmax(1)])
    res = {"A_cnn": _summarize("A. CNN 단독", y_va, cnn_pred, classes)}

    color_model = _rf().fit(x_tr, y_tr)               # 색 특징은 CNN과 무관 → Train 학습해도 누출 없음
    color_pred = color_model.predict(x_va)
    color_prob_va = color_model.predict_proba(x_va)

    if scheme == "cv":
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        fusion_pred = cross_val_predict(_rf(), np.hstack([p_va, color_prob_va]), y_va, cv=skf)
        fusion_raw_pred = cross_val_predict(_rf(), np.hstack([p_va, x_va]), y_va, cv=skf)
        tag = f"색 모델은 Train 학습 / 융합 모델은 Validation {folds}겹 교차검증"
    else:
        fusion_pred = _rf().fit(np.hstack([p_tr, color_model.predict_proba(x_tr)]), y_tr) \
            .predict(np.hstack([p_va, color_prob_va]))
        fusion_raw_pred = _rf().fit(np.hstack([p_tr, x_tr]), y_tr).predict(np.hstack([p_va, x_va]))
        tag = "모두 Train 학습 → Validation 평가 (CNN 과신 위험, 참고용)"
    print(f"\n학습 방식: {tag}")
    res["B_color"] = _summarize("B. 색 단독 RF", y_va, color_pred, classes)

    flag = (cnn_pred != "healthy") != (color_pred != "healthy")
    cnn_wrong = cnn_pred != y_va
    kept = ~flag
    rule = {"flagged": int(flag.sum()), "flag_rate": round(float(flag.mean()), 4),
            "cnn_errors_total": int(cnn_wrong.sum()),
            "cnn_errors_caught": int((flag & cnn_wrong).sum()),
            "correct_flagged": int((flag & ~cnn_wrong).sum()),
            "accuracy_on_kept": round(float((cnn_pred[kept] == y_va[kept]).mean()), 4) if kept.any() else None,
            "kept": int(kept.sum())}
    print(f"\n[C. CNN + 색 규칙] 보류 {rule['flagged']}장 ({rule['flag_rate']:.1%}) / "
          f"CNN 오답 {rule['cnn_errors_total']}개 중 {rule['cnn_errors_caught']}개를 보류로 걸러냄 / "
          f"맞았는데 보류된 것 {rule['correct_flagged']}장 / 자동 처리 {rule['kept']}장 정확도 {rule['accuracy_on_kept']}")
    res["C_rule"] = rule
    res["D_fusion_stack"] = _summarize("D. CNN 확률 + 색 모델 확률 융합 (6개)", y_va, fusion_pred, classes)
    res["D2_fusion_raw"] = _summarize("D2. CNN 확률 + 색 특징 원본 융합 (25개)", y_va, fusion_raw_pred, classes)
    return {"scheme": tag, "color_file": str(color_csv), "dropped": list(drop), "classes": classes, "results": res}


def after_training(args):
    """학습 직후: 확률 CSV 저장 → 융합 비교 → training_report.json에 'fusion' 항목 추가."""
    color_csv = Path(args.color)
    if not color_csv.exists():
        print(f"\n[융합 건너뜀] 색 특징 파일이 없습니다: {color_csv}\n"
              f"  먼저 실행: python -m tools.focus_color_features --mask leaf")
        return
    output = Path(args.output)
    print(f"\n[융합] 저장된 최고 모델로 확률 계산: {output}")
    rows = predict_probs(output, Path(args.data))

    probs_csv = Path("reports/data") / f"cnn_probs_{output.name}.csv"
    probs_csv.parent.mkdir(parents=True, exist_ok=True)
    with probs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {probs_csv}")

    fusion = run_fusion(rows, color_csv, args.drop, args.scheme, args.folds)
    fusion["cnn_probs_file"] = str(probs_csv)

    report_path = output / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["fusion"] = fusion                              # 원본 항목은 그대로 두고 추가만 한다
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n융합 결과 추가: {report_path} ('fusion' 항목)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- 원본 옵션 (finetune.py와 동일) ---
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="artifacts/beans_resnet18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    # --- 추가 옵션 (융합) ---
    parser.add_argument("--fusion", action="store_true", help="학습 직후 CNN + 색상 융합 비교 실행")
    parser.add_argument("--color", default="reports/data/color_features_leaf.csv", help="색 특징 CSV")
    parser.add_argument("--drop", nargs="*", default=FUSION_DEFAULT_DROP, help="색 특징에서 제외할 이름")
    parser.add_argument("--scheme", choices=["cv", "train"], default="cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true",
                        help="학습을 건너뛰고 --output의 기존 모델로 융합만 실행")
    args = parser.parse_args()
    if not args.skip_train:
        run(args)
    if args.fusion or args.skip_train:
        after_training(args)
