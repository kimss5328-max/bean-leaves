"""[2] 색상 특징 CSV로 RandomForest를 학습하고 Validation에서 평가한다.

[선택 개선 실험] CNN(ResNet18)과 같은 train/val 분할을 사용한다. Test는 사용하지 않는다.

사용 예
    python -m tools.train_color_ml
    python -m tools.train_color_ml --cnn-val-acc 0.9549   # CNN 결과와 나란히 출력
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

META_COLS = {"split", "label", "file", "focus_found"}


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"데이터가 없습니다: {path}")
    feats = [c for c in rows[0] if c not in META_COLS]
    return rows, feats


def split_xy(rows, feats, split):
    part = [r for r in rows if r["split"] == split]
    if not part:
        raise SystemExit(f"'{split}' 분할 데이터가 CSV에 없습니다.")
    x = np.array([[float(r[c]) for c in feats] for r in part], dtype=np.float32)
    y = np.array([r["label"] for r in part])
    focus = np.array([r["focus_found"] == "1" for r in part])
    return x, y, focus


def main():
    parser = argparse.ArgumentParser(description="색상 특징 RandomForest 학습·검증")
    parser.add_argument("--features", default="reports/data/color_features_leaf.csv")
    parser.add_argument("--output", default=None,
                        help="기본값: 특징 파일 이름에 맞춰 color_ml_report_<mask>[_drop].json")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--drop", nargs="*", default=[],
                        help="제외할 특징 이름 (예: focus_sharpness_mean) — 제거 실험용")
    parser.add_argument("--cnn-val-acc", type=float, default=None,
                        help="비교용 CNN Best Val Accuracy (예: 0.9549)")
    args = parser.parse_args()

    feat_path = Path(args.features)
    rows, feats = load(feat_path)
    unknown = [d for d in args.drop if d not in feats]
    if unknown:
        raise SystemExit(f"없는 특징 이름: {unknown}")
    feats = [c for c in feats if c not in args.drop]
    if args.output is None:
        stem = feat_path.stem.replace("color_features", "color_ml_report")
        args.output = str(feat_path.with_name(stem + ("_drop" if args.drop else "") + ".json"))
    print(f"특징 파일: {feat_path}" + (f" / 제외: {args.drop}" if args.drop else ""))
    x_tr, y_tr, f_tr = split_xy(rows, feats, "train")
    x_va, y_va, f_va = split_xy(rows, feats, "val")
    classes = sorted(set(y_tr))

    model = RandomForestClassifier(n_estimators=args.trees, random_state=42,
                                   class_weight="balanced", n_jobs=-1)
    model.fit(x_tr, y_tr)
    pred = model.predict(x_va)

    acc = accuracy_score(y_va, pred)
    cm = confusion_matrix(y_va, pred, labels=classes)
    per_class = classification_report(y_va, pred, labels=classes, output_dict=True, zero_division=0)
    top = sorted(zip(feats, model.feature_importances_), key=lambda t: t[1], reverse=True)[:10]

    print(f"Train {len(y_tr)}장 / Val {len(y_va)}장 / 특징 {len(feats)}개")
    print(f"초점 검출률  Train {f_tr.mean():.1%} / Val {f_va.mean():.1%}")
    print(f"\n[색상 RandomForest] Validation Accuracy = {acc:.4f} ({int((pred == y_va).sum())}/{len(y_va)})")
    if args.cnn_val_acc is not None:
        print(f"[CNN Head-only]      Validation Accuracy = {args.cnn_val_acc:.4f}  (비교 기준)")

    print("\n혼동행렬 (행=정답, 열=예측)")
    print(" " * 20 + "".join(f"{c[:12]:>14}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>20}" + "".join(f"{v:>14}" for v in row))

    print("\n클래스별 지표")
    for c in classes:
        m = per_class[c]
        print(f"{c:>20}  precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1-score']:.3f}")

    print("\n중요 특징 Top 10")
    for name, imp in top:
        print(f"  {name:<24} {imp:.4f}")

    report = {
        "method": "마스크 + HSV 색상 특징 + RandomForest",
        "features_file": str(feat_path),
        "dropped_features": args.drop,
        "status": "구현·측정 완료 (Validation 기준, Test 미사용)",
        "train_count": int(len(y_tr)), "val_count": int(len(y_va)),
        "feature_count": len(feats),
        "focus_detect_rate": {"train": round(float(f_tr.mean()), 4), "val": round(float(f_va.mean()), 4)},
        "val_accuracy": round(float(acc), 4),
        "cnn_val_accuracy_reference": args.cnn_val_acc,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {c: {k: round(per_class[c][k], 4) for k in ("precision", "recall", "f1-score")}
                      for c in classes},
        "top_features": [{"name": n, "importance": round(float(i), 4)} for n, i in top],
        "params": {"n_estimators": args.trees, "random_state": 42, "class_weight": "balanced"},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
