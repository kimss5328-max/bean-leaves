"""CNN 확률 + 색상 특징 융합 실험 (평가가이드 6-3장 C·D 방법론).

[선택 개선 실험] Test는 사용하지 않는다. 제공 코드는 수정하지 않는다.

입력
    --cnn    tools.cnn_predict 결과 (reports/data/cnn_probs_<모델>.csv)
    --color  tools.focus_color_features 결과 (reports/data/color_features_leaf.csv)

비교하는 방법 (모두 같은 Validation 133장)
    A. CNN 단독            : CNN 확률의 최댓값 클래스 (학습 없음)
    B.  색 단독 RF            : 색 특징 22개 → RandomForest (Train 1,034장 학습, E4와 동일)
    C.  CNN + 색 규칙          : CNN 결과는 그대로, 두 모델이 "정상/질병"에서 엇갈리면 '확인 필요'로 보류
    D.  CNN + 색 모델 확률 융합 : CNN 확률 3개 + B의 확률 3개 → RandomForest (stacking)
    D2. CNN + 색 특징 원본 융합 : CNN 확률 3개 + 색 특징 22개 → RandomForest

융합 모델(D, D2)의 학습 방식 (--scheme)
    cv    (기본) Validation 안에서 5겹 교차검증. CNN이 학습하지 않은 이미지의 확률만 사용하므로 공정하다.
    train      Train으로 학습. CNN은 Train 이미지를 이미 학습했으므로 Train의 CNN 확률이
               지나치게 확신에 차 있어 융합 모델이 CNN을 과신할 수 있다(참고용).

    python -m tools.train_fusion --cnn reports/data/cnn_probs_beans_resnet18.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

META = {"split", "label", "file", "focus_found", "pred"}
DEFAULT_DROP = ["focus_sharpness_mean", "mask_area_ratio"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row: dict) -> tuple:
    return row["split"], row["label"], Path(row["file"]).stem


def cm_of(y, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes),) * 2, int)
    for a, b in zip(y, p):
        cm[idx[a], idx[b]] += 1
    return cm


def summarize(name, y, p, classes, healthy="healthy"):
    y, p = np.asarray(y), np.asarray(p)
    cm = cm_of(y, p, classes)
    acc = float((y == p).mean())
    sick_y, sick_p = y != healthy, p != healthy
    binary = float((sick_y == sick_p).mean())
    both_sick = sick_y & sick_p
    type_acc = float((y[sick_y] == p[sick_y]).mean()) if sick_y.any() else 0.0
    out = {"accuracy": round(acc, 4), "correct": int((y == p).sum()), "total": int(len(y)),
           "binary_accuracy": round(binary, 4), "disease_type_accuracy": round(type_acc, 4),
           "disease_type_confusions": int((both_sick & (y != p)).sum()),
           "sick_to_healthy": int((sick_y & ~sick_p).sum()),
           "healthy_to_sick": int((~sick_y & sick_p).sum()),
           "confusion_matrix": cm.tolist()}
    print(f"\n[{name}] Accuracy {acc:.4f} ({out['correct']}/{out['total']})  "
          f"2분류 {binary:.3f}  질병종류 {type_acc:.3f}  "
          f"병↔병 {out['disease_type_confusions']}  병→정상 {out['sick_to_healthy']}  정상→병 {out['healthy_to_sick']}")
    for c, row in zip(classes, cm):
        print(f"   {c:>18} " + " ".join(f"{v:>4}" for v in row))
    return out


def rf(seed=42):
    return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)


def main():
    parser = argparse.ArgumentParser(description="CNN + 색상 융합 실험")
    parser.add_argument("--cnn", required=True)
    parser.add_argument("--color", default="reports/data/color_features_leaf.csv")
    parser.add_argument("--drop", nargs="*", default=DEFAULT_DROP, help="색 특징에서 제외할 이름")
    parser.add_argument("--scheme", choices=["cv", "train"], default="cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cnn_rows, color_rows = read_csv(Path(args.cnn)), read_csv(Path(args.color))
    prob_cols = [c for c in cnn_rows[0] if c.startswith("prob_")]
    classes = [c[5:] for c in prob_cols]
    color_cols = [c for c in color_rows[0] if c not in META and c not in args.drop]

    cnn_by = {key(r): r for r in cnn_rows}
    joined = [(cnn_by[key(r)], r) for r in color_rows if key(r) in cnn_by]
    missing = len(color_rows) - len(joined)
    print(f"클래스 {classes} / 색 특징 {len(color_cols)}개 / 결합 {len(joined)}행 (누락 {missing})")
    if missing:
        print("  경고: CNN CSV와 색 CSV의 이미지가 일부 맞지 않습니다 (split/label/파일명 기준).")

    def arrays(split):
        part = [(c, r) for c, r in joined if r["split"] == split]
        y = np.array([r["label"] for _, r in part])
        pc = np.array([[float(c[k]) for k in prob_cols] for c, _ in part])
        xc = np.array([[float(r[k]) for k in color_cols] for _, r in part])
        return y, pc, xc

    y_va, p_va, x_va = arrays("val")
    cnn_pred = np.array([classes[i] for i in p_va.argmax(1)])
    res = {"A_cnn": summarize("A. CNN 단독", y_va, cnn_pred, classes)}

    y_tr, p_tr, x_tr = arrays("train")
    # 색 모델은 CNN과 무관하므로 Train 1,034장으로 학습해도 누출이 없다 (E4와 같은 모델)
    color_model = rf().fit(x_tr, y_tr)
    color_pred = color_model.predict(x_va)
    color_prob_va = color_model.predict_proba(x_va)       # Val에 대한 색 모델 확률 (학습에 안 쓴 이미지)

    if args.scheme == "cv":
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        # D: CNN 확률 3개 + 색 모델 확률 3개 = 6개로 쌓기(stacking). Val 안에서 교차검증
        fusion_pred = cross_val_predict(rf(), np.hstack([p_va, color_prob_va]), y_va, cv=skf)
        # D2: CNN 확률 3개 + 색 특징 22개 원본. 학습 데이터(약 106장)에 비해 특징이 많아 불리하다
        fusion_raw_pred = cross_val_predict(rf(), np.hstack([p_va, x_va]), y_va, cv=skf)
        tag = f"색 모델은 Train 학습 / 융합 모델은 Validation {args.folds}겹 교차검증"
    else:
        fusion_pred = rf().fit(np.hstack([p_tr, color_model.predict_proba(x_tr)]), y_tr) \
            .predict(np.hstack([p_va, color_prob_va]))
        fusion_raw_pred = rf().fit(np.hstack([p_tr, x_tr]), y_tr).predict(np.hstack([p_va, x_va]))
        tag = "모두 Train 학습 → Validation 평가 (CNN 과신 위험, 참고용)"
    print(f"\n학습 방식: {tag}")
    res["B_color"] = summarize("B. 색 단독 RF", y_va, color_pred, classes)

    # C. 규칙: 정상/질병 판단이 엇갈리면 보류
    sick_cnn, sick_col = cnn_pred != "healthy", color_pred != "healthy"
    flag = sick_cnn != sick_col
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

    res["D_fusion_stack"] = summarize("D. CNN 확률 + 색 모델 확률 융합 (6개)", y_va, fusion_pred, classes)
    res["D2_fusion_raw"] = summarize("D2. CNN 확률 + 색 특징 원본 융합 (25개)", y_va, fusion_raw_pred, classes)

    out = Path(args.output or Path(args.cnn).with_name(
        Path(args.cnn).stem.replace("cnn_probs", "fusion_report") + f"_{args.scheme}.json"))
    out.write_text(json.dumps({"cnn_file": args.cnn, "color_file": args.color, "dropped": args.drop,
                               "scheme": tag, "classes": classes, "results": res},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
