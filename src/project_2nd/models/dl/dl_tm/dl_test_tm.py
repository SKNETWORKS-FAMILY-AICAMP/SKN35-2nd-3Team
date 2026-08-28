# -*- coding: utf-8 -*-
"""
dl_train_tm.py가 학습만 담당(ROC-AUC/PR-AUC/Lift, calibration까지만 계산)하고,
accuracy/precision/recall/f1은 이 스크립트에서 따로 계산함 — pjw님 파이프라인의
train_dnn.py/test_dnn.py 분리 구조와 동일하게 맞춤.

각 fold_k 모델을 그 모델이 학습에 전혀 안 쓴 fold k(진짜 held-out)에 대해
평가함. F1을 최대화하는 분류 임계값을 그 fold 자체에서 찾아서 적용
(대용량 fold라 스칼라 하나 고르는 거라 오버피팅 위험은 낮음 — 더 엄밀하게
하려면 별도 validation split에서 임계값을 찾고 test에는 적용만 해야 함).

산출물: models/dl/saved/test_metrics.json — export_model_dl_tm.py가 이 파일을
읽어서 models 테이블 등록용 accuracy/precision/recall/f1/roc_auc를 채움.

실행 (프로젝트 루트에서, dl_train_tm.py 학습 완료 후):
    python src/project_2nd/models/dl/dl_tm/dl_test_tm.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
)

from dl_train_tm import ClosureMLP, CONT_COLS, CAT_COLS, TARGET_COL, FOLD_COL, transform_cont, N_FOLDS


def load_fold_artifacts(artifact_dir: Path, k: int, econfig: dict):
    fold_dir = artifact_dir / f"fold_{k}"
    with open(fold_dir / "scaler.json", encoding="utf-8") as f:
        scaler = json.load(f)
    with open(fold_dir / "calibration.json", encoding="utf-8") as f:
        calib_info = json.load(f)

    model = ClosureMLP(
        n_cont=len(econfig["cont_cols"]),
        cat_cards=econfig["cat_cards"],
        emb_dims=econfig["emb_dims"],
    )
    state = torch.load(fold_dir / "model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    calib = calib_info.get("calibration_params", {"a": 1.0, "b": 0.0})
    return {
        "model": model,
        "cont_mean": np.array(scaler["mean"], dtype=np.float32),
        "cont_std": np.array(scaler["std"], dtype=np.float32),
        "calib_a": calib.get("a", 1.0),
        "calib_b": calib.get("b", 0.0),
    }


def predict_probs(fold_artifacts, df_sub, batch_size=8192):
    model = fold_artifacts["model"]
    cont_mean, cont_std = fold_artifacts["cont_mean"], fold_artifacts["cont_std"]
    calib_a, calib_b = fold_artifacts["calib_a"], fold_artifacts["calib_b"]

    n = len(df_sub)
    x_cont_all = (transform_cont(df_sub[CONT_COLS].to_numpy(dtype=np.float32)) - cont_mean) / cont_std
    x_cat_all = df_sub[CAT_COLS].to_numpy(dtype=np.int64)
    probs = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bc = torch.from_numpy(x_cont_all[i:i + batch_size])
            bcat = torch.from_numpy(x_cat_all[i:i + batch_size])
            logits = model(bc, bcat).numpy()
            calibrated_logits = calib_a * logits + calib_b
            probs[i:i + batch_size] = 1.0 / (1.0 + np.exp(-calibrated_logits))
    return probs


def find_best_threshold_f1(y_true, probs):
    """F1을 최대화하는 분류 임계값 탐색."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = np.where((precisions + recalls) > 0,
                    2 * precisions * recalls / (precisions + recalls + 1e-12), 0.0)
    best_idx = np.argmax(f1s[:-1])  # thresholds는 precisions/recalls보다 1개 적음
    return float(thresholds[best_idx])


def evaluate_fold(probs, y_true, label=""):
    auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    order = np.argsort(-probs)
    top5 = order[: max(1, int(len(order) * 0.05))]
    base_rate = float(y_true.mean())
    lift = float(y_true[top5].mean() / base_rate) if base_rate > 0 else float("nan")

    threshold = find_best_threshold_f1(y_true, probs)
    preds_binary = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds_binary))
    prec = float(precision_score(y_true, preds_binary, zero_division=0))
    rec = float(recall_score(y_true, preds_binary, zero_division=0))
    f1 = float(f1_score(y_true, preds_binary, zero_division=0))

    print(f"[{label}] ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  Top5%Lift={lift:.2f}x  "
          f"threshold={threshold:.4f}  Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}  (n={len(y_true)})")

    return {
        "roc_auc": auc, "pr_auc": pr_auc, "top5pct_lift": lift, "base_rate": base_rate,
        "threshold": threshold, "accuracy": acc, "precision_score": prec,
        "recall_score": rec, "f1_score": f1, "n": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/modeling_dataset_preprocessed_pmh.csv")
    ap.add_argument("--artifact-dir", default="models/dl/saved")
    ap.add_argument("--out", default="models/dl/saved/test_metrics.json")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    with open(artifact_dir / "ensemble_config.json", encoding="utf-8") as f:
        econfig = json.load(f)

    usecols = CONT_COLS + CAT_COLS + [FOLD_COL, TARGET_COL]
    dtype_map = {c: "float32" for c in CONT_COLS}
    dtype_map.update({c: "int32" for c in CAT_COLS})
    dtype_map[TARGET_COL] = "int8"
    dtype_map[FOLD_COL] = "int8"
    df = pd.read_csv(args.data, encoding="utf-8-sig", usecols=usecols, dtype=dtype_map)

    fold_metrics = []
    for k in range(args.n_folds):
        test_df = df[df[FOLD_COL] == k].reset_index(drop=True)
        fold_artifacts = load_fold_artifacts(artifact_dir, k, econfig)
        probs = predict_probs(fold_artifacts, test_df)
        y_true = test_df[TARGET_COL].to_numpy()
        metrics_k = evaluate_fold(probs, y_true, label=f"Fold {k}")
        fold_metrics.append(metrics_k)

    print(f"\n{'='*70}\n[전체 {args.n_folds}-fold 평균 테스트 성능]\n{'='*70}")
    summary = {}
    for key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("top5pct_lift", "Top5%Lift"),
                        ("accuracy", "Accuracy"), ("precision_score", "Precision"),
                        ("recall_score", "Recall"), ("f1_score", "F1")]:
        values = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        print(f"{label:10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    result = {"n_folds": args.n_folds, "fold_metrics": fold_metrics, "cv_summary": summary}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {out_path}")
    print("다음 단계: export_model_dl_tm.py로 models 테이블 등록용 JSON 만들기")


if __name__ == "__main__":
    main()