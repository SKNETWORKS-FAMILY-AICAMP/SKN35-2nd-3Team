"""원본 vs 정제본을 5-fold 전체(GroupKFold, fold 컬럼 기준)로 비교.
각 fold를 한 번씩 테스트셋으로 써서 5번 학습 후 평균±표준편차로 정리.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ORIGINAL = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"
REFINED = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset_refined.csv"

NON_FEATURE = {"store_id", "fold", "is_closed_next"}


def run_all_folds(path, label):
    df = pd.read_csv(path)
    target = "is_closed_next"
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]

    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category")

    fold_metrics = []
    importances_per_fold = []

    for test_fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != test_fold]
        test = df[df["fold"] == test_fold]

        X_train, y_train = train[feature_cols], train[target]
        X_test, y_test = test[feature_cols], test[target]

        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metrics.append({
            "fold": test_fold,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        importances_per_fold.append(pd.Series(model.feature_importances_, index=feature_cols))

    metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    mean_importance = pd.concat(importances_per_fold, axis=1).mean(axis=1).sort_values(ascending=False)
    return metrics_df, mean_importance


out_path = Path(__file__).resolve().parent / "compare_baseline_5fold_result.txt"
with open(out_path, "w", encoding="utf-8") as f:
    summary = {}
    for label, path in [("original", ORIGINAL), ("refined", REFINED)]:
        metrics_df, mean_importance = run_all_folds(path, label)

        f.write(f"=== {label}: fold별 성능 ===\n")
        f.write(metrics_df.to_string())
        f.write("\n\n")
        f.write(f"=== {label}: 평균 ± 표준편차 ===\n")
        f.write((metrics_df.mean().round(4).astype(str) + " ± " + metrics_df.std().round(4).astype(str)).to_string())
        f.write("\n\n")
        f.write(f"=== {label}: fold 평균 feature importance top 15 ===\n")
        f.write(mean_importance.head(15).to_string())
        f.write("\n\n")

        summary[label] = metrics_df.mean()

    f.write("=== 5-fold 평균 비교 요약 ===\n")
    f.write(pd.DataFrame(summary).to_string())
    f.write("\n")

print("done")
