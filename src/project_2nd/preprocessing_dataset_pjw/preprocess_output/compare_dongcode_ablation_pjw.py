"""정제본(refined) 기준으로 dong_code 있음/없음을 5-fold로 비교.
dong_code 그룹평균과 dong_historical_rate 그룹평균의 상관관계가 0.87로 완전
중복이 아니었어서, 원본 ID를 남겨야 하는지 검증.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

REFINED = Path(__file__).resolve().parents[4] / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
NON_FEATURE = {"store_id", "fold", "is_closed_next"}


def run_all_folds(df, feature_cols, target):
    df = df.copy()
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category")

    fold_metrics = []
    importances_per_fold = []
    for test_fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != test_fold]
        test = df[df["fold"] == test_fold]

        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(train[feature_cols], train[target])

        proba = model.predict_proba(test[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metrics.append({
            "fold": test_fold,
            "accuracy": accuracy_score(test[target], pred),
            "precision": precision_score(test[target], pred),
            "recall": recall_score(test[target], pred),
            "f1": f1_score(test[target], pred),
            "roc_auc": roc_auc_score(test[target], proba),
        })
        importances_per_fold.append(pd.Series(model.feature_importances_, index=feature_cols))

    metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    mean_importance = pd.concat(importances_per_fold, axis=1).mean(axis=1).sort_values(ascending=False)
    return metrics_df, mean_importance


df = pd.read_csv(REFINED)
target = "is_closed_next"
full_feature_cols = [c for c in df.columns if c not in NON_FEATURE]
no_dong_feature_cols = [c for c in full_feature_cols if c != "dong_code"]

out_path = Path(__file__).resolve().parent / "compare_dongcode_ablation_result_pjw.txt"
with open(out_path, "w", encoding="utf-8") as f:
    summary = {}
    for label, feature_cols in [("with_dong_code", full_feature_cols), ("without_dong_code", no_dong_feature_cols)]:
        metrics_df, mean_importance = run_all_folds(df, feature_cols, target)

        f.write(f"=== {label}: fold별 성능 ===\n")
        f.write(metrics_df.to_string())
        f.write("\n\n")
        f.write(f"=== {label}: 평균 ± 표준편차 ===\n")
        f.write((metrics_df.mean().round(4).astype(str) + " ± " + metrics_df.std().round(4).astype(str)).to_string())
        f.write("\n\n")
        f.write(f"=== {label}: fold 평균 feature importance top 10 ===\n")
        f.write(mean_importance.head(10).to_string())
        f.write("\n\n")

        summary[label] = metrics_df.mean()

    f.write("=== 비교 요약 ===\n")
    f.write(pd.DataFrame(summary).to_string())
    f.write("\n")

print("done")
