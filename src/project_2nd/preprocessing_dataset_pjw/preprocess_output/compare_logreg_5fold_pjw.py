"""LogisticRegression으로 원본 vs 정제본을 5-fold 비교.
트리 모델(LightGBM)과 달리 상호작용을 스스로 못 만들기 때문에,
직접 만들어준 파생 비율 피처(업종특화도/1인당경쟁밀도/증감률)가
여기선 더 도움이 될 수 있다는 가설을 검증.
"""

import time
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ORIGINAL = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"
REFINED = Path(__file__).resolve().parents[4] / "data" / "processed" / "modeling_dataset_refined_pjw.csv"

NON_FEATURE = {"store_id", "fold", "is_closed_next"}


def build_pipeline(cat_cols, num_cols):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=100, solver="saga", class_weight="balanced", n_jobs=-1)),
    ])


def run_all_folds(path):
    df = pd.read_csv(path)
    target = "is_closed_next"
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    fold_metrics = []
    for test_fold in sorted(df["fold"].unique()):
        t0 = time.time()
        train = df[df["fold"] != test_fold]
        test = df[df["fold"] == test_fold]

        pipe = build_pipeline(cat_cols, num_cols)
        pipe.fit(train[feature_cols], train[target])

        proba = pipe.predict_proba(test[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_metrics.append({
            "fold": test_fold,
            "accuracy": accuracy_score(test[target], pred),
            "precision": precision_score(test[target], pred),
            "recall": recall_score(test[target], pred),
            "f1": f1_score(test[target], pred),
            "roc_auc": roc_auc_score(test[target], proba),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"fold {test_fold} done in {time.time()-t0:.1f}s", flush=True)

    return pd.DataFrame(fold_metrics).set_index("fold")


out_path = Path(__file__).resolve().parent / "compare_logreg_5fold_result_pjw.txt"
with open(out_path, "w", encoding="utf-8") as f:
    summary = {}
    for label, path in [("original", ORIGINAL), ("refined", REFINED)]:
        print(f"=== {label} 시작 ===", flush=True)
        metrics_df = run_all_folds(path)
        f.write(f"=== {label}: fold별 성능 ===\n")
        f.write(metrics_df.to_string())
        f.write("\n\n")
        f.write(f"=== {label}: 평균 ± 표준편차 ===\n")
        m = metrics_df.drop(columns="seconds")
        f.write((m.mean().round(4).astype(str) + " ± " + m.std().round(4).astype(str)).to_string())
        f.write("\n\n")
        summary[label] = m.mean()

    f.write("=== 5-fold 평균 비교 요약 ===\n")
    f.write(pd.DataFrame(summary).to_string())
    f.write("\n")

print("done")
