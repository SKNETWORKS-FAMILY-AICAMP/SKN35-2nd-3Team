"""원본 modeling_dataset.csv vs 정제된 modeling_dataset_refined.csv로 LightGBM을
동일 조건(fold==4를 테스트셋)으로 학습시켜 성능을 비교한다.
팀원분의 lightgbm_baseline.txt 검증 방식과 동일한 기준(튜닝 없음, 파이프라인 검증용).
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ORIGINAL = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"
REFINED = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset_refined.csv"

NON_FEATURE = {"store_id", "fold", "is_closed_next"}


def run(path, label):
    df = pd.read_csv(path)
    target = "is_closed_next"
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]

    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category")

    train = df[df["fold"] != 4]
    test = df[df["fold"] == 4]

    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_features": len(feature_cols),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return metrics, importances


results = {}
importances = {}
for label, path in [("original", ORIGINAL), ("refined", REFINED)]:
    m, imp = run(path, label)
    results[label] = m
    importances[label] = imp

out_path = Path(__file__).resolve().parent / "compare_baseline_result.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== 성능 비교 (fold==4 테스트셋) ===\n")
    f.write(pd.DataFrame(results).to_string())
    f.write("\n\n")
    for label in results:
        f.write(f"=== {label} feature importance top 15 ===\n")
        f.write(importances[label].head(15).to_string())
        f.write("\n\n")

print("done")
