import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ORIGINAL = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"
REFINED = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset_refined.csv"
NON_FEATURE = {"store_id", "fold", "is_closed_next"}


def run(path, label):
    t0 = time.time()
    df = pd.read_csv(path)
    target = "is_closed_next"
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"[{label}] load {time.time()-t0:.1f}s, cat_cols={cat_cols}, n_num={len(num_cols)}", flush=True)

    train = df[df["fold"] != 4]
    test = df[df["fold"] == 4]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=100, solver="saga", class_weight="balanced", n_jobs=-1)),
    ])

    t1 = time.time()
    pipe.fit(train[feature_cols], train[target])
    print(f"[{label}] fit {time.time()-t1:.1f}s", flush=True)

    proba = pipe.predict_proba(test[feature_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(test[target], pred),
        "precision": precision_score(test[target], pred),
        "recall": recall_score(test[target], pred),
        "f1": f1_score(test[target], pred),
        "roc_auc": roc_auc_score(test[target], proba),
    }
    print(f"[{label}] {metrics}", flush=True)
    return metrics


out_path = Path(__file__).resolve().parent / "compare_logreg_fold4_result.txt"
results = {}
for label, path in [("original", ORIGINAL), ("refined", REFINED)]:
    results[label] = run(path, label)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(pd.DataFrame(results).to_string())
    f.write("\n")

print("done")
