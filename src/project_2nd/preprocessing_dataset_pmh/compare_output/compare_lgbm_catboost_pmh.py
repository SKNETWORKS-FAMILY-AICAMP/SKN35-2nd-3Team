"""LightGBM vs CatBoost를 5-fold 전체(fold 컬럼 기준)로 비교.

data/processed/modeling_dataset_preprocessed_pmh.csv(전처리 완료본)를 그대로 쓴다.
각 fold를 한 번씩 테스트셋으로 써서 5번 학습 후 평균±표준편차로 정리한다.
피처는 인코딩 전 원본 범주형 컬럼(문자열)을 그대로 category로 사용하고,
preprocess_modeling_dataset_pmh.py가 추가한 `_enc` 정수 컬럼은 같은 정보의
중복이라 피처에서 제외한다(라벨 인코딩보다 category 타입이 트리 분기에 더
유리하고, 두 표현을 동시에 넣으면 feature importance가 불필요하게 갈린다).
"""

from pathlib import Path

import catboost as cb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"

NON_FEATURE = {"store_id", "fold", "is_closed_next"}

print("loading modeling_dataset_preprocessed_pmh.csv ...")
df = pd.read_csv(SRC, dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                              'industry_jung_code': str, 'industry_dae_code': str,
                              'snapshot_date': str})
TARGET = "is_closed_next"
enc_cols = [c for c in df.columns if c.endswith('_enc')]
feature_cols = [c for c in df.columns if c not in NON_FEATURE and c not in enc_cols]
cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
print(f"features: {len(feature_cols)}개 (범주형 {len(cat_cols)}개: {cat_cols})")

for c in cat_cols:
    df[c] = df[c].astype("category")


def run_lgbm(train, test):
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    return proba, importance


def run_catboost(train, test):
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]
    model = cb.CatBoostClassifier(random_state=42, verbose=False, cat_features=cat_cols)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    importance = pd.Series(model.get_feature_importance(), index=feature_cols)
    return proba, importance


RUNNERS = {"lightgbm": run_lgbm, "catboost": run_catboost}


def run_all_folds(runner):
    fold_metrics = []
    importances_per_fold = []

    for test_fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != test_fold]
        test = df[df["fold"] == test_fold]
        y_test = test[TARGET]

        proba, importance = runner(train, test)
        pred = (proba >= 0.5).astype(int)

        fold_metrics.append({
            "fold": test_fold,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        })
        importances_per_fold.append(importance)

    metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    mean_importance = pd.concat(importances_per_fold, axis=1).mean(axis=1).sort_values(ascending=False)
    return metrics_df, mean_importance


out_path = Path(__file__).resolve().parent / "compare_lgbm_catboost_result_pmh.txt"
with open(out_path, "w", encoding="utf-8") as f:
    summary = {}
    for label, runner in RUNNERS.items():
        print(f"=== {label} 5-fold 학습 중 ===")
        metrics_df, mean_importance = run_all_folds(runner)

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

    f.write("=== 5-fold 평균 비교 요약 (LightGBM vs CatBoost) ===\n")
    f.write(pd.DataFrame(summary).to_string())
    f.write("\n")

print(f"완료: {out_path}")
