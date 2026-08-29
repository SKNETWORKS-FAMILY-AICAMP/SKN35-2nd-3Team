"""XGBoost / ExtraTreesClassifier를 pmh 정제본 기준 5-fold로 비교.

minha님의 compare_lgbm_catboost_pmh.py와 동일한 데이터·NON_FEATURE·fold 방식을
그대로 써서, 이미 나와 있는 LightGBM/CatBoost 5-fold 결과와 바로 이어붙여
비교할 수 있게 한다(모델만 바뀌고 조건은 통제).

- XGBoost: LightGBM/CatBoost와 동일하게 원본 문자열 범주형을 category dtype으로
  네이티브 지원(enable_categorical=True). `_enc` 컬럼은 중복이라 제외.
- ExtraTreesClassifier: sklearn 트리는 category dtype을 직접 못 받아서, 같은
  범주형 정보를 담은 `_enc` 라벨인코딩 컬럼으로 대체(원본 문자열 컬럼은 제외).
  즉 "같은 정보, 다른 표현"으로 정보량 자체는 두 모델이 동일하게 갖는다.

주의: minha님 원본 스크립트와 동일하게 `transitioned_next`를 제외하지 않았다
(feature set을 100% 동일하게 맞춰 모델만 비교하기 위함). transitioned_next는
pjw 쪽에서 타깃 누수로 판단해 제거한 컬럼이라, 절대 성능 수치는 다소 부풀려져
있을 수 있다 — 팀 공유 시 이 점 명시할 것.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"

NON_FEATURE = {"store_id", "fold", "is_closed_next"}
TARGET = "is_closed_next"

print("loading modeling_dataset_preprocessed_pmh.csv ...")
df = pd.read_csv(
    SRC,
    dtype={
        "store_id": str,
        "dong_code": str,
        "industry_code": str,
        "industry_jung_code": str,
        "industry_dae_code": str,
        "snapshot_date": str,
    },
)

enc_cols = [c for c in df.columns if c.endswith("_enc")]
base_feature_cols = [c for c in df.columns if c not in NON_FEATURE and c not in enc_cols]
cat_cols = [c for c in base_feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
print(f"공통 피처(minha님 기준): {len(base_feature_cols)}개 (범주형 {len(cat_cols)}개: {cat_cols})")

# --- XGBoost용: 원본 문자열 컬럼을 category dtype으로 ---
df_xgb = df.copy()
for c in cat_cols:
    df_xgb[c] = df_xgb[c].astype("category")
xgb_feature_cols = base_feature_cols

# --- ExtraTrees용: 문자열 범주형 대신 _enc 라벨인코딩 컬럼으로 치환 ---
df_et = df.copy()
for c in cat_cols:
    enc_col = f"{c}_enc"
    if enc_col in df_et.columns:
        df_et[c] = df_et[enc_col]
    else:
        # encoders_pmh.json에 없는 범주형(예: snapshot_date는 원래 수치형이라 여기 안옴)
        df_et[c] = LabelEncoder().fit_transform(df_et[c].astype(str))
et_feature_cols = base_feature_cols


def run_xgb(train, test):
    X_train, y_train = train[xgb_feature_cols], train[TARGET]
    X_test = test[xgb_feature_cols]
    model = XGBClassifier(
        random_state=42,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    importance = pd.Series(model.feature_importances_, index=xgb_feature_cols)
    return proba, importance


def run_extratrees(train, test):
    X_train, y_train = train[et_feature_cols], train[TARGET]
    X_test = test[et_feature_cols]
    model = ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    importance = pd.Series(model.feature_importances_, index=et_feature_cols)
    return proba, importance


RUNNERS = {
    "xgboost": (df_xgb, run_xgb),
    "extratrees": (df_et, run_extratrees),
}


def run_all_folds(data, runner):
    fold_metrics = []
    importances_per_fold = []

    for test_fold in sorted(data["fold"].unique()):
        train = data[data["fold"] != test_fold]
        test = data[data["fold"] == test_fold]
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
            "pr_auc": average_precision_score(y_test, proba),
        })
        importances_per_fold.append(importance)

    metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
    mean_importance = pd.concat(importances_per_fold, axis=1).mean(axis=1).sort_values(ascending=False)
    return metrics_df, mean_importance


out_path = Path(__file__).resolve().parent / "compare_xgb_extratrees_result_pmh.txt"
with open(out_path, "w", encoding="utf-8") as f:
    summary = {}
    for label, (data, runner) in RUNNERS.items():
        print(f"=== {label} 5-fold 학습 중 ===")
        metrics_df, mean_importance = run_all_folds(data, runner)

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

    f.write("=== 5-fold 평균 비교 요약 (XGBoost vs ExtraTrees) ===\n")
    f.write(pd.DataFrame(summary).to_string())
    f.write("\n")

print(f"완료: {out_path}")
