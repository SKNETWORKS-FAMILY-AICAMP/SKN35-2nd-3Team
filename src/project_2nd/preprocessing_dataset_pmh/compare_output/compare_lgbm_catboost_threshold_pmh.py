"""LightGBM vs CatBoost를 5-fold OOF(out-of-fold) 예측으로 비교하고,
고정 임계값(0.5) 대신 F1을 최대화하는 임계값을 찾아 재비교한다.

compare_lgbm_catboost_pmh.py와 같은 fold별 학습/평가 구조를 쓰되, 이번에는
각 fold의 테스트 확률(proba)을 전부 모아 하나의 OOF 배열로 합친 뒤:
  1. 임계값에 의존하지 않는 지표(ROC-AUC, PR-AUC/average precision)로 먼저 비교
  2. OOF 전체에서 F1을 최대화하는 임계값을 탐색(precision_recall_curve 기반)
  3. 그 임계값을 기준으로 fold별 accuracy/precision/recall/f1을 다시 계산해서
     기본 임계값(0.5) 결과와 나란히 비교

주의: 여기서 찾는 임계값은 "OOF 전체"에서 고른 하나의 전역 임계값이라, 개별
fold 입장에서는 자기 자신의 예측이 그 임계값 선택에 아주 약간 기여한다
(다른 4개 fold로 학습한 예측이므로 학습 과정 자체에 누수는 없음). 엄밀한
중첩 교차검증은 아니지만, 어떤 모델이 임계값 튜닝의 이득을 더 보는지
비교하는 목적에는 충분하다.

속도: CatBoost 기본 1000회 고정 반복이 고카디널리티 범주형(dong_code 428종,
industry_code 192종) 때문에 폴드당 40분 가까이 걸려서, train의 10%를
검증셋으로 떼어 early stopping(30라운드 무개선 시 중단)을 LightGBM/CatBoost
둘 다에 동일하게 적용해 불필요한 반복을 줄였다.
"""

from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                              precision_recall_curve, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import train_test_split

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
print(f"features: {len(feature_cols)}개 (범주형 {len(cat_cols)}개)")

for c in cat_cols:
    df[c] = df[c].astype("category")

y_true_all = df[TARGET].to_numpy()


def run_lgbm(train, test):
    tr, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train[TARGET])
    model = lgb.LGBMClassifier(random_state=42, verbose=-1, n_estimators=1000, learning_rate=0.1)
    model.fit(tr[feature_cols], tr[TARGET],
              eval_set=[(val[feature_cols], val[TARGET])], eval_metric='auc',
              callbacks=[lgb.early_stopping(30, verbose=False)])
    return model.predict_proba(test[feature_cols])[:, 1]


def run_catboost(train, test):
    tr, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train[TARGET])
    model = cb.CatBoostClassifier(random_state=42, verbose=False, cat_features=cat_cols,
                                   iterations=1000, learning_rate=0.1,
                                   early_stopping_rounds=30, eval_metric='AUC')
    model.fit(tr[feature_cols], tr[TARGET], eval_set=(val[feature_cols], val[TARGET]))
    return model.predict_proba(test[feature_cols])[:, 1]


RUNNERS = {"lightgbm": run_lgbm, "catboost": run_catboost}


def collect_oof(runner):
    """fold별로 학습해서 전체 행에 대한 OOF 확률을 원래 순서대로 채운다."""
    oof = np.zeros(len(df))
    fold_col = df["fold"].to_numpy()
    for test_fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != test_fold]
        test = df[df["fold"] == test_fold]
        proba = runner(train, test)
        oof[fold_col == test_fold] = proba
    return oof


def metrics_at_threshold(oof, fold_col, threshold):
    rows = []
    for f in sorted(np.unique(fold_col)):
        mask = fold_col == f
        y_t, y_p = y_true_all[mask], oof[mask]
        pred = (y_p >= threshold).astype(int)
        rows.append({
            "fold": f,
            "accuracy": accuracy_score(y_t, pred),
            "precision": precision_score(y_t, pred, zero_division=0),
            "recall": recall_score(y_t, pred, zero_division=0),
            "f1": f1_score(y_t, pred, zero_division=0),
        })
    return pd.DataFrame(rows).set_index("fold")


def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    # precision_recall_curve는 thresholds보다 prec/rec가 1개 더 많음(마지막은 임계값 무한대)
    best_idx = np.argmax(f1[:-1])
    return thr[best_idx], f1[best_idx]


out_path = Path(__file__).resolve().parent / "compare_lgbm_catboost_threshold_result_pmh.txt"
fold_col = df["fold"].to_numpy()

with open(out_path, "w", encoding="utf-8") as f:
    summary_default = {}
    summary_best = {}
    for label, runner in RUNNERS.items():
        print(f"=== {label} 5-fold OOF 예측 생성 중 ===")
        oof = collect_oof(runner)

        roc_auc = roc_auc_score(y_true_all, oof)
        pr_auc = average_precision_score(y_true_all, oof)
        best_thr, best_f1 = best_f1_threshold(y_true_all, oof)

        f.write(f"=== {label}: 임계값 무관 지표 (OOF 전체) ===\n")
        f.write(f"roc_auc={roc_auc:.6f}  pr_auc(average_precision)={pr_auc:.6f}\n\n")

        f.write(f"=== {label}: 기본 임계값(0.5) fold별 성능 ===\n")
        metrics_default = metrics_at_threshold(oof, fold_col, 0.5)
        f.write(metrics_default.to_string())
        f.write("\n\n")

        f.write(f"=== {label}: F1 최적 임계값={best_thr:.4f} (OOF 전체 F1={best_f1:.4f}) fold별 성능 ===\n")
        metrics_best = metrics_at_threshold(oof, fold_col, best_thr)
        f.write(metrics_best.to_string())
        f.write("\n\n")

        summary_default[label] = metrics_default.mean()
        summary_best[label] = metrics_best.mean()
        summary_best[label]["threshold"] = best_thr
        summary_default[label]["threshold"] = 0.5
        summary_default[label]["roc_auc"] = roc_auc
        summary_best[label]["roc_auc"] = roc_auc
        summary_default[label]["pr_auc"] = pr_auc
        summary_best[label]["pr_auc"] = pr_auc

    f.write("=== 기본 임계값(0.5) 평균 비교 요약 ===\n")
    f.write(pd.DataFrame(summary_default).to_string())
    f.write("\n\n")
    f.write("=== F1 최적 임계값 평균 비교 요약 ===\n")
    f.write(pd.DataFrame(summary_best).to_string())
    f.write("\n")

print(f"완료: {out_path}")
