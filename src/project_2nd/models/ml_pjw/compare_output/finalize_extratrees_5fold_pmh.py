"""승자로 뽑힌 ExtraTrees(Optuna 튜닝 하이퍼파라미터 고정)를 5-fold 전체로 재검증.

tune_xgb_extratrees_pmh.py는 fold(0,1,2)=train / fold(3)=val / fold(4)=test 딱
1개 분할로만 최종 수치를 냈다 — minha님의 LightGBM/CatBoost 5-fold 평균과
공정 비교하려면 우리도 5-fold로 검증해야 한다(DL팀 계획서의 "상위 모델은
5-fold 추가 검증" 관례를 그대로 따름).

방법(minha님의 compare_lgbm_catboost_threshold_pmh.py와 동일한 방식):
1. 튜닝에서 찾은 하이퍼파라미터를 고정한 채 5-fold 전부 학습·예측
   (매 fold, 나머지 4개 fold로 학습 -> 해당 fold로 테스트)
2. 5개 fold의 out-of-fold(OOF) 확률을 전부 모아서 임계값 무관 지표(ROC-AUC, PR-AUC) 계산
   -> 이게 LightGBM/CatBoost의 OOF 수치와 바로 비교 가능한 값
3. OOF 전체에서 F1 최대화 임계값을 찾아 0.5 임계값과 다시 비교
"""

from pathlib import Path

import numpy as np
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

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"
OUT_DIR = Path(__file__).resolve().parent

NON_FEATURE = {"store_id", "fold", "is_closed_next"}
TARGET = "is_closed_next"

# tune_xgb_extratrees_pmh.py에서 찾은 승자(ExtraTrees) 최적 하이퍼파라미터 고정
WINNER_PARAMS = {
    "n_estimators": 280,
    "max_depth": 22,
    "min_samples_leaf": 35,
    "min_samples_split": 10,
    "max_features": 0.8175392099040567,
}

print("loading modeling_dataset_preprocessed_pmh.csv ...")
df = pd.read_csv(
    SRC,
    dtype={
        "store_id": str, "dong_code": str, "industry_code": str,
        "industry_jung_code": str, "industry_dae_code": str, "snapshot_date": str,
    },
)

enc_cols = [c for c in df.columns if c.endswith("_enc")]
base_feature_cols = [c for c in df.columns if c not in NON_FEATURE and c not in enc_cols]
cat_cols = [c for c in base_feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

df_et = df.copy()
for c in cat_cols:
    enc_col = f"{c}_enc"
    df_et[c] = df_et[enc_col] if enc_col in df_et.columns else LabelEncoder().fit_transform(df_et[c].astype(str))

oof_proba = np.full(len(df_et), np.nan)
fold_metrics = []

for k in sorted(df_et["fold"].unique()):
    print(f"=== fold {k} 학습 중 ===")
    train = df_et[df_et["fold"] != k]
    test = df_et[df_et["fold"] == k]

    model = ExtraTreesClassifier(**WINNER_PARAMS, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(train[base_feature_cols], train[TARGET])
    proba = model.predict_proba(test[base_feature_cols])[:, 1]
    oof_proba[test.index] = proba

    pred = (proba >= 0.5).astype(int)
    y_test = test[TARGET]
    fold_metrics.append({
        "fold": k,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    })
    print(f"  fold {k}: roc_auc={fold_metrics[-1]['roc_auc']:.4f}, pr_auc={fold_metrics[-1]['pr_auc']:.4f}")

metrics_df = pd.DataFrame(fold_metrics).set_index("fold")

# --- OOF 전체 지표 (minha님 threshold report와 동일 방식) ---
y_all = df_et[TARGET].values
oof_roc_auc = roc_auc_score(y_all, oof_proba)
oof_pr_auc = average_precision_score(y_all, oof_proba)

thresholds = [i / 200 for i in range(1, 200)]
f1_scores = [f1_score(y_all, (oof_proba >= t).astype(int), zero_division=0) for t in thresholds]
best_idx = max(range(len(thresholds)), key=lambda i: f1_scores[i])
best_threshold = thresholds[best_idx]

pred_05 = (oof_proba >= 0.5).astype(int)
pred_opt = (oof_proba >= best_threshold).astype(int)

report = []
report.append("# ExtraTrees(튜닝) 5-fold 재검증 (pmh 데이터)\n")
report.append(f"고정 하이퍼파라미터: {WINNER_PARAMS}\n")
report.append("## fold별 성능 (0.5 임계값)\n")
report.append(metrics_df.to_string())
report.append("\n")
report.append("## 평균 ± 표준편차\n")
report.append((metrics_df.mean().round(4).astype(str) + " ± " + metrics_df.std().round(4).astype(str)).to_string())
report.append("\n")
report.append("## OOF(5-fold 전체) 임계값 무관 지표 — minha님 LightGBM/CatBoost와 동일 방식 비교 가능\n")
report.append(f"- ROC-AUC: {oof_roc_auc:.4f}")
report.append(f"- PR-AUC: {oof_pr_auc:.4f}\n")
report.append("## OOF 기준 0.5 임계값 vs F1 최적 임계값\n")
report.append(f"| 지표 | 0.5 임계값 | 최적 임계값({best_threshold:.3f}) |")
report.append("|---|---|---|")
report.append(f"| accuracy | {accuracy_score(y_all, pred_05):.4f} | {accuracy_score(y_all, pred_opt):.4f} |")
report.append(f"| precision | {precision_score(y_all, pred_05, zero_division=0):.4f} | {precision_score(y_all, pred_opt, zero_division=0):.4f} |")
report.append(f"| recall | {recall_score(y_all, pred_05, zero_division=0):.4f} | {recall_score(y_all, pred_opt, zero_division=0):.4f} |")
report.append(f"| f1 | {f1_score(y_all, pred_05, zero_division=0):.4f} | {f1_score(y_all, pred_opt, zero_division=0):.4f} |")
report.append("")
report.append("## 팀 전체 비교 (최종)\n")
report.append("| | ROC-AUC(OOF/5-fold) | PR-AUC(OOF/5-fold) |")
report.append("|---|---|---|")
report.append("| minha님 LightGBM | 0.7476 | 0.4050 |")
report.append("| minha님 CatBoost | 0.7431 | 0.3965 |")
report.append("| 우리 XGBoost(기본값, 5-fold 평균) | 0.7455 | 0.4004 |")
report.append("| 우리 ExtraTrees(기본값, 5-fold 평균) | 0.7134 | 0.3504 |")
report.append(f"| **우리 ExtraTrees(튜닝, OOF 5-fold)** | **{oof_roc_auc:.4f}** | **{oof_pr_auc:.4f}** |")

out_md = OUT_DIR / "finalize_extratrees_5fold_report_pmh.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"완료: {out_md}")
print(f"OOF ROC-AUC={oof_roc_auc:.4f}, OOF PR-AUC={oof_pr_auc:.4f}, 최적임계값={best_threshold:.3f}")
