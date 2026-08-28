"""XGBoost/ExtraTrees를 Optuna로 튜닝하고, 이긴 쪽에 임계값 최적화까지 적용.

DL팀 계획서(사람용 - 데스크탑 딥러닝 작업관리표.xlsx)의 고정분할 관례를 그대로 재사용:
  - train: fold 0,1,2
  - validation: fold 3 (Optuna 탐색 목적함수 + 임계값 튜닝 전용)
  - test: fold 4 (마지막에 딱 한 번만 확인, 탐색에는 절대 사용 안 함)

1. Optuna로 XGBoost/ExtraTrees 각각 튜닝 (목적함수 = validation PR-AUC, 팀 표준 1순위 지표)
2. validation PR-AUC 더 높은 쪽을 승자로 선택
3. 승자 모델의 validation 확률 분포에서 F1 최대화 임계값 탐색
4. train+validation(fold 0~3)으로 재학습 후 test(fold 4)에서 0.5 임계값 vs 최적 임계값 최종 비교

compare_xgb_extratrees_pmh.py(기본 하이퍼파라미터 5-fold 비교)와 같은 피처 구성을 그대로 사용.
"""

from pathlib import Path

import optuna
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


def progress_callback(name, n_trials):
    def _cb(study, trial):
        print(f"[{name}] trial {trial.number + 1}/{n_trials} done — value={trial.value:.4f}, best={study.best_value:.4f}", flush=True)
    return _cb

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"
OUT_DIR = Path(__file__).resolve().parent

NON_FEATURE = {"store_id", "fold", "is_closed_next"}
TARGET = "is_closed_next"
XGB_TRIALS = 40
ET_TRIALS = 15
SEED = 42

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

df_xgb = df.copy()
for c in cat_cols:
    df_xgb[c] = df_xgb[c].astype("category")

df_et = df.copy()
for c in cat_cols:
    enc_col = f"{c}_enc"
    df_et[c] = df_et[enc_col] if enc_col in df_et.columns else LabelEncoder().fit_transform(df_et[c].astype(str))

train_mask = df["fold"].isin([0, 1, 2])
val_mask = df["fold"] == 3
test_mask = df["fold"] == 4
trainval_mask = df["fold"].isin([0, 1, 2, 3])

print(f"train={train_mask.sum():,} / val={val_mask.sum():,} / test={test_mask.sum():,}")


def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


# ============================================================
# 1. Optuna 튜닝
# ============================================================

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
    }
    model = XGBClassifier(
        **params, random_state=SEED, enable_categorical=True, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=30,
    )
    model.fit(
        df_xgb.loc[train_mask, base_feature_cols], df_xgb.loc[train_mask, TARGET],
        eval_set=[(df_xgb.loc[val_mask, base_feature_cols], df_xgb.loc[val_mask, TARGET])],
        verbose=False,
    )
    proba = model.predict_proba(df_xgb.loc[val_mask, base_feature_cols])[:, 1]
    return average_precision_score(df_xgb.loc[val_mask, TARGET], proba)


def et_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }
    model = ExtraTreesClassifier(**params, random_state=SEED, n_jobs=-1, class_weight="balanced")
    model.fit(df_et.loc[train_mask, base_feature_cols], df_et.loc[train_mask, TARGET])
    proba = model.predict_proba(df_et.loc[val_mask, base_feature_cols])[:, 1]
    return average_precision_score(df_et.loc[val_mask, TARGET], proba)


report = []
report.append("# XGBoost/ExtraTrees Optuna 튜닝 + 임계값 최적화 (pmh 데이터)\n")
report.append(f"train=fold(0,1,2) {train_mask.sum():,}행 · validation=fold(3) {val_mask.sum():,}행 · test=fold(4) {test_mask.sum():,}행\n")
report.append("목적함수: validation PR-AUC (팀 표준 1순위 지표)\n")

print(f"=== XGBoost Optuna 튜닝 ({XGB_TRIALS} trials) ===")
study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=XGB_TRIALS, show_progress_bar=False, callbacks=[progress_callback("xgboost", XGB_TRIALS)])
print(f"XGBoost best val PR-AUC: {study_xgb.best_value:.4f}")
print(f"XGBoost best params: {study_xgb.best_params}")

print(f"=== ExtraTrees Optuna 튜닝 ({ET_TRIALS} trials) ===")
study_et = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study_et.optimize(et_objective, n_trials=ET_TRIALS, show_progress_bar=False, callbacks=[progress_callback("extratrees", ET_TRIALS)])
print(f"ExtraTrees best val PR-AUC: {study_et.best_value:.4f}")
print(f"ExtraTrees best params: {study_et.best_params}")

report.append("## 튜닝 결과 (validation PR-AUC 기준)\n")
report.append(f"- XGBoost: {XGB_TRIALS} trials, best val PR-AUC = {study_xgb.best_value:.4f}")
report.append(f"  - best params: {study_xgb.best_params}")
report.append(f"- ExtraTrees: {ET_TRIALS} trials, best val PR-AUC = {study_et.best_value:.4f}")
report.append(f"  - best params: {study_et.best_params}\n")

# ============================================================
# 2. 승자 선택
# ============================================================
if study_xgb.best_value >= study_et.best_value:
    winner = "xgboost"
    winner_params = study_xgb.best_params
    df_winner = df_xgb
else:
    winner = "extratrees"
    winner_params = study_et.best_params
    df_winner = df_et

print(f"=== 승자: {winner} ===")
report.append(f"## 승자: **{winner}** (validation PR-AUC 더 높은 쪽)\n")

# ============================================================
# 3. 승자 재학습(train만) -> validation에서 임계값 탐색
# ============================================================
if winner == "xgboost":
    winner_model_val = XGBClassifier(
        **winner_params, random_state=SEED, enable_categorical=True, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=30,
    )
    winner_model_val.fit(
        df_winner.loc[train_mask, base_feature_cols], df_winner.loc[train_mask, TARGET],
        eval_set=[(df_winner.loc[val_mask, base_feature_cols], df_winner.loc[val_mask, TARGET])],
        verbose=False,
    )
else:
    winner_model_val = ExtraTreesClassifier(**winner_params, random_state=SEED, n_jobs=-1, class_weight="balanced")
    winner_model_val.fit(df_winner.loc[train_mask, base_feature_cols], df_winner.loc[train_mask, TARGET])

val_proba = winner_model_val.predict_proba(df_winner.loc[val_mask, base_feature_cols])[:, 1]
val_y = df_winner.loc[val_mask, TARGET]

thresholds = [i / 200 for i in range(1, 200)]
f1_scores = [f1_score(val_y, (val_proba >= t).astype(int), zero_division=0) for t in thresholds]
best_idx = max(range(len(thresholds)), key=lambda i: f1_scores[i])
best_threshold = thresholds[best_idx]
print(f"validation에서 F1 최대화 임계값: {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")
report.append(f"## 임계값 최적화 (validation, F1 기준)\n")
report.append(f"- 최적 임계값: **{best_threshold:.3f}** (validation F1={f1_scores[best_idx]:.4f})\n")

# ============================================================
# 4. train+validation(fold 0~3)으로 재학습 -> test(fold 4)에서 최종 1회 평가
# ============================================================
print("=== 최종 평가: train+val(fold 0~3) 재학습 -> test(fold 4) ===")
if winner == "xgboost":
    # 최종 모델은 early stopping용 val이 없으니, 튜닝 때 찾은 n_estimators를 그대로 고정 학습
    final_params = dict(winner_params)
    final_model = XGBClassifier(
        **final_params, random_state=SEED, enable_categorical=True, tree_method="hist", eval_metric="aucpr",
    )
else:
    final_model = ExtraTreesClassifier(**winner_params, random_state=SEED, n_jobs=-1, class_weight="balanced")

final_model.fit(df_winner.loc[trainval_mask, base_feature_cols], df_winner.loc[trainval_mask, TARGET])
test_proba = final_model.predict_proba(df_winner.loc[test_mask, base_feature_cols])[:, 1]
test_y = df_winner.loc[test_mask, TARGET]

metrics_default = metrics_at(test_y, test_proba, 0.5)
metrics_tuned = metrics_at(test_y, test_proba, best_threshold)

report.append("## 최종 테스트(fold 4, 최초 1회 확인) — 0.5 임계값 vs 최적 임계값\n")
report.append("| 지표 | 0.5 임계값 | 최적 임계값({:.3f}) |".format(best_threshold))
report.append("|---|---|---|")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
    report.append(f"| {k} | {metrics_default[k]:.4f} | {metrics_tuned[k]:.4f} |")
report.append("")

out_md = OUT_DIR / "tune_xgb_extratrees_report_pmh.md"
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"완료: {out_md}")
print(f"승자: {winner}, 최종 test ROC-AUC(0.5)={metrics_default['roc_auc']:.4f}, PR-AUC(0.5)={metrics_default['pr_auc']:.4f}")
print(f"최적 임계값 F1: {metrics_tuned['f1']:.4f} vs 0.5 임계값 F1: {metrics_default['f1']:.4f}")
