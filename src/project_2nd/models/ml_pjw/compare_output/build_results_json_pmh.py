"""pmh 데이터 기반 ML 비교/튜닝 결과 전체를 팀 공유용 JSON 하나로 정리.

수치 출처(전부 이 리포지토리 안의 실제 결과 파일에서 그대로 가져옴, 새로 계산 안 함):
  - LightGBM/CatBoost: preprocessing_dataset_pmh/compare_output/
      compare_lgbm_catboost_result_pmh.txt (기본 5-fold, precision/recall/f1/roc_auc)
      compare_lgbm_catboost_threshold_result_pmh.txt (OOF 임계값 무관 지표 roc_auc/pr_auc,
      F1 최적 임계값)
  - XGBoost/ExtraTrees(기본값): models/ml_pjw/compare_output/compare_xgb_extratrees_result_pmh.txt
  - ExtraTrees(튜닝, 최종): models/ml_pjw/compare_output/finalize_extratrees_5fold_report_pmh.md
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

result = {
    "dataset": "data/processed/modeling_dataset_preprocessed_pmh.csv",
    "target": "is_closed_next",
    "n_rows": 1889582,
    "closure_rate": 0.1065,
    "evaluation": {
        "cv_scheme": "GroupKFold(store_id), K=5, fold column shared across pjw/pmh datasets",
        "primary_metric": "pr_auc (average_precision)",
        "secondary_metrics": ["roc_auc", "accuracy", "precision", "recall", "f1"],
    },
    "known_caveat": "minha(pmh) preprocessing keeps `transitioned_next`, a column pjw identified as target-leakage-adjacent (mutually exclusive with the target at the same future timestep). All pmh-data model numbers below may be slightly inflated by it.",
    "models": [
        {
            "name": "LightGBM",
            "owner": "minha",
            "hyperparameters": "default",
            "n_features": 31,
            "eval_method": "5-fold OOF pooled",
            "metrics": {
                "accuracy": 0.906254, "precision": 0.755662, "recall": 0.176815,
                "f1": 0.286572, "roc_auc": 0.747568, "pr_auc": 0.404997,
            },
            "threshold_tuned": {
                "threshold": 0.205764,
                "metrics": {
                    "accuracy": 0.886603, "precision": 0.458041, "recall": 0.353986,
                    "f1": 0.399331, "roc_auc": 0.747568, "pr_auc": 0.404997,
                },
            },
            "source": "preprocessing_dataset_pmh/compare_output/compare_lgbm_catboost_threshold_result_pmh.txt",
        },
        {
            "name": "CatBoost",
            "owner": "minha",
            "hyperparameters": "default",
            "n_features": 31,
            "eval_method": "5-fold OOF pooled",
            "metrics": {
                "accuracy": 0.905866, "precision": 0.712437, "recall": 0.194509,
                "f1": 0.305581, "roc_auc": 0.743070, "pr_auc": 0.396501,
            },
            "threshold_tuned": {
                "threshold": 0.258849,
                "metrics": {
                    "accuracy": 0.884351, "precision": 0.444048, "recall": 0.339836,
                    "f1": 0.384953, "roc_auc": 0.743070, "pr_auc": 0.396501,
                },
            },
            "source": "preprocessing_dataset_pmh/compare_output/compare_lgbm_catboost_threshold_result_pmh.txt",
        },
        {
            "name": "XGBoost",
            "owner": "pjw",
            "hyperparameters": "default",
            "n_features": 31,
            "eval_method": "5-fold mean (per-fold, not OOF-pooled)",
            "metrics": {
                "accuracy": 0.906062, "precision": 0.733025, "recall": 0.185349,
                "f1": 0.295875, "roc_auc": 0.745466, "pr_auc": 0.400366,
            },
            "source": "models/ml_pjw/compare_output/compare_xgb_extratrees_result_pmh.txt",
        },
        {
            "name": "ExtraTrees",
            "owner": "pjw",
            "hyperparameters": "default",
            "n_features": 31,
            "eval_method": "5-fold mean (per-fold, not OOF-pooled)",
            "metrics": {
                "accuracy": 0.897277, "precision": 0.542513, "recall": 0.225602,
                "f1": 0.318672, "roc_auc": 0.713367, "pr_auc": 0.350362,
            },
            "source": "models/ml_pjw/compare_output/compare_xgb_extratrees_result_pmh.txt",
        },
        {
            "name": "ExtraTrees",
            "owner": "pjw",
            "hyperparameters": "optuna_tuned",
            "hyperparameters_detail": {
                "n_estimators": 280, "max_depth": 22, "min_samples_leaf": 35,
                "min_samples_split": 10, "max_features": 0.8175392099040567,
                "class_weight": "balanced", "random_state": 42,
            },
            "tuning": {
                "method": "Optuna TPE, 15 trials",
                "objective": "validation PR-AUC (fold 3, trained on fold 0,1,2)",
                "best_val_pr_auc": 0.4115,
            },
            "n_features": 31,
            "eval_method": "5-fold OOF pooled (confirmatory re-run of tuned hyperparameters)",
            "metrics": {
                "accuracy": 0.7877, "precision": 0.2599, "recall": 0.5378,
                "f1": 0.3505, "roc_auc": 0.7486, "pr_auc": 0.4123,
            },
            "threshold_tuned": {
                "threshold": 0.655,
                "metrics": {
                    "accuracy": 0.8881, "precision": 0.4668, "recall": 0.3578,
                    "f1": 0.4051, "roc_auc": 0.7486, "pr_auc": 0.4123,
                },
            },
            "is_best": True,
            "source": "models/ml_pjw/compare_output/finalize_extratrees_5fold_report_pmh.md",
        },
    ],
    "final_model": {
        "name": "ExtraTreesClassifier",
        "hyperparameters": {
            "n_estimators": 280, "max_depth": 22, "min_samples_leaf": 35,
            "min_samples_split": 10, "max_features": 0.8175392099040567,
            "class_weight": "balanced", "random_state": 42,
        },
        "recommended_threshold": 0.655,
        "metrics_at_recommended_threshold": {
            "accuracy": 0.8881, "precision": 0.4668, "recall": 0.3578,
            "f1": 0.4051, "roc_auc": 0.7486, "pr_auc": 0.4123,
        },
        "trained_on": "fold in [0,1,2,3] (train+validation), tested on fold 4",
        "shap_feature_importance": "models/shap/shap_feature_importance_pmh.json",
    },
}

out_path = OUT_DIR / "ml_results_summary_pmh.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"저장: {out_path}")
