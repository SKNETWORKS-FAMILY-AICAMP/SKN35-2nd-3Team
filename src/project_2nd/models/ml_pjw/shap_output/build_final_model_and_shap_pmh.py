"""최종 모델(ExtraTrees, Optuna 튜닝) 저장 + SHAP 기반 설명 산출물 생성.

이전 시도(shap_extratrees_pmh.py)는 전역 중요도만 뽑고 모델도 SHAP 원값도 안 남겨서
"진짜 서빙에 쓸 수 있는 것"이 하나도 안 나왔다 — 이번엔 세 가지를 전부 만든다.

1. 학습된 모델을 파일로 저장 (best_model_extratrees_pmh.joblib)
   -> 앱이 실시간 예측할 때 이 파일을 불러 쓰면 됨(재학습 불필요)
2. 전역 피처 중요도 (report/발표용, shap_feature_importance_pmh.json)
3. shap_top_features_examples_pmh.json — 테스트 표본에 대해 models/shap/explain_prediction.py의
   공용 함수로 계산한 진짜 SHAP 값. predictions.shap_top_features와 스키마는 동일하지만,
   ⚠️ 이건 DB에 넣을 데이터가 아니라 "함수가 스키마대로 잘 작동하는지" 보여주는
   테스트/참고용 예시다 — predictions 테이블은 앱이 실행 중 실시간으로 채우는
   운영 데이터라(load_to_tidb.py 참고) 우리가 배치로 미리 적재하는 대상이 아님.

SHAP 계산은 한 번만 하고(compute_shap_matrix), 전역 랭킹과 개별 예측용 포맷 둘 다
그 결과에서 뽑아 쓴다 — 이전처럼 같은 계산을 중복으로 돌리지 않는다.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shap"))
from explain_prediction import compute_shap_matrix, format_shap_row  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"
SAVED_DIR = Path(__file__).resolve().parent.parent / "saved"
SAVED_DIR.mkdir(exist_ok=True)
# 결과 JSON은 팀 공용 폴더(models/shap/)에 저장 - explain_prediction.py와 같은 위치
OUT_DIR = Path(__file__).resolve().parents[2] / "shap"

NON_FEATURE = {"store_id", "fold", "is_closed_next"}
TARGET = "is_closed_next"
SHAP_SAMPLE_SIZE = 1000  # 이전 5000 -> 1000으로 축소 (전역 랭킹 안정성엔 충분, 계산 시간 단축)
N_EXAMPLES_TO_SAVE = 20  # shap_top_features 예시로 JSON에 저장할 개수(전체는 너무 커짐)
SEED = 42

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

trainval_mask = df_et["fold"].isin([0, 1, 2, 3])
test_mask = df_et["fold"] == 4

print("최종 모델(train+val, fold 0~3) 학습 중 ...", flush=True)
model = ExtraTreesClassifier(**WINNER_PARAMS, random_state=SEED, n_jobs=-1, class_weight="balanced")
model.fit(df_et.loc[trainval_mask, base_feature_cols], df_et.loc[trainval_mask, TARGET])
print("학습 완료", flush=True)

model_path = SAVED_DIR / "best_model_extratrees_pmh.joblib"
joblib.dump({"model": model, "feature_cols": base_feature_cols, "hyperparameters": WINNER_PARAMS,
             "threshold": 0.655, "trained_on": "fold in [0,1,2,3]"}, model_path)
print(f"모델 저장: {model_path}", flush=True)

test_df = df_et.loc[test_mask, base_feature_cols]
store_ids = df_et.loc[test_mask, "store_id"]
sample_idx = test_df.sample(n=min(SHAP_SAMPLE_SIZE, len(test_df)), random_state=SEED).index
sample_X = test_df.loc[sample_idx].reset_index(drop=True)
sample_store_ids = store_ids.loc[sample_idx].reset_index(drop=True)
print(f"SHAP 계산용 표본: {len(sample_X):,}행 (test/fold4에서 랜덤 추출)", flush=True)

print("SHAP 값 계산 중 (TreeExplainer, 한 번만) ...", flush=True)
sv_all = compute_shap_matrix(model, sample_X)
print("SHAP 계산 완료", flush=True)

# --- 1) 전역 피처 중요도 ---
mean_abs_shap = np.abs(sv_all).mean(axis=0)
mean_shap = sv_all.mean(axis=0)
global_result = pd.DataFrame({
    "feature": base_feature_cols,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
global_result["rank"] = global_result.index + 1
total = global_result["mean_abs_shap"].sum()
global_result["pct_of_total"] = (global_result["mean_abs_shap"] / total * 100).round(2)

global_json = {
    "model": "ExtraTreesClassifier (Optuna tuned)",
    "hyperparameters": WINNER_PARAMS,
    "explain_method": "SHAP TreeExplainer",
    "sample": {"source": "fold4 (test)", "n_rows": int(len(sample_X)), "seed": SEED},
    "feature_importance": [
        {
            "rank": int(row["rank"]), "feature": row["feature"],
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
            "mean_shap": round(float(row["mean_shap"]), 6),
            "pct_of_total": float(row["pct_of_total"]),
        }
        for _, row in global_result.iterrows()
    ],
}
global_path = OUT_DIR / "shap_feature_importance_pmh.json"
with open(global_path, "w", encoding="utf-8") as f:
    json.dump(global_json, f, ensure_ascii=False, indent=2)
print(f"저장: {global_path}", flush=True)

# --- 2) 실제 개별 예측용 shap_top_features (predictions 테이블 스키마와 동일 형식) ---
examples = []
for i in range(min(N_EXAMPLES_TO_SAVE, len(sample_X))):
    top_features = format_shap_row(sv_all[i], sample_X.iloc[i], base_feature_cols, top_k=5)
    examples.append({
        "store_id": sample_store_ids.iloc[i],
        "shap_top_features": top_features,
    })

examples_json = {
    "schema_source": "src/project_2nd/db/테이블_설명.md (predictions.shap_top_features)",
    "note": (
        "실제 학습된 모델로 계산한 진짜 SHAP 값(가짜 데이터 아님) — 다만 이건 DB에 넣을 데이터가 "
        "아니라 explain_prediction()/explain_batch() 함수가 스키마대로 잘 작동하는지 보여주는 "
        "테스트/참고용 예시다. predictions 테이블은 앱이 실시간으로 채우는 운영 데이터라 "
        "우리가 배치로 미리 적재하지 않는다(db/etl/load_to_tidb.py 참고)."
    ),
    "n_examples": len(examples),
    "example_predictions": examples,
}
examples_path = OUT_DIR / "shap_top_features_examples_pmh.json"
with open(examples_path, "w", encoding="utf-8") as f:
    json.dump(examples_json, f, ensure_ascii=False, indent=2)
print(f"저장: {examples_path}", flush=True)

print("\n=== 완료 요약 ===")
print(f"모델: {model_path}")
print(f"전역 중요도: {global_path}")
print(f"개별 예측 예시({len(examples)}건): {examples_path}")
print("\ntop 10 전역 중요도:")
print(global_result.head(10)[["rank", "feature", "mean_abs_shap", "pct_of_total"]].to_string(index=False))
