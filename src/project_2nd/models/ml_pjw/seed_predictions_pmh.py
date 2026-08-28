"""최종 모델(pmh_ml_extratrees_pjw_v1)로 실제 매장 50건에 대해 진짜 예측을 돌려서
TiDB predictions 테이블에 시연용 데이터로 넣는다.

- user_id는 전부 NULL(익명) - 실제 유저 기록이 아니라 데모/시연용 시드 데이터임을 명확히 함
- query_type='existing_store' (기존 매장 위험도 조회 시나리오)
- score/shap_top_features는 진짜 저장된 모델로 계산한 실제 값(가짜 아님)
- store_id/industry_code는 stores/industries 테이블에 이미 있는 값만 사용(FK 제약 충족)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shap"))
from explain_prediction import explain_batch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"
MODEL_PATH = Path(__file__).resolve().parent / "saved" / "best_model_extratrees_pmh.joblib"

sys.path.insert(0, str(REPO_ROOT))
from app.shared.db import get_engine  # noqa: E402

MODEL_ID = "pmh_ml_extratrees_pjw_v1"
N_SEED = 50
SEED = 42
TOP_K = 5

print("모델 불러오는 중 ...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

print("데이터 로딩 중 ...")
df = pd.read_csv(
    SRC,
    dtype={
        "store_id": str, "dong_code": str, "industry_code": str,
        "industry_jung_code": str, "industry_dae_code": str, "snapshot_date": str,
    },
)

enc_cols = [c for c in df.columns if c.endswith("_enc")]
cat_cols_original = [c for c in df.columns if c not in enc_cols and not pd.api.types.is_numeric_dtype(df[c])
                      and c not in {"store_id", "fold"}]
df_et = df.copy()
for c in cat_cols_original:
    enc_col = f"{c}_enc"
    if enc_col in df_et.columns and c in feature_cols:
        df_et[c] = df_et[enc_col]

test = df_et[df_et["fold"] == 4]
sample_idx = test.sample(n=N_SEED, random_state=SEED).index
sample = test.loc[sample_idx].reset_index(drop=True)
sample_raw = df.loc[sample_idx].reset_index(drop=True)  # industry_code 원본 문자열용

X = sample[feature_cols]
print(f"표본 {len(X)}건에 대해 예측 + SHAP 계산 중 ...")
proba = model.predict_proba(X)[:, 1]
top_features_all = explain_batch(model, X, top_k=TOP_K)

rows = []
now = datetime.now()
for i in range(len(sample)):
    rows.append({
        "model_id": MODEL_ID,
        "user_id": None,
        "query_type": "existing_store",
        "store_id": sample_raw.loc[i, "store_id"],
        "query_lat": None,
        "query_lng": None,
        "industry_code": sample_raw.loc[i, "industry_code"],
        "score": round(float(proba[i]), 5),
        "shap_top_features": json.dumps(top_features_all[i], ensure_ascii=False),
        "created_at": now,
    })

print(f"predictions 테이블에 {len(rows)}건 적재 중 ...")
engine = get_engine()
sql = text(
    "INSERT INTO predictions (model_id, user_id, query_type, store_id, query_lat, query_lng, "
    "industry_code, score, shap_top_features, created_at) "
    "VALUES (:model_id, :user_id, :query_type, :store_id, :query_lat, :query_lng, "
    ":industry_code, :score, :shap_top_features, :created_at)"
)
with engine.begin() as conn:
    conn.execute(sql, rows)

print(f"완료: {len(rows)}건 적재")
