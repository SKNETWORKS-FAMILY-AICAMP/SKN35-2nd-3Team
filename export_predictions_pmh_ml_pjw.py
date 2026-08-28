# -*- coding: utf-8 -*-
"""
pmh_ml_extratrees_pjw_v1 모델로 실제 매장 표본에 대해 예측 + SHAP을 계산해서
data/features/predictions_for_db_pmh_ml_pjw.json으로 저장한다.
(write_model.py의 load_predictions_json_to_db()가 그대로 읽을 수 있는 형식)

DB 접속 불필요 - 로컬 모델 파일(models/ml_pjw/saved/best_model_extratrees_pmh.joblib)과
전처리된 CSV만 있으면 실행 가능.

실행 (프로젝트 루트에서):
    python export_predictions_pmh_ml_pjw.py
    python export_predictions_pmh_ml_pjw.py --n-samples 100
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "project_2nd" / "models" / "shap"))
from explain_prediction import explain_batch  # noqa: E402

MODEL_ID = "pmh_ml_extratrees_pjw_v1"
SRC = Path("data/processed/modeling_dataset_preprocessed_pmh.csv")
MODEL_PATH = Path("src/project_2nd/models/ml_pjw/saved/best_model_extratrees_pmh.joblib")
TOP_K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/features/predictions_for_db_pmh_ml_pjw.json")
    ap.add_argument("--snapshot-date", default=None,
                     help="이 시점 데이터로만 예측 (기본: 데이터 내 최신 스냅샷, "
                          "DL팀 shap_explain_tm.py와 동일한 관례 - df['snapshot_date'].max())")
    args = ap.parse_args()

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
    cat_cols_original = [
        c for c in df.columns
        if c not in enc_cols and c not in {"store_id", "fold"} and not pd.api.types.is_numeric_dtype(df[c])
    ]
    df_et = df.copy()
    for c in cat_cols_original:
        enc_col = f"{c}_enc"
        if enc_col in df_et.columns and c in feature_cols:
            df_et[c] = df_et[enc_col]

    # 최신 실제 스냅샷만 사용 - 임의 과거 시점 섞지 않고 "지금 시점 기준 위험도"를
    # 보여주는 데모이므로 최신 데이터로만 뽑는다. build_modeling_dataset.py가
    # label_available=False인 마지막 원본 스냅샷(서빙 전용)은 이미 걸러내므로,
    # 이 데이터 안에서의 max()가 곧 최신 "라벨 있는" 스냅샷(2026-08-28 기준 202512,
    # 2025년 12월)이다 - DL팀 shap_explain_tm.py와 동일한 관례.
    target_snapshot = args.snapshot_date or df_et["snapshot_date"].astype(str).max()
    print(f"사용 스냅샷: {target_snapshot}")
    # (fold==4는 학습에 안 쓰인 test 구간 유지 - 방법론 일관성)
    test = df_et[(df_et["fold"] == 4) & (df_et["snapshot_date"].astype(str) == target_snapshot)]
    sample_idx = test.sample(n=args.n_samples, random_state=args.seed).index
    sample = test.loc[sample_idx].reset_index(drop=True)
    sample_raw = df.loc[sample_idx].reset_index(drop=True)  # industry_code 원본 문자열용

    X = sample[feature_cols]
    print(f"표본 {len(X)}건에 대해 예측 + SHAP 계산 중 ...")
    proba = model.predict_proba(X)[:, 1]
    top_features_all = explain_batch(model, X, top_k=TOP_K)

    records = []
    for i in range(len(sample)):
        records.append({
            "model_id": MODEL_ID,
            "user_id": None,
            "query_type": "existing_store",
            "store_id": sample_raw.loc[i, "store_id"],
            "query_lat": None,
            "query_lng": None,
            "industry_code": sample_raw.loc[i, "industry_code"],
            "score": round(float(proba[i]), 5),
            "shap_top_features": top_features_all[i],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {out_path} ({len(records)}건)")


if __name__ == "__main__":
    main()
