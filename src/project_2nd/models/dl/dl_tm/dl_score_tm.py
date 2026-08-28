# -*- coding: utf-8 -*-
"""
train_dnn.py로 학습된 모델을 로드해서, 최신 스냅샷 기준 활성 매장 전체에 대해
폐업 위험 점수(score)를 계산하고 `predictions` 테이블 컬럼에 맞춰 CSV로 저장.

query_type='existing_store'만 다룸 (기존점주 대상 배치 스코어링).
query_type='new_location'(예비창업자가 임의 좌표+업종을 입력하는 경우)은 이 배치 스크립트로는
안 됨 — 그 좌표에 대해 build_spatial_features.py / build_population_features.py와 동일한
피처 조인을 실시간으로 새로 계산해야 하는 앱 서빙 로직이라 별도 구현이 필요함. 이 스크립트는
학습 데이터에 이미 존재하는 매장들의 "현재 상태 기준 미래 위험도"만 배치로 계산함.

실행 (프로젝트 루트에서):
    python models/dl/dl_tm/dl_score_tm.py
    python models/dl/dl_tm/dl_score_tm.py --model-id dnn_mlp_v1 \
        --data data/processed/modeling_dataset_preprocessed_pmh.csv \
        --artifact-dir models/dl/saved \
        --out data/features/predictions_raw.csv

주의: predictions.model_id는 models 테이블을 FK로 참조함.
      DB에 올리기 전에 write_model.py 등으로 models 테이블에 model_id 행을
      먼저 등록해둬야 함 (여기서는 --model-id 문자열 값만 CSV에 채워둠).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# train_dnn.py를 같은 폴더(또는 PYTHONPATH)에서 그대로 import해서 재사용
from dl_train_tm import ClosureMLP, CONT_COLS, CAT_COLS, ID_COLS


def load_artifacts(artifact_dir: Path):
    with open(artifact_dir / "scaler.json", encoding="utf-8") as f:
        scaler = json.load(f)
    with open(artifact_dir / "feature_config.json", encoding="utf-8") as f:
        fconfig = json.load(f)

    model = ClosureMLP(
        n_cont=len(fconfig["cont_cols"]),
        cat_cards=fconfig["cat_cards"],
        emb_dims=fconfig["emb_dims"],
    )
    state = torch.load(artifact_dir / "model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    cont_mean = np.array(scaler["mean"], dtype=np.float32)
    cont_std = np.array(scaler["std"], dtype=np.float32)
    return model, cont_mean, cont_std, fconfig


def predict_scores(model, df, cont_mean, cont_std, calib_a=1.0, calib_b=0.0, batch_size=8192):
    """df는 CONT_COLS + CAT_COLS 컬럼을 전부 갖고 있어야 함.

    calib_a/calib_b: train_dnn.py가 fold4(held-out)로 적합한 Platt scaling 파라미터.
    pos_weight로 학습한 모델의 raw sigmoid(logit)은 실제 확률과 크게 어긋나므로
    반드시 calibrated_prob = sigmoid(a*logit + b)로 변환해서 써야 함.
    feature_config.json에 없으면(구버전 아티팩트) 기본값 a=1, b=0으로 무보정 처리.
    """
    n = len(df)
    x_cont_all = (df[CONT_COLS].to_numpy(dtype=np.float32) - cont_mean) / cont_std
    x_cat_all = df[CAT_COLS].to_numpy(dtype=np.int64)
    scores = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bc = torch.from_numpy(x_cont_all[i:i + batch_size])
            bcat = torch.from_numpy(x_cat_all[i:i + batch_size])
            logits = model(bc, bcat).numpy()
            calibrated_logits = calib_a * logits + calib_b
            scores[i:i + batch_size] = 1.0 / (1.0 + np.exp(-calibrated_logits))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features/modeling_dataset_preprocessed_pmh.csv")
    ap.add_argument("--artifact-dir", default="models/dl/saved")
    ap.add_argument("--out", default="data/features/predictions_raw.csv")
    ap.add_argument("--model-id", default="dnn_mlp_v1",
                     help="models 테이블에 미리 등록해둘 model_id 값")
    ap.add_argument("--snapshot", type=int, default=None,
                     help="특정 snapshot_date로 스코어링 (기본: 데이터 내 최신 스냅샷)")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    model, cont_mean, cont_std, fconfig = load_artifacts(artifact_dir)
    test_metrics = fconfig.get("phaseA_test_metrics_fold4", {})
    print("모델 로드 완료. 학습 시 fold4 test 성능:", test_metrics)

    calib = test_metrics.get("calibration_params", {"a": 1.0, "b": 0.0})
    calib_a, calib_b = calib.get("a", 1.0), calib.get("b", 0.0)
    print(f"Platt scaling 적용: prob = sigmoid({calib_a:.4f} * logit + {calib_b:.4f})")

    usecols = CONT_COLS + CAT_COLS + ID_COLS
    dtype_map = {c: "float32" for c in CONT_COLS}
    dtype_map.update({c: "int32" for c in CAT_COLS})
    df = pd.read_csv(args.data, encoding="utf-8-sig", usecols=usecols, dtype=dtype_map)

    target_snapshot = args.snapshot or df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == target_snapshot].reset_index(drop=True)
    print(f"스코어링 대상 snapshot={target_snapshot}: {len(latest):,} 매장")

    scores = predict_scores(model, latest, cont_mean, cont_std, calib_a=calib_a, calib_b=calib_b)
    latest["score"] = np.round(scores, 5)  # predictions.score DECIMAL(6,5)

    out_df = pd.DataFrame({
        "model_id": args.model_id,
        "user_id": None,                 # 익명 배치 스코어링이므로 NULL
        "query_type": "existing_store",
        "store_id": latest["store_id"],
        "query_lat": None,               # existing_store일 땐 NULL
        "query_lng": None,
        "industry_code": latest["industry_code"],
        "score": latest["score"],
        "shap_top_features": None,       # shap_explain_tm.py가 채워서 최종 CSV로 다시 저장
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_path}  ({len(out_df):,} rows)")
    print(out_df["score"].describe())
    print("\n다음 단계: ../../shap/shap_explain_tm.py로 shap_top_features 채우기")


if __name__ == "__main__":
    main()
