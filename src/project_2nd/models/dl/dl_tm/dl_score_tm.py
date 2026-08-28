# -*- coding: utf-8 -*-
"""
dl_train_tm.py가 만든 fold_0~fold_4 앙상블을 로드해서, 최신 스냅샷 기준 활성 매장
전체에 대해 폐업 위험 점수(score)를 계산하고 `predictions` 테이블 컬럼에 맞춰 CSV로 저장.

핵심: 매장마다 fold_of(store_id)를 다시 계산해서, "그 매장을 학습에 전혀 쓰지 않은
fold 모델"로만 스코어링함. 5개 fold가 서로 다른 20%씩을 held-out 하므로 전체 매장이
정확히 하나씩의 진짜 out-of-fold 모델로 커버됨 (데이터 누수 0).

query_type='existing_store'만 다룸 (기존점주 대상 배치 스코어링).
query_type='new_location'(예비창업자가 임의 좌표+업종을 입력하는 경우)은 이 배치 스크립트로는
안 됨 — 그 좌표에 대해 build_spatial_features.py / build_population_features.py와 동일한
피처 조인을 실시간으로 새로 계산해야 하는 앱 서빙 로직이라 별도 구현이 필요함.

실행 (프로젝트 루트에서):
    python models/dl/dl_tm/dl_score_tm.py
    python models/dl/dl_tm/dl_score_tm.py --model-id dnn_mlp_v2 \
        --data data/processed/modeling_dataset_preprocessed_pmh.csv \
        --artifact-dir src/project_2nd/models/dl/saved \
        --out data/features/predictions_raw.json

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

# dl_train_tm.py를 같은 폴더에서 그대로 import해서 재사용
from dl_train_tm import ClosureMLP, CONT_COLS, CAT_COLS, ID_COLS, transform_cont, fold_of, N_FOLDS


def load_ensemble(artifact_dir: Path, n_folds: int = N_FOLDS):
    """fold_0/ ~ fold_{n_folds-1}/ 를 전부 로드해서 {fold_id: (model, cont_mean, cont_std, calib_a, calib_b)} 반환."""
    with open(artifact_dir / "ensemble_config.json", encoding="utf-8") as f:
        econfig = json.load(f)

    folds = {}
    for k in range(n_folds):
        fold_dir = artifact_dir / f"fold_{k}"
        with open(fold_dir / "scaler.json", encoding="utf-8") as f:
            scaler = json.load(f)
        with open(fold_dir / "calibration.json", encoding="utf-8") as f:
            calib_info = json.load(f)

        model = ClosureMLP(
            n_cont=len(econfig["cont_cols"]),
            cat_cards=econfig["cat_cards"],
            emb_dims=econfig["emb_dims"],
        )
        state = torch.load(fold_dir / "model_state.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        calib = calib_info.get("calibration_params", {"a": 1.0, "b": 0.0})
        folds[k] = {
            "model": model,
            "cont_mean": np.array(scaler["mean"], dtype=np.float32),
            "cont_std": np.array(scaler["std"], dtype=np.float32),
            "calib_a": calib.get("a", 1.0),
            "calib_b": calib.get("b", 0.0),
        }
    return folds, econfig


def predict_scores_for_fold(fold_artifacts, df_sub, batch_size=8192):
    """df_sub: 이 fold 모델이 스코어링을 담당하는 매장들(=이 fold를 학습에서 제외한 모델)."""
    model = fold_artifacts["model"]
    cont_mean, cont_std = fold_artifacts["cont_mean"], fold_artifacts["cont_std"]
    calib_a, calib_b = fold_artifacts["calib_a"], fold_artifacts["calib_b"]

    n = len(df_sub)
    x_cont_all = (transform_cont(df_sub[CONT_COLS].to_numpy(dtype=np.float32)) - cont_mean) / cont_std
    x_cat_all = df_sub[CAT_COLS].to_numpy(dtype=np.int64)
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
    ap.add_argument("--data", default="data/processed/modeling_dataset_preprocessed_pmh.csv")
    ap.add_argument("--artifact-dir", default="src/project_2nd/models/dl/saved")
    ap.add_argument("--out", default="data/features/predictions_raw.json")
    ap.add_argument("--model-id", default="dnn_mlp_v2",
                     help="models 테이블에 미리 등록해둘 model_id 값")
    ap.add_argument("--snapshot", type=int, default=None,
                     help="특정 snapshot_date로 스코어링 (기본: 데이터 내 최신 스냅샷)")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS,
                     help="dl_train_tm.py --n-folds와 동일하게 맞출 것")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    folds, econfig = load_ensemble(artifact_dir, n_folds=args.n_folds)
    print(f"{args.n_folds}-fold 앙상블 로드 완료.")
    print("5-fold CV 요약 성능:", econfig.get("cv_summary"))

    usecols = CONT_COLS + CAT_COLS + ID_COLS
    dtype_map = {c: "float32" for c in CONT_COLS}
    dtype_map.update({c: "int32" for c in CAT_COLS})
    df = pd.read_csv(args.data, encoding="utf-8-sig", usecols=usecols, dtype=dtype_map)

    target_snapshot = args.snapshot or df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == target_snapshot].reset_index(drop=True)
    print(f"스코어링 대상 snapshot={target_snapshot}: {len(latest):,} 매장")

    # 매장마다 fold_of(store_id)로 "이 매장을 안 본 모델"을 결정
    latest["_fold"] = latest["store_id"].apply(lambda sid: fold_of(sid, k=N_FOLDS))
    latest["score"] = np.nan

    for k in sorted(latest["_fold"].unique()):
        if k not in folds:
            print(f"⚠ fold {k}에 해당하는 모델이 없음(--n-folds가 학습 시보다 작았을 가능성) — 건너뜀")
            continue
        mask = latest["_fold"] == k
        sub = latest.loc[mask]
        scores_k = predict_scores_for_fold(folds[k], sub)
        latest.loc[mask, "score"] = scores_k
        print(f"  fold {k} 모델로 {mask.sum():,}개 매장 스코어링 완료")

    latest["score"] = np.round(latest["score"], 5)  # predictions.score DECIMAL(6,5)

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
    out_df = out_df.astype(object).where(pd.notnull(out_df), None)  # NaN -> JSON null (유효한 JSON 보장)
    records = out_df.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\n저장 완료: {out_path}  ({len(out_df):,} rows)")
    print(latest["score"].describe())
    print("\n다음 단계: ../../shap/shap_explain_tm.py로 shap_top_features 채우기")


if __name__ == "__main__":
    main()
