# -*- coding: utf-8 -*-
"""
models/dl/dl_tm/dl_score_tm.py가 만든 predictions_raw.json에 shap_top_features(JSON)를 채워
DB `predictions` 테이블에 그대로 적재 가능한 최종 CSV를 만든다.

dl_score_tm.py와 동일한 원칙: 매장마다 fold_of(store_id)로 "그 매장을 학습에 안 쓴
fold 모델"을 찾아서, 그 모델 기준으로 SHAP을 계산함 (배경표본도 같은 fold 데이터에서
뽑음 -> 설명 모델과 배경표본의 분포가 일치해야 KernelExplainer 결과가 의미 있음).

MLP는 트리 모델과 달리 TreeExplainer를 못 쓰므로 KernelExplainer + 샘플링 사용
(model-agnostic, 원본 피처(스케일링 전) 공간에서 바로 설명 가능해서
 shap_top_features의 feature_value를 사람이 보는 실제 값으로 채울 수 있음).

주의 (중요, 실행 전에 꼭 읽을 것):
    KernelExplainer는 인스턴스 1건당 배경표본 기반 퍼뮤테이션을 nsamples번 돌리고
    회귀로 기여도를 추정하는 방식이라, 트리 모델의 TreeExplainer보다 훨씬 느림.
    활성 매장 전체(수십만 건)를 한 번에 설명하려 하면 오래 걸릴 수 있으니,
    --sample-size로 설명 대상 건수를 조절해서 쓸 것(fold별로 분배됨). 기본값은
    위험도 상위 --select=top_risk 방식으로 가장 중요한 매장부터 채우도록 되어 있음.

    전체 매장을 다 커버해야 하면:
      1) --sample-size를 늘려가며 여러 번 실행(누적 append) 하거나
      2) --select random으로 청크를 나눠 여러 프로세스로 병렬 실행하거나
      3) (장기적으로) DeepExplainer/GradientExplainer로 바꾸면 훨씬 빠름 —
         PyTorch 모델이라 shap.DeepExplainer(model, background_tensor)로 바로 대체 가능.
         지금은 계획대로 KernelExplainer로 구현해둠.

실행 (프로젝트 루트에서):
    python models/shap/shap_explain_tm.py
    python models/shap/shap_explain_tm.py --sample-size 5000 --select top_risk --top-k 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

# dl_train_tm.py(ClosureMLP 등 정의)는 models/dl/dl_tm/에 있으므로 sys.path에 추가해서 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dl" / "dl_tm"))
from dl_train_tm import ClosureMLP, CONT_COLS, CAT_COLS, ID_COLS, transform_cont, fold_of, N_FOLDS

FEATURE_COLS = CONT_COLS + CAT_COLS  # predict_fn/shap이 다루는 원본 피처 순서


def load_fold_artifacts(artifact_dir: Path, k: int, econfig: dict):
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
    return {
        "model": model,
        "cont_mean": np.array(scaler["mean"], dtype=np.float32),
        "cont_std": np.array(scaler["std"], dtype=np.float32),
        "calib_a": calib.get("a", 1.0),
        "calib_b": calib.get("b", 0.0),
    }


def make_predict_fn(model, cont_mean, cont_std, n_cont, calib_a=1.0, calib_b=0.0):
    """KernelExplainer가 호출할 함수. 입력 X는 (n, n_cont+n_cat) 원본(스케일 전) 값.

    dl_score_tm.py와 동일하게 Platt scaling(calib_a/b)을 적용해서, SHAP이 설명하는 값이
    실제로 DB predictions.score에 저장되는 값(calibrated prob)과 일치하도록 함.
    """
    def predict_fn(X):
        X = np.asarray(X, dtype=np.float32)
        x_cont = (transform_cont(X[:, :n_cont]) - cont_mean) / cont_std
        x_cat = np.rint(X[:, n_cont:]).astype(np.int64)  # 배경표본의 실제 카테고리 값이라 반올림해도 안전
        with torch.no_grad():
            logits = model(torch.from_numpy(x_cont), torch.from_numpy(x_cat)).numpy()
        calibrated_logits = calib_a * logits + calib_b
        return 1.0 / (1.0 + np.exp(-calibrated_logits))
    return predict_fn


def explain_fold_subset(fold_artifacts, fold_pool_df, explain_df, n_cont,
                         background_size, nsamples, top_k, fold_label=""):
    """한 fold 안에서: fold_pool_df(그 fold 소속 전체 매장)로 배경표본을 뽑고,
    explain_df(그 fold 소속 중 설명 대상으로 뽑힌 매장)를 SHAP으로 설명."""
    bg_size = min(background_size * 4, len(fold_pool_df))
    bg_idx = np.random.choice(len(fold_pool_df), size=bg_size, replace=False)
    bg_raw = fold_pool_df.iloc[bg_idx][FEATURE_COLS].to_numpy(dtype=np.float32)
    background = shap.kmeans(bg_raw, min(background_size, len(bg_raw)))

    predict_fn = make_predict_fn(
        fold_artifacts["model"], fold_artifacts["cont_mean"], fold_artifacts["cont_std"],
        n_cont, calib_a=fold_artifacts["calib_a"], calib_b=fold_artifacts["calib_b"],
    )
    explainer = shap.KernelExplainer(predict_fn, background)

    X_explain = explain_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    print(f"  [{fold_label}] SHAP 계산: {len(X_explain):,}건 x nsamples={nsamples} "
          f"(배경표본 {background.data.shape[0]}개)")
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=False)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # 일부 shap 버전은 (n, features, 1) 형태로 반환
        shap_values = shap_values[:, :, 0]

    shap_items_list = []
    for i in range(len(X_explain)):
        row_shap = shap_values[i]
        row_vals = X_explain[i]
        top_idx = np.argsort(-np.abs(row_shap))[:top_k]
        items = [
            {
                "feature": FEATURE_COLS[j],
                "shap_value": round(float(row_shap[j]), 5),
                "feature_value": round(float(row_vals[j]), 4),
            }
            for j in top_idx
        ]
        shap_items_list.append(items)  # JSON 파일에 그대로 중첩 객체로 들어가도록 문자열화 안 함
    return shap_items_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/modeling_dataset_preprocessed_pmh.csv")
    ap.add_argument("--artifact-dir", default="src/project_2nd/models/dl/saved")
    ap.add_argument("--predictions-json", default="data/features/predictions_raw.json",
                     help="models/dl/dl_tm/dl_score_tm.py 산출물")
    ap.add_argument("--out", default="data/features/predictions_for_db.json",
                     help="shap_top_features까지 채운 최종 DB 적재용 JSON")
    ap.add_argument("--snapshot", type=int, default=None,
                     help="dl_score_tm.py와 동일한 snapshot으로 맞출 것 (기본: 최신)")
    ap.add_argument("--sample-size", type=int, default=2000,
                     help="SHAP 설명을 계산할 매장 수 (전체 대비 샘플, fold별로 비례 분배)")
    ap.add_argument("--select", choices=["top_risk", "random"], default="top_risk",
                     help="top_risk: score 상위 매장부터 / random: 무작위 샘플 (fold별로 각각 적용)")
    ap.add_argument("--background-size", type=int, default=50,
                     help="KernelExplainer 배경표본 크기 (클수록 정확하지만 느려짐)")
    ap.add_argument("--nsamples", default="auto",
                     help="KernelExplainer 인스턴스당 퍼뮤테이션 수 ('auto' 또는 정수)")
    ap.add_argument("--top-k", type=int, default=5, help="매장당 상위 몇 개 피처를 JSON에 담을지")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS,
                     help="dl_train_tm.py --n-folds와 동일하게 맞출 것")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    artifact_dir = Path(args.artifact_dir)
    with open(artifact_dir / "ensemble_config.json", encoding="utf-8") as f:
        econfig = json.load(f)
    n_cont = len(CONT_COLS)
    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)

    usecols = CONT_COLS + CAT_COLS + ID_COLS
    dtype_map = {c: "float32" for c in CONT_COLS}
    dtype_map.update({c: "int32" for c in CAT_COLS})
    df = pd.read_csv(args.data, encoding="utf-8-sig", usecols=usecols, dtype=dtype_map)

    target_snapshot = args.snapshot or df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == target_snapshot].reset_index(drop=True)
    print(f"snapshot={target_snapshot}: 후보 매장 {len(latest):,}건")

    preds_records = json.load(open(args.predictions_json, encoding="utf-8"))
    preds = pd.DataFrame(preds_records)
    latest = latest.merge(preds[["store_id", "score"]], on="store_id", how="inner")
    latest["_fold"] = latest["store_id"].apply(lambda sid: fold_of(sid, k=N_FOLDS))
    print(f"predictions_raw.json과 매칭된 매장 {len(latest):,}건")

    all_explain_frames = []
    for k in range(args.n_folds):
        fold_pool = latest[latest["_fold"] == k].reset_index(drop=True)
        if len(fold_pool) == 0:
            continue

        # 이 fold 몫만큼 설명 대상 선정 (fold별 비중에 비례해서 --sample-size를 나눔)
        n_explain_k = max(1, round(args.sample_size * len(fold_pool) / len(latest)))
        n_explain_k = min(n_explain_k, len(fold_pool))
        if args.select == "top_risk":
            idx = fold_pool["score"].to_numpy().argsort()[::-1][:n_explain_k]
        else:
            idx = np.random.choice(len(fold_pool), size=n_explain_k, replace=False)
        explain_df = fold_pool.iloc[idx].reset_index(drop=True)

        print(f"\n[Fold {k}] 매장 {len(fold_pool):,}건 중 {len(explain_df):,}건 설명 대상 선정")
        fold_artifacts = load_fold_artifacts(artifact_dir, k, econfig)
        shap_items_list = explain_fold_subset(
            fold_artifacts, fold_pool, explain_df, n_cont,
            args.background_size, nsamples, args.top_k, fold_label=f"Fold {k}",
        )
        explain_df["shap_top_features"] = shap_items_list
        all_explain_frames.append(explain_df[["store_id", "shap_top_features"]])

    explained_all = pd.concat(all_explain_frames, ignore_index=True) if all_explain_frames else \
        pd.DataFrame(columns=["store_id", "shap_top_features"])

    # ---- predictions_raw.json 최종본과 merge ----
    preds = preds.drop(columns=["shap_top_features"])
    final = preds.merge(explained_all, on="store_id", how="left")
    # 컬럼 순서를 predictions 테이블과 동일하게 정렬
    final = final[["model_id", "user_id", "query_type", "store_id", "query_lat",
                   "query_lng", "industry_code", "score", "shap_top_features"]]

    # NaN(매칭 안 된 shap_top_features, user_id/query_lat/query_lng 등) -> JSON null로
    final = final.astype(object).where(pd.notnull(final), None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = final.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    n_filled = sum(1 for r in records if r["shap_top_features"] is not None)
    print(f"\n저장 완료: {out_path}")
    print(f"전체 {len(records):,} rows 중 shap_top_features 채워진 행: {n_filled:,}건 "
          f"(나머지는 null로 남음 — DB 컬럼이 NULL 허용이라 그대로 적재 가능)")
    print("나머지 매장까지 채우려면 --sample-size를 늘리거나 --select random으로 여러 번 나눠 돌린 뒤")
    print("현재 스크립트의 merge 로직을 '이미 채워진 행은 건너뛰기'로 바꿔서 이어서 실행하면 됨.")


if __name__ == "__main__":
    main()
