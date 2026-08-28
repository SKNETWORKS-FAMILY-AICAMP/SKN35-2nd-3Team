# -*- coding: utf-8 -*-
"""
models/dl/dl_tm/dl_score_tm.py가 만든 predictions_raw.csv에 shap_top_features(JSON)를 채워
DB `predictions` 테이블에 그대로 적재 가능한 최종 CSV를 만든다.

MLP는 트리 모델과 달리 TreeExplainer를 못 쓰므로 KernelExplainer + 샘플링 사용
(model-agnostic, 원본 피처(스케일링 전) 공간에서 바로 설명 가능해서
 shap_top_features의 feature_value를 사람이 보는 실제 값으로 채울 수 있음).

주의 (중요, 실행 전에 꼭 읽을 것):
    KernelExplainer는 인스턴스 1건당 배경표본 기반 퍼뮤테이션을 nsamples번 돌리고
    회귀로 기여도를 추정하는 방식이라, 트리 모델의 TreeExplainer보다 훨씬 느림.
    activate 매장 전체(수십만 건)를 한 번에 설명하려 하면 오래 걸릴 수 있으니,
    --sample-size로 설명 대상 건수를 조절해서 쓸 것. 기본값은 위험도 상위 --select=top_risk
    방식으로 가장 중요한(=UI에 노출될 가능성이 높은) 매장부터 채우도록 되어 있음.

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
from dl_train_tm import ClosureMLP, CONT_COLS, CAT_COLS, ID_COLS

FEATURE_COLS = CONT_COLS + CAT_COLS  # predict_fn/shap이 다루는 원본 피처 순서


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


def make_predict_fn(model, cont_mean, cont_std, n_cont, calib_a=1.0, calib_b=0.0):
    """KernelExplainer가 호출할 함수. 입력 X는 (n, n_cont+n_cat) 원본(스케일 전) 값.

    dl_score_tm.py와 동일하게 Platt scaling(calib_a/b)을 적용해서, SHAP이 설명하는 값이
    실제로 DB predictions.score에 저장되는 값(calibrated prob)과 일치하도록 함.
    """
    def predict_fn(X):
        X = np.asarray(X, dtype=np.float32)
        x_cont = (X[:, :n_cont] - cont_mean) / cont_std
        x_cat = np.rint(X[:, n_cont:]).astype(np.int64)  # 배경표본의 실제 카테고리 값이라 반올림해도 안전
        with torch.no_grad():
            logits = model(torch.from_numpy(x_cont), torch.from_numpy(x_cat)).numpy()
        calibrated_logits = calib_a * logits + calib_b
        return 1.0 / (1.0 + np.exp(-calibrated_logits))
    return predict_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features/modeling_dataset_preprocessed_pmh.csv")
    ap.add_argument("--artifact-dir", default="models/dl/saved")
    ap.add_argument("--predictions-csv", default="data/features/predictions_raw.csv",
                     help="models/dl/dl_tm/dl_score_tm.py 산출물")
    ap.add_argument("--out", default="data/features/predictions_for_db.csv",
                     help="shap_top_features까지 채운 최종 DB 적재용 CSV")
    ap.add_argument("--snapshot", type=int, default=None,
                     help="dl_score_tm.py와 동일한 snapshot으로 맞출 것 (기본: 최신)")
    ap.add_argument("--sample-size", type=int, default=2000,
                     help="SHAP 설명을 계산할 매장 수 (전체 매장 대비 샘플링)")
    ap.add_argument("--select", choices=["top_risk", "random"], default="top_risk",
                     help="top_risk: score 상위 매장부터 / random: 무작위 샘플")
    ap.add_argument("--background-size", type=int, default=50,
                     help="KernelExplainer 배경표본 크기 (클수록 정확하지만 느려짐)")
    ap.add_argument("--nsamples", default="auto",
                     help="KernelExplainer 인스턴스당 퍼뮤테이션 수 ('auto' 또는 정수)")
    ap.add_argument("--top-k", type=int, default=5, help="매장당 상위 몇 개 피처를 JSON에 담을지")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)

    artifact_dir = Path(args.artifact_dir)
    model, cont_mean, cont_std, fconfig = load_artifacts(artifact_dir)
    n_cont = len(CONT_COLS)

    calib = fconfig.get("phaseA_test_metrics_fold4", {}).get("calibration_params", {"a": 1.0, "b": 0.0})
    calib_a, calib_b = calib.get("a", 1.0), calib.get("b", 0.0)
    print(f"Platt scaling 적용(dl_score_tm.py와 동일): prob = sigmoid({calib_a:.4f} * logit + {calib_b:.4f})")

    usecols = CONT_COLS + CAT_COLS + ID_COLS
    dtype_map = {c: "float32" for c in CONT_COLS}
    dtype_map.update({c: "int32" for c in CAT_COLS})
    df = pd.read_csv(args.data, encoding="utf-8-sig", usecols=usecols, dtype=dtype_map)

    target_snapshot = args.snapshot or df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == target_snapshot].reset_index(drop=True)
    print(f"snapshot={target_snapshot}: 후보 매장 {len(latest):,}건")

    preds = pd.read_csv(args.predictions_csv, encoding="utf-8-sig")
    latest = latest.merge(preds[["store_id", "score"]], on="store_id", how="inner")
    print(f"predictions_raw.csv와 매칭된 매장 {len(latest):,}건")

    # ---- 배경표본: 전체에서 무작위 추출 후 kmeans로 요약(속도 향상) ----
    bg_idx = np.random.choice(len(latest), size=min(args.background_size * 4, len(latest)), replace=False)
    bg_raw = latest.loc[bg_idx, FEATURE_COLS].to_numpy(dtype=np.float32)
    background = shap.kmeans(bg_raw, min(args.background_size, len(bg_raw)))

    predict_fn = make_predict_fn(model, cont_mean, cont_std, n_cont, calib_a=calib_a, calib_b=calib_b)
    explainer = shap.KernelExplainer(predict_fn, background)

    # ---- 설명 대상 선택 ----
    n_explain = min(args.sample_size, len(latest))
    if args.select == "top_risk":
        explain_idx = latest["score"].to_numpy().argsort()[::-1][:n_explain]
    else:
        explain_idx = np.random.choice(len(latest), size=n_explain, replace=False)
    explain_df = latest.iloc[explain_idx].reset_index(drop=True)
    X_explain = explain_df[FEATURE_COLS].to_numpy(dtype=np.float32)

    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)
    print(f"SHAP 계산 시작: {n_explain:,}건 x nsamples={nsamples} (배경표본 {background.data.shape[0]}개)")
    print("※ 인스턴스 수가 많으면 오래 걸릴 수 있음 — 진행 중 상태 로그로 대략적인 속도 가늠 가능")

    shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=False)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # 일부 shap 버전은 (n, features, 1) 형태로 반환
        shap_values = shap_values[:, :, 0]

    # ---- 매장별 top-k JSON 생성 ----
    shap_json_list = []
    for i in range(n_explain):
        row_shap = shap_values[i]
        row_vals = X_explain[i]
        top_idx = np.argsort(-np.abs(row_shap))[: args.top_k]
        items = [
            {
                "feature": FEATURE_COLS[j],
                "shap_value": round(float(row_shap[j]), 5),
                "feature_value": round(float(row_vals[j]), 4),
            }
            for j in top_idx
        ]
        shap_json_list.append(json.dumps(items, ensure_ascii=False))

    explain_df["shap_top_features"] = shap_json_list

    # ---- predictions_raw.csv 최종본과 merge ----
    preds = preds.drop(columns=["shap_top_features"])
    final = preds.merge(
        explain_df[["store_id", "shap_top_features"]], on="store_id", how="left"
    )
    # 컬럼 순서를 predictions 테이블과 동일하게 정렬
    final = final[["model_id", "user_id", "query_type", "store_id", "query_lat",
                   "query_lng", "industry_code", "score", "shap_top_features"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_filled = final["shap_top_features"].notna().sum()
    print(f"\n저장 완료: {out_path}")
    print(f"전체 {len(final):,} rows 중 shap_top_features 채워진 행: {n_filled:,}건 "
          f"(나머지는 NULL로 남음 — DB 컬럼이 NULL 허용이라 그대로 적재 가능)")
    print("나머지 매장까지 채우려면 --sample-size를 늘리거나 --select random으로 여러 번 나눠 돌린 뒤")
    print("현재 스크립트의 merge 로직을 '이미 채워진 행은 건너뛰기'로 바꿔서 이어서 실행하면 됨.")


if __name__ == "__main__":
    main()
