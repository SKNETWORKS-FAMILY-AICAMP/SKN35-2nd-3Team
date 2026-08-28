# -*- coding: utf-8 -*-
"""
서울 상권 폐업 위험 예측 - DNN(MLP) 학습 스크립트

실행 (프로젝트 루트 C:\\SKN35-2nd-3Team 에서):
    python models/dl/dl_tm/dl_train_tm.py
    python models/dl/dl_tm/dl_train_tm.py --data data/processed/modeling_dataset_preprocessed_pmh.csv

아키텍처
    범주형 5개 임베딩 + 연속형 19개 피처 concat
    -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
    -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.3)
    -> Linear(32)  -> BatchNorm -> ReLU -> Dropout(0.2)
    -> Linear(1)   (logit, 추론 시 sigmoid)
    Loss: BCEWithLogitsLoss(pos_weight=neg/pos) — 폐업률 ~10.6% 불균형 보정

검증 전략
    modeling_dataset의 기존 `fold` 컬럼(store_id 해시 기반 GroupKFold, K=5)을 그대로 사용해
    LightGBM 베이스라인과 동일 기준으로 비교 가능하게 함.
      Phase A : fold 0,1,2 = train / fold 3 = early-stopping val / fold 4 = held-out test
                -> ROC-AUC / PR-AUC / 상위 5% Lift 산출 (베이스라인: ROC-AUC 0.721~0.728, PR-AUC 0.300~0.317)
      Phase B : Phase A에서 찾은 best_epoch만큼 fold 0~4 전체로 재학습 -> 실서빙용 프로덕션 모델

산출물 (--artifact-dir, 기본 models/dl/saved/)
    model_state.pt        state_dict (ClosureMLP 정의는 이 파일 것을 그대로 재사용)
    scaler.json            연속형 피처 표준화 파라미터 (전체 데이터 기준, Phase B)
    feature_config.json   피처 목록/순서, 카테고리 cardinality, 임베딩 차원, 성능 지표
    (dl_score_tm.py / ../../shap/shap_explain_tm.py가 이 3개 파일을 그대로 로드해서 사용함)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 피처 정의 (predict/shap 스크립트에서도 그대로 import해서 씀)
# ---------------------------------------------------------------------------
CONT_COLS = [
    "lng", "lat",
    "same_industry_count_300m", "total_count_300m",
    "nearest_same_industry_distance_m", "dong_industry_count",
    "coord_cluster_size", "store_age_months", "previously_transitioned",
    "keyword_growth_score",
    "korean_pop", "foreign_long_pop", "foreign_short_pop", "total_pop_avg",
    "foreign_short_ratio", "tourist_zone_candidate", "population_is_proxied",
    "industry_historical_rate", "dong_historical_rate", "dong_industry_historical_rate",
]

# industry_group_enc/jung_code_enc/jung_name_enc/industry_name_enc는
# industry_dae_code_enc / industry_code_enc와 1:1 중복(nunique 동일)이라 제외했음.
# (확인: df[col].nunique()가 각 쌍끼리 정확히 같음 -> 별도 정보 없음)
CAT_COLS = ["industry_dae_code_enc", "industry_code_enc", "gu_name_enc",
            "dong_code_enc", "floor_category_enc"]
EMB_DIMS = [4, 24, 8, 30, 3]  # cardinality 대비 경험적 heuristic (min(50,(card+1)//2) 근사)

TARGET_COL = "is_closed_next"
FOLD_COL = "fold"
ID_COLS = ["store_id", "industry_code", "snapshot_date"]  # 학습엔 미사용, predict 단계에서 필요

USECOLS = CONT_COLS + CAT_COLS + [TARGET_COL, FOLD_COL] + ID_COLS


class ClosureMLP(nn.Module):
    """predict/shap 스크립트에서도 동일하게 import해서 state_dict를 로드함."""

    def __init__(self, n_cont, cat_cards, emb_dims):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cat_cards, emb_dims)])
        in_dim = n_cont + sum(emb_dims)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_cont, x_cats):
        embs = [e(x_cats[:, i]) for i, e in enumerate(self.embs)]
        x = torch.cat([x_cont] + embs, dim=1)
        return self.net(x).squeeze(1)


def load_data(data_path):
    dtype_map = {c: "float32" for c in CONT_COLS}
    dtype_map.update({c: "int32" for c in CAT_COLS})
    dtype_map[TARGET_COL] = "int8"
    dtype_map[FOLD_COL] = "int8"
    df = pd.read_csv(data_path, encoding="utf-8-sig", usecols=USECOLS, dtype=dtype_map)
    return df


def to_tensors(df, cont_mean, cont_std, device):
    x_cont = (df[CONT_COLS].to_numpy(dtype=np.float32) - cont_mean) / cont_std
    x_cats = df[CAT_COLS].to_numpy(dtype=np.int64)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    return (torch.from_numpy(x_cont).to(device),
            torch.from_numpy(x_cats).to(device),
            torch.from_numpy(y).to(device))


def iterate_batches(n, batch_size, shuffle=True):
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def run_epoch(model, opt, lossfn, x_cont, x_cats, y, batch_size, train=True,
              epoch_num=None, total_epochs=None):
    model.train(train)
    total_loss = 0.0
    n = x_cont.shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    all_probs = [] if not train else None

    phase_label = "train" if train else "val  "
    epoch_tag = f"epoch {epoch_num:02d}/{total_epochs}" if epoch_num else ""
    pbar = tqdm(iterate_batches(n, batch_size, shuffle=train), total=n_batches,
                desc=f"{epoch_tag} [{phase_label}]", leave=False, ncols=100)

    seen = 0
    for idx in pbar:
        idx_t = torch.from_numpy(idx).to(x_cont.device)
        bc, bcat, by = x_cont[idx_t], x_cats[idx_t], y[idx_t]
        if train:
            opt.zero_grad()
            out = model(bc, bcat)
            loss = lossfn(out, by)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                out = model(bc, bcat)
                loss = lossfn(out, by)
                all_probs.append(torch.sigmoid(out).cpu().numpy())
        total_loss += loss.item() * len(idx)
        seen += len(idx)
        pbar.set_postfix(loss=f"{total_loss/seen:.4f}")
    avg_loss = total_loss / n
    probs = np.concatenate(all_probs) if all_probs is not None else None
    return avg_loss, probs


def evaluate(probs, y_true, label=""):
    auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    order = np.argsort(-probs)
    top5 = order[: max(1, int(len(order) * 0.05))]
    base_rate = float(y_true.mean())
    lift = float(y_true[top5].mean() / base_rate) if base_rate > 0 else float("nan")
    print(f"[{label}] ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  Top5% Lift={lift:.2f}x  "
          f"(base_rate={base_rate:.4f}, n={len(y_true)})")
    return {"roc_auc": auc, "pr_auc": pr_auc, "top5pct_lift": lift, "base_rate": base_rate, "n": int(len(y_true))}


def make_pos_weight(y, device):
    pos = y.sum().item()
    neg = len(y) - pos
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)


def train_model(x_cont, x_cats, y, cat_cards, device, val_data=None, max_epochs=30, patience=4,
                 batch_size=4096, lr=1e-3):
    model = ClosureMLP(n_cont=x_cont.shape[1], cat_cards=cat_cards, emb_dims=EMB_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = make_pos_weight(y, device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score, best_epoch, best_state, no_improve = -1, 0, None, 0
    history = []
    stage = "Phase A" if val_data is not None else "Phase B"
    epoch_range = tqdm(range(1, max_epochs + 1), desc=f"{stage} 전체 진행률", ncols=100)
    for epoch in epoch_range:
        t0 = time.time()
        train_loss, _ = run_epoch(model, opt, lossfn, x_cont, x_cats, y, batch_size, train=True,
                                   epoch_num=epoch, total_epochs=max_epochs)
        log = f"epoch {epoch:02d}  train_loss={train_loss:.4f}  ({time.time()-t0:.1f}s)"

        if val_data is not None:
            vx_cont, vx_cats, vy = val_data
            val_loss, val_probs = run_epoch(model, opt, lossfn, vx_cont, vx_cats, vy, batch_size, train=False,
                                             epoch_num=epoch, total_epochs=max_epochs)
            vy_np = vy.cpu().numpy()
            val_auc = roc_auc_score(vy_np, val_probs)
            val_pr = average_precision_score(vy_np, val_probs)
            log += f"  val_loss={val_loss:.4f}  val_ROC-AUC={val_auc:.4f}  val_PR-AUC={val_pr:.4f}"
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                             "val_roc_auc": float(val_auc), "val_pr_auc": float(val_pr)})
            score = val_pr  # 불균형 데이터라 PR-AUC 기준 early stopping
            if score > best_score:
                best_score, best_epoch, no_improve = score, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
        tqdm.write(log)
        if val_data is not None and no_improve >= patience:
            tqdm.write(f"  -> early stopping (best epoch={best_epoch}, val_PR-AUC={best_score:.4f})")
            epoch_range.close()
            break

    if val_data is not None and best_state is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, history
    return model, max_epochs, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/modeling_dataset_preprocessed_pmh.csv",
                     help="프로젝트 루트 기준 상대경로 (기본: data/processed/modeling_dataset_preprocessed_pmh.csv)")
    ap.add_argument("--artifact-dir", default="models/dl/saved",
                     help="모델/스케일러/피처설정 저장 위치")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    print("데이터 로딩 중...")
    df = load_data(args.data)
    print(f"전체 {len(df):,} rows,  결측치 합계={df.isna().sum().sum()}")
    print("fold 분포:\n", df[FOLD_COL].value_counts().sort_index())
    print(f"is_closed_next 비율: {df[TARGET_COL].mean():.4f}")

    cat_cards = [int(df[c].max()) + 1 for c in CAT_COLS]
    print("카테고리 cardinality:", dict(zip(CAT_COLS, cat_cards)))

    # ---------------- Phase A: fold 0,1,2 train / 3 val / 4 test ----------------
    print("\n=== Phase A: fold 기반 검증 (베이스라인과 비교용) ===")
    train_df = df[df[FOLD_COL].isin([0, 1, 2])]
    val_df = df[df[FOLD_COL] == 3]
    test_df = df[df[FOLD_COL] == 4]

    cont_mean = train_df[CONT_COLS].to_numpy(dtype=np.float32).mean(axis=0)
    cont_std = train_df[CONT_COLS].to_numpy(dtype=np.float32).std(axis=0)
    cont_std[cont_std == 0] = 1.0

    xtr = to_tensors(train_df, cont_mean, cont_std, device)
    xval = to_tensors(val_df, cont_mean, cont_std, device)
    xtest = to_tensors(test_df, cont_mean, cont_std, device)

    model_a, best_epoch, history = train_model(
        xtr[0], xtr[1], xtr[2], cat_cards, device, val_data=xval,
        max_epochs=args.max_epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
    )

    print("\n--- Phase A 최종 평가 (held-out fold 4) ---")
    model_a.eval()
    with torch.no_grad():
        test_logits = model_a(xtest[0], xtest[1]).cpu().numpy()
    test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))
    test_y = xtest[2].cpu().numpy()
    print("[캘리브레이션 전]")
    test_metrics = evaluate(test_probs_raw, test_y, label="TEST(fold4, raw)")
    print(f"  raw 예측확률 평균={test_probs_raw.mean():.4f}  vs  실제 폐업률={test_y.mean():.4f}  "
          f"<- pos_weight 사용 시 보통 크게 어긋남(정상)")

    # ---------------- Platt scaling 캘리브레이션 ----------------
    # pos_weight로 불균형 보정을 하면 sigmoid(logit)의 절대값(=score)이 실제 확률과
    # 크게 어긋남(랭킹 지표 ROC-AUC/PR-AUC/Lift는 monotonic이라 영향 없음, 하지만
    # UI에서 "생존점수=(1-score)x100" 처럼 score를 확률로 직접 쓰므로 보정 필요).
    # -> fold4(모델이 한번도 학습/얼리스토핑에 쓰지 않은 진짜 held-out)의 raw logit에
    #    1차원 로지스틱 회귀(Platt scaling)를 적합해서 calibrated_prob = sigmoid(a*logit+b)로 변환.
    calib_lr = LogisticRegression(C=1e10, max_iter=1000)
    calib_lr.fit(test_logits.reshape(-1, 1), test_y)
    calib_a = float(calib_lr.coef_[0][0])
    calib_b = float(calib_lr.intercept_[0])
    test_probs_calibrated = 1.0 / (1.0 + np.exp(-(calib_a * test_logits + calib_b)))
    print("[캘리브레이션 후]")
    test_metrics_calibrated = evaluate(test_probs_calibrated, test_y, label="TEST(fold4, calibrated)")
    print(f"  calibrated 예측확률 평균={test_probs_calibrated.mean():.4f}  vs  실제 폐업률={test_y.mean():.4f}")
    test_metrics["calibrated"] = test_metrics_calibrated
    test_metrics["calibration_params"] = {"a": calib_a, "b": calib_b,
                                           "formula": "prob = sigmoid(a * raw_logit + b)"}

    # ---------------- Phase B: 전체 데이터로 프로덕션 모델 재학습 ----------------
    print(f"\n=== Phase B: 전체 데이터(fold 0~4)로 프로덕션 모델 재학습 (epochs={best_epoch}) ===")
    full_cont_mean = df[CONT_COLS].to_numpy(dtype=np.float32).mean(axis=0)
    full_cont_std = df[CONT_COLS].to_numpy(dtype=np.float32).std(axis=0)
    full_cont_std[full_cont_std == 0] = 1.0

    xall = to_tensors(df, full_cont_mean, full_cont_std, device)
    model_prod, _, _ = train_model(
        xall[0], xall[1], xall[2], cat_cards, device, val_data=None,
        max_epochs=best_epoch, patience=best_epoch,
        batch_size=args.batch_size, lr=args.lr,
    )

    # ---------------- 저장 ----------------
    torch.save(model_prod.state_dict(), artifact_dir / "model_state.pt")

    scaler = {"cont_cols": CONT_COLS, "mean": full_cont_mean.tolist(), "std": full_cont_std.tolist()}
    with open(artifact_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)

    feature_config = {
        "cont_cols": CONT_COLS,
        "cat_cols": CAT_COLS,
        "cat_cards": cat_cards,
        "emb_dims": EMB_DIMS,
        "target_col": TARGET_COL,
        "trained_epochs": best_epoch,
        "phaseA_test_metrics_fold4": test_metrics,
        "phaseA_history": history,
        "model_arch": "emb -> concat -> 128-BN-ReLU-Drop(0.3) -> 64-BN-ReLU-Drop(0.3) -> 32-BN-ReLU-Drop(0.2) -> 1(logit)",
        "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)",
    }
    with open(artifact_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료 -> {artifact_dir}/model_state.pt, scaler.json, feature_config.json")
    print("참고 - LightGBM 베이스라인: ROC-AUC 0.721~0.728, PR-AUC 0.300~0.317")
    print(f"DNN(fold4 test) : ROC-AUC {test_metrics['roc_auc']:.4f}, PR-AUC {test_metrics['pr_auc']:.4f}, "
          f"Top5% Lift {test_metrics['top5pct_lift']:.2f}x")


if __name__ == "__main__":
    main()
