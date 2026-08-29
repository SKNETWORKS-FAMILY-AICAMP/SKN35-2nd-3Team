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

산출물 (--artifact-dir, 기본 src/project_2nd/models/dl/saved/)
    model_state.pt        state_dict (ClosureMLP 정의는 이 파일 것을 그대로 재사용)
    scaler.json            연속형 피처 표준화 파라미터 (전체 데이터 기준, Phase B)
    feature_config.json   피처 목록/순서, 카테고리 cardinality, 임베딩 차원, 성능 지표
    (dl_score_tm.py / ../../shap/shap_explain_tm.py가 이 3개 파일을 그대로 로드해서 사용함)
"""
import argparse
import json
import time
import hashlib
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
EMB_DIMS = [4, 32, 10, 48, 3]  # dong_code(428종)/industry_code(192종) 임베딩 확대

# 카운트/거리/인구 계열은 롱테일 분포라 z-score만으로는 큰 값에 민감해지기 쉬움.
# log1p로 눌러준 뒤 표준화 -> DNN이 패턴을 더 잘 잡도록.
# (비율/플래그/좁은 범위 값들은 그대로: foreign_short_ratio, *_historical_rate,
#  tourist_zone_candidate, population_is_proxied, previously_transitioned, lng/lat,
#  store_age_months(0~24), keyword_growth_score(0~1.8 정도)는 제외)
LOG1P_COLS = [
    "same_industry_count_300m", "total_count_300m",
    "nearest_same_industry_distance_m", "dong_industry_count", "coord_cluster_size",
    "korean_pop", "foreign_long_pop", "foreign_short_pop", "total_pop_avg",
]
_LOG1P_MASK = np.array([c in LOG1P_COLS for c in CONT_COLS])


def transform_cont(x_cont_raw: np.ndarray) -> np.ndarray:
    """CONT_COLS 순서의 원본 값 배열에 log1p를 선택 적용. score/shap 스크립트도 동일 함수를 import해서 씀."""
    x = x_cont_raw.astype(np.float32, copy=True)
    x[:, _LOG1P_MASK] = np.log1p(np.clip(x[:, _LOG1P_MASK], a_min=0, a_max=None))
    return x


TARGET_COL = "is_closed_next"
FOLD_COL = "fold"
ID_COLS = ["store_id", "industry_code", "snapshot_date"]  # 학습엔 미사용, predict 단계에서 필요
N_FOLDS = 5


def fold_of(store_id: str, k: int = N_FOLDS) -> int:
    """ml/build_modeling_dataset.py의 fold 배정과 완전히 동일한 해시.
    (검증됨: 데이터의 fold 컬럼과 20만 건 전수 비교 결과 mismatch=0)
    -> 학습 시 안 쓰인 store_id라도 이 함수만으로 '어느 fold 모델이 이 매장을 안 봤는지' 알 수 있음.
    """
    h = hashlib.md5(store_id.encode()).hexdigest()
    return int(h, 16) % k

USECOLS = CONT_COLS + CAT_COLS + [TARGET_COL, FOLD_COL] + ID_COLS


class ClosureMLP(nn.Module):
    """predict/shap 스크립트에서도 동일하게 import해서 state_dict를 로드함."""

    def __init__(self, n_cont, cat_cards, emb_dims):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cat_cards, emb_dims)])
        in_dim = n_cont + sum(emb_dims)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.25),
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
    x_cont = (transform_cont(df[CONT_COLS].to_numpy(dtype=np.float32)) - cont_mean) / cont_std
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
              epoch_num=None, total_epochs=None, grad_clip=None):
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
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def train_model(x_cont, x_cats, y, cat_cards, device, val_data=None, max_epochs=50, patience=6,
                 batch_size=4096, lr=1e-3, weight_decay=1e-5, grad_clip=5.0):
    model = ClosureMLP(n_cont=x_cont.shape[1], cat_cards=cat_cards, emb_dims=EMB_DIMS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2, min_lr=1e-5
    ) if val_data is not None else None
    pos_weight = make_pos_weight(y, device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score, best_epoch, best_state, no_improve = -1, 0, None, 0
    history = []
    stage = "Phase A" if val_data is not None else "Phase B"
    epoch_range = tqdm(range(1, max_epochs + 1), desc=f"{stage} 전체 진행률", ncols=100)
    for epoch in epoch_range:
        t0 = time.time()
        train_loss, _ = run_epoch(model, opt, lossfn, x_cont, x_cats, y, batch_size, train=True,
                                   epoch_num=epoch, total_epochs=max_epochs, grad_clip=grad_clip)
        log = f"epoch {epoch:02d}  train_loss={train_loss:.4f}  ({time.time()-t0:.1f}s)"

        if val_data is not None:
            vx_cont, vx_cats, vy = val_data
            val_loss, val_probs = run_epoch(model, opt, lossfn, vx_cont, vx_cats, vy, batch_size, train=False,
                                             epoch_num=epoch, total_epochs=max_epochs)
            vy_np = vy.cpu().numpy()
            val_auc = roc_auc_score(vy_np, val_probs)
            val_pr = average_precision_score(vy_np, val_probs)
            cur_lr = opt.param_groups[0]["lr"]
            log += f"  val_loss={val_loss:.4f}  val_ROC-AUC={val_auc:.4f}  val_PR-AUC={val_pr:.4f}  lr={cur_lr:.2e}"
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                             "val_roc_auc": float(val_auc), "val_pr_auc": float(val_pr), "lr": cur_lr})
            score = val_pr  # 불균형 데이터라 PR-AUC 기준 early stopping
            scheduler.step(score)
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
    ap.add_argument("--artifact-dir", default="src/project_2nd/models/dl/saved",
                     help="모델/스케일러/피처설정 저장 위치 (fold_0/ ~ fold_4/ 서브폴더 생성)")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS,
                     help="몇 개 fold까지 학습할지 (기본 5=전체, 시간 없으면 1~2로 줄여서 빠르게 테스트 가능)")
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

    # =====================================================================
    # 5-fold 완전 교차검증 + 프로덕션 앙상블
    #
    # fold_k 모델은 "fold k를 제외한 전체(4/5)"로 학습되고, fold k는 그 모델이
    # 한 번도 보지 못한 진짜 held-out 데이터로 성능 평가 + Platt calibration에 씀.
    # 서빙(dl_score_tm.py)할 때는 매장의 store_id로 fold_of()를 다시 계산해서
    # "그 매장을 한 번도 학습에 쓰지 않은 모델"로만 스코어링 -> 데이터 누수 0.
    # (5개 폴드 전체를 각각 다르게 held-out하므로 전체 매장이 커버되고,
    #  동시에 5개 독립 모델을 쓰는 것과 같은 효과라 앙상블 분산 감소 효과도 있음)
    #
    # 단계 (fold k마다):
    #   1) val_fold = (k+1)%5, train_folds = 나머지 3개  -> early stopping으로 best_epoch 탐색
    #   2) train_folds + val_fold(=4/5, fold k만 제외)로 best_epoch만큼 재학습 -> fold_k 프로덕션 모델
    #   3) fold k(진짜 unseen)에서 최종 평가 + Platt scaling 계산
    # =====================================================================
    n_folds = min(args.n_folds, N_FOLDS)
    fold_metrics = []

    for k in range(n_folds):
        val_fold = (k + 1) % N_FOLDS
        train_folds = [f for f in range(N_FOLDS) if f not in (k, val_fold)]
        print(f"\n{'='*70}\n[Fold {k}]  train_folds={train_folds}  val_fold={val_fold}  test_fold(held-out)={k}\n{'='*70}")

        train_df = df[df[FOLD_COL].isin(train_folds)]
        val_df = df[df[FOLD_COL] == val_fold]
        test_df = df[df[FOLD_COL] == k]

        cont_mean = transform_cont(train_df[CONT_COLS].to_numpy(dtype=np.float32)).mean(axis=0)
        cont_std = transform_cont(train_df[CONT_COLS].to_numpy(dtype=np.float32)).std(axis=0)
        cont_std[cont_std == 0] = 1.0

        xtr = to_tensors(train_df, cont_mean, cont_std, device)
        xval = to_tensors(val_df, cont_mean, cont_std, device)
        xtest = to_tensors(test_df, cont_mean, cont_std, device)

        print(f"[Fold {k}] 1/2 - early stopping용 후보 모델 학습 (train {len(train_df):,} / val {len(val_df):,})")
        _, best_epoch, _ = train_model(
            xtr[0], xtr[1], xtr[2], cat_cards, device, val_data=xval,
            max_epochs=args.max_epochs, patience=args.patience,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        )

        # ---- 2) fold k만 제외한 4/5 전체로 프로덕션 모델 재학습 ----
        print(f"\n[Fold {k}] 2/2 - fold {k} 제외 전체(4/5)로 프로덕션 모델 재학습 (epochs={best_epoch})")
        prod_df = df[df[FOLD_COL] != k]
        prod_cont_mean = transform_cont(prod_df[CONT_COLS].to_numpy(dtype=np.float32)).mean(axis=0)
        prod_cont_std = transform_cont(prod_df[CONT_COLS].to_numpy(dtype=np.float32)).std(axis=0)
        prod_cont_std[prod_cont_std == 0] = 1.0

        xprod = to_tensors(prod_df, prod_cont_mean, prod_cont_std, device)
        model_k, _, _ = train_model(
            xprod[0], xprod[1], xprod[2], cat_cards, device, val_data=None,
            max_epochs=best_epoch, patience=best_epoch,
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        )

        # ---- 3) fold k(진짜 unseen)로 최종 평가 + Platt calibration ----
        xtest_prod = to_tensors(test_df, prod_cont_mean, prod_cont_std, device)
        model_k.eval()
        with torch.no_grad():
            test_logits = model_k(xtest_prod[0], xtest_prod[1]).cpu().numpy()
        test_y = xtest_prod[2].cpu().numpy()
        test_probs_raw = 1.0 / (1.0 + np.exp(-test_logits))
        print(f"\n[Fold {k}] 최종 평가 (fold {k}, 이 모델이 학습에 전혀 안 쓴 데이터)")
        evaluate(test_probs_raw, test_y, label=f"Fold{k} raw")

        calib_lr = LogisticRegression(C=1e10, max_iter=1000)
        calib_lr.fit(test_logits.reshape(-1, 1), test_y)
        calib_a = float(calib_lr.coef_[0][0])
        calib_b = float(calib_lr.intercept_[0])
        test_probs_cal = 1.0 / (1.0 + np.exp(-(calib_a * test_logits + calib_b)))
        metrics_k = evaluate(test_probs_cal, test_y, label=f"Fold{k} calibrated")
        metrics_k["calibration_params"] = {"a": calib_a, "b": calib_b,
                                            "formula": "prob = sigmoid(a * raw_logit + b)"}
        metrics_k["trained_epochs"] = best_epoch
        fold_metrics.append(metrics_k)

        # ---- 저장: fold_k/ 서브폴더 ----
        fold_dir = artifact_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_k.state_dict(), fold_dir / "model_state.pt")
        with open(fold_dir / "scaler.json", "w", encoding="utf-8") as f:
            json.dump({"cont_cols": CONT_COLS, "mean": prod_cont_mean.tolist(),
                       "std": prod_cont_std.tolist(), "log1p_cols": LOG1P_COLS}, f,
                      ensure_ascii=False, indent=2)
        with open(fold_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(metrics_k, f, ensure_ascii=False, indent=2)
        print(f"[Fold {k}] 저장 완료 -> {fold_dir}/")

    # ---------------- 전체 fold 평균 성능 (single-split보다 훨씬 신뢰도 높은 추정치) ----------------
    print(f"\n{'='*70}\n[전체 {n_folds}-fold 평균 성능 (calibrated)]\n{'='*70}")
    roc_list = [m["roc_auc"] for m in fold_metrics]
    pr_list = [m["pr_auc"] for m in fold_metrics]
    lift_list = [m["top5pct_lift"] for m in fold_metrics]
    print(f"ROC-AUC   : {np.mean(roc_list):.4f} ± {np.std(roc_list):.4f}   (fold별: {[round(x,4) for x in roc_list]})")
    print(f"PR-AUC    : {np.mean(pr_list):.4f} ± {np.std(pr_list):.4f}   (fold별: {[round(x,4) for x in pr_list]})")
    print(f"Top5% Lift: {np.mean(lift_list):.2f}x ± {np.std(lift_list):.2f}   (fold별: {[round(x,2) for x in lift_list]})")
    print("참고 - LightGBM 베이스라인: ROC-AUC 0.721~0.728, PR-AUC 0.300~0.317, Lift 4.12x (single-split)")
    print("\n다음 단계: dl_test_tm.py로 accuracy/precision/recall/f1까지 계산해서 models 테이블용 지표 완성")

    feature_config = {
        "cont_cols": CONT_COLS,
        "cat_cols": CAT_COLS,
        "cat_cards": cat_cards,
        "emb_dims": EMB_DIMS,
        "target_col": TARGET_COL,
        "n_folds": n_folds,
        "fold_metrics": fold_metrics,
        "cv_summary": {
            "roc_auc_mean": float(np.mean(roc_list)), "roc_auc_std": float(np.std(roc_list)),
            "pr_auc_mean": float(np.mean(pr_list)), "pr_auc_std": float(np.std(pr_list)),
            "top5pct_lift_mean": float(np.mean(lift_list)), "top5pct_lift_std": float(np.std(lift_list)),
        },
        "model_arch": "emb -> concat -> 256-BN-ReLU-Drop(.35) -> 128-BN-ReLU-Drop(.3) -> "
                      "64-BN-ReLU-Drop(.25) -> 32-BN-ReLU-Drop(.2) -> 1(logit)",
        "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)",
        "serving_note": ("각 fold_k/ 모델은 fold k 매장을 학습에 전혀 쓰지 않음. "
                          "dl_score_tm.py는 store_id로 fold_of()를 다시 계산해서 "
                          "그 매장을 안 본 모델로만 스코어링함(데이터 누수 0)."),
    }
    with open(artifact_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    print(f"\n전체 저장 완료 -> {artifact_dir}/fold_0 ~ fold_{n_folds-1}/, ensemble_config.json")


if __name__ == "__main__":
    main()
