"""심층 MLP(DNN)를 최대 7 epoch 학습하고 공통 평가지표를 저장한다."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "artifacts" / "data"
MODEL_DIR = HERE / "artifacts" / "model"
CONFIG_PATH = HERE / "config.json"
MAX_EPOCHS = 7


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    order = indices.copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def predict(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    for batch_indices in iter_batches(indices, batch_size, shuffle=False, seed=0):
        xb = torch.from_numpy(np.asarray(x[batch_indices], dtype=np.float32)).to(device)
        probabilities.append(torch.sigmoid(model(xb)).cpu().numpy())
        targets.append(np.asarray(y[batch_indices], dtype=np.int8))
    return np.concatenate(targets), np.concatenate(probabilities)


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int8)
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "threshold": float(threshold),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    epochs = int(config["epochs"])
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs는 1~{MAX_EPOCHS}만 허용합니다. 현재 값: {epochs}")

    required_files = [DATA_DIR / "X.npy", DATA_DIR / "y.npy", DATA_DIR / "split.npy"]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "DNN 학습 배열이 없습니다. prepare_dnn_dataset.py를 먼저 실행하세요:\n"
            + "\n".join(missing)
        )

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.load(DATA_DIR / "X.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y.npy", mmap_mode="r")
    split = np.load(DATA_DIR / "split.npy", mmap_mode="r")
    train_indices = np.flatnonzero(split == 0)
    validation_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)

    positives = float(np.asarray(y[train_indices]).sum())
    negatives = float(len(train_indices) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("학습 데이터에 0과 1 클래스가 모두 있어야 합니다.")

    model = DeepMLP(input_dim=x.shape[1], dropout=float(config["dropout"])).to(device)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    batch_size = int(config["batch_size"])
    patience = int(config["early_stopping_patience"])
    best_pr_auc = -np.inf
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []

    print(f"장치: {device}, 입력 피처: {x.shape[1]}, 최대 epoch: {epochs}")
    print(f"학습 표본: {len(train_indices):,}, pos_weight: {negatives / positives:.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_indices in iter_batches(train_indices, batch_size, shuffle=True, seed=seed + epoch):
            if len(batch_indices) < 2:
                continue
            xb = torch.from_numpy(np.asarray(x[batch_indices], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.asarray(y[batch_indices], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            seen += len(batch_indices)

        val_y, val_probability = predict(
            model, x, y, validation_indices, batch_size=batch_size, device=device
        )
        val_pr_auc = float(average_precision_score(val_y, val_probability))
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "validation_pr_auc": val_pr_auc,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}/{epochs} - loss={epoch_record['train_loss']:.6f} "
            f"- val_pr_auc={val_pr_auc:.6f}"
        )

        if val_pr_auc > best_pr_auc + 1e-6:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"조기 종료: {patience}회 연속 개선 없음")
                break

    if best_state is None:
        raise RuntimeError("저장할 최적 모델이 없습니다.")
    model.load_state_dict(best_state)

    val_y, val_probability = predict(model, x, y, validation_indices, batch_size, device)
    threshold = choose_threshold(val_y, val_probability)
    test_y, test_probability = predict(model, x, y, test_indices, batch_size, device)
    result = {
        "model_name": config["model_name"],
        "architecture": [int(x.shape[1]), 256, 128, 64, 1],
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "validation": metrics(val_y, val_probability, threshold),
        "test": metrics(test_y, test_probability, threshold),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": int(x.shape[1]),
            "hidden_layers": [256, 128, 64],
            "dropout": float(config["dropout"]),
            "threshold": threshold,
            "config": config,
        },
        MODEL_DIR / "dnn_model.pt",
    )
    (MODEL_DIR / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"모델 및 결과 저장: {MODEL_DIR}")


if __name__ == "__main__":
    main()
