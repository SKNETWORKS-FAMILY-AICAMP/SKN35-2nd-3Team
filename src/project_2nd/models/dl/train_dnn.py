"""NVIDIA CUDA에서 서울 상권 폐업 예측 Deep MLP를 학습·평가한다.

이 스크립트는 CPU로 자동 대체하지 않는다. CUDA가 확인되지 않으면 학습 전에
즉시 중단한다. 모델 선택은 validation PR-AUC, 분류 임계값 선택은 validation
F1만 사용하며 test는 마지막 1회 평가에만 사용한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_mlp import DeepMLP, ModelConfig, set_global_seed


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed" / "dnn_pjw_official"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "saved" / "runs"
MAX_ALLOWED_EPOCHS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA Deep MLP 학습")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def choose_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """validation 데이터에서 F1이 가장 큰 임계값과 그 F1을 반환한다."""

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    """동일 폭 구간 기반 Expected Calibration Error를 계산한다."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bucket == index
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return ece


def top_fraction_metrics(
    labels: np.ndarray, scores: np.ndarray, fraction: float
) -> tuple[float, float, int]:
    count = max(1, int(math.ceil(len(scores) * fraction)))
    chosen = np.argpartition(scores, -count)[-count:]
    positives = int(labels[chosen].sum())
    precision = positives / count
    total_positives = int(labels.sum())
    recall = positives / total_positives if total_positives else 0.0
    return float(precision), float(recall), count


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, object]:
    """불균형 이진 분류에 필요한 전체 평가 지표를 계산한다."""

    predictions = (scores >= threshold).astype(np.uint8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    p5, r5, n5 = top_fraction_metrics(labels, scores, 0.05)
    p10, r10, n10 = top_fraction_metrics(labels, scores, 0.10)
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "brier_score_uncalibrated": float(brier_score_loss(labels, scores)),
        "ece_10bin_uncalibrated": expected_calibration_error(labels, scores, bins=10),
        "precision_at_top_5_percent": p5,
        "recall_at_top_5_percent": r5,
        "top_5_percent_count": n5,
        "precision_at_top_10_percent": p10,
        "recall_at_top_10_percent": r10,
        "top_10_percent_count": n10,
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "support": {"negative_0": int((labels == 0).sum()), "positive_1": int((labels == 1).sum())},
    }


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    """NumPy 분할을 PyTorch DataLoader로 만든다.

    Windows에서 안정적으로 실행되도록 기본 worker는 0이다. 데이터는 한 번만
    연속 메모리로 복사하고 각 epoch에서는 이 텐서를 재사용한다.
    """

    x_tensor = torch.from_numpy(np.ascontiguousarray(features[indices], dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(labels[indices], dtype=np.float32))
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        rows = batch_labels.shape[0]
        total_loss += float(loss.detach().item()) * rows
        total_rows += rows
    return total_loss / total_rows


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_rows = 0
    total_loss = 0.0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        rows = batch_labels.shape[0]
        total_loss += float(loss.item()) * rows
        total_rows += rows
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy().astype(np.uint8, copy=False))
    return total_loss / total_rows, np.concatenate(all_logits), np.concatenate(all_labels)


def save_plots(
    plot_dir: Path,
    history: list[dict[str, float]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(item["epoch"]) for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [item["validation_pr_auc"] for item in history], marker="o")
    axes[1].set_title("Validation PR-AUC")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(plot_dir / "training_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision)
    axis.set(xlabel="Recall", ylabel="Precision", title="Test Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "pr_curve.png", dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Test ROC Curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    predictions = (scores >= threshold).astype(np.uint8)
    fig, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(labels, predictions, labels=[0, 1], ax=axis, colorbar=False)
    axis.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="uniform")
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(predicted, observed, marker="o", label="DNN (uncalibrated)")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="ideal")
    axis.set(xlabel="Mean predicted risk score", ylabel="Observed positive rate", title="Calibration Curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "calibration_curve.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_epochs <= MAX_ALLOWED_EPOCHS:
        raise ValueError(f"max_epochs는 1~{MAX_ALLOWED_EPOCHS}만 허용됩니다.")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience는 1 이상이어야 합니다.")
    if args.batch_size < 2:
        raise ValueError("BatchNorm 학습을 위해 batch_size는 2 이상이어야 합니다.")

    # CPU fallback은 금지한다. 이 검증은 데이터 로딩과 RUN 폴더 생성보다 먼저 한다.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 False입니다. CPU로 대체하지 않고 학습을 중단합니다.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)

    data_dir = args.data_dir.resolve()
    required = {
        "X": data_dir / "X.npy",
        "y": data_dir / "y.npy",
        "split": data_dir / "split.npy",
        "metadata": data_dir / "preprocessing_metadata.json",
        "rows": data_dir / "row_metadata.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DNN 입력 산출물이 없습니다: {missing}")

    with required["metadata"].open(encoding="utf-8") as file:
        preprocessing_metadata = json.load(file)
    feature_names = preprocessing_metadata["feature_names"]

    features = np.load(required["X"], mmap_mode="r", allow_pickle=False)
    labels = np.load(required["y"], mmap_mode="r", allow_pickle=False)
    split = np.load(required["split"], mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("X shape와 preprocessing metadata의 피처 수가 다릅니다.")
    if not (features.shape[0] == labels.shape[0] == split.shape[0]):
        raise ValueError("X, y, split의 행 수가 다릅니다.")
    if not np.isfinite(features).all():
        raise ValueError("X에 NaN/Inf가 있습니다.")

    train_indices = np.flatnonzero(np.isin(split, [0, 1, 2]))
    validation_indices = np.flatnonzero(split == 3)
    test_indices = np.flatnonzero(split == 4)
    train_labels = np.asarray(labels[train_indices], dtype=np.uint8)
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("train에 양성과 음성이 모두 있어야 합니다.")
    pos_weight_value = negatives / positives

    set_global_seed(args.seed)
    config = ModelConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )

    git_commit = command_output(["git", "rev-parse", "HEAD"]) or "unknown"
    git_branch = command_output(["git", "branch", "--show-current"]) or "unknown"
    short_sha = git_commit[:7] if git_commit != "unknown" else "unknown"
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    run_id = f"DNN_{now:%Y%m%d_%H%M%S}_{short_sha}_{preprocessing_metadata['feature_policy']}"
    run_dir = args.runs_dir.resolve() / run_id
    model_dir = run_dir / "model"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    plots_dir = run_dir / "plots"
    for directory in (model_dir, metrics_dir, predictions_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=False)
    command_log = run_dir / "command_log.txt"

    def log(message: str) -> None:
        print(message, flush=True)
        with command_log.open("a", encoding="utf-8") as file:
            file.write(message + "\n")

    log(f"RUN_ID: {run_id}")
    log(f"장치: cuda:0 ({gpu_name})")
    log(
        f"분할: train={len(train_indices):,}, validation={len(validation_indices):,}, "
        f"test={len(test_indices):,}"
    )
    log(f"입력 피처 수: {features.shape[1]}")
    log(f"pos_weight: {pos_weight_value:.6f}")
    log(
        f"max epochs={config.max_epochs}, patience={config.early_stopping_patience}, "
        f"seed={config.seed}, batch={config.batch_size}"
    )

    train_loader = make_loader(
        features, labels, train_indices, config.batch_size, True, config.seed, args.num_workers
    )
    validation_loader = make_loader(
        features, labels, validation_indices, config.batch_size, False, config.seed, args.num_workers
    )
    test_loader = make_loader(
        features, labels, test_indices, config.batch_size, False, config.seed, args.num_workers
    )

    model = DeepMLP(
        input_dim=features.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[dict[str, float]] = []
    best_pr_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = model_dir / "dnn_model.pt"

    try:
        for epoch in range(1, config.max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            validation_loss, validation_logits, validation_labels = predict_loader(
                model, validation_loader, criterion, device
            )
            validation_scores = sigmoid(validation_logits)
            validation_pr_auc = float(
                average_precision_score(validation_labels, validation_scores)
            )
            validation_roc_auc = float(roc_auc_score(validation_labels, validation_scores))
            epoch_result = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_pr_auc": validation_pr_auc,
                "validation_roc_auc": validation_roc_auc,
            }
            history.append(epoch_result)
            log(
                f"Epoch {epoch}/{config.max_epochs} | train_loss={train_loss:.6f} | "
                f"val_loss={validation_loss:.6f} | val_PR-AUC={validation_pr_auc:.6f} | "
                f"val_ROC-AUC={validation_roc_auc:.6f}"
            )

            if validation_pr_auc > best_pr_auc + 1e-8:
                best_pr_auc = validation_pr_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": features.shape[1],
                        "feature_names": feature_names,
                        "model_config": config.to_dict(),
                        "best_epoch": best_epoch,
                        "best_validation_pr_auc": best_pr_auc,
                        "output_scale": "logit",
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    log(f"Early stopping: {epoch} epoch에서 중단")
                    break
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU OOM입니다. 인수인계서에 따라 batch 4096→2048 변경 승인이 필요합니다."
        ) from error

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    validation_loss, validation_logits, validation_labels = predict_loader(
        model, validation_loader, criterion, device
    )
    validation_scores = sigmoid(validation_logits)
    threshold, validation_best_f1 = choose_f1_threshold(validation_labels, validation_scores)
    validation_metrics = classification_metrics(validation_labels, validation_scores, threshold)
    validation_metrics["loss"] = validation_loss
    validation_metrics["best_f1_from_threshold_search"] = validation_best_f1

    # threshold를 확정한 뒤에만 test를 1회 평가한다.
    test_loss, test_logits, test_labels = predict_loader(model, test_loader, criterion, device)
    test_scores = sigmoid(test_logits)
    test_metrics = classification_metrics(test_labels, test_scores, threshold)
    test_metrics["loss"] = test_loss

    checkpoint["validation_threshold"] = threshold
    checkpoint["validation_metrics"] = validation_metrics
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, checkpoint_path)

    json_dump(metrics_dir / "history.json", history)
    metrics = {
        "primary_metric": "PR-AUC (average precision)",
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_pr_auc,
        "threshold_selected_on": "validation fold 3, maximum F1",
        "threshold": threshold,
        "risk_score_note": "weighted-BCE sigmoid output; uncalibrated risk score, not calibrated probability",
        "validation": validation_metrics,
        "test": test_metrics,
    }
    json_dump(metrics_dir / "metrics.json", metrics)
    json_dump(model_dir / "config.json", config.to_dict())

    row_metadata = pd.read_parquet(required["rows"])
    if len(row_metadata) != len(split):
        raise ValueError("row_metadata와 DNN 배열의 행 수가 다릅니다.")
    test_rows = row_metadata.iloc[test_indices].copy()
    test_rows["y_true"] = test_labels
    test_rows["risk_score"] = test_scores
    test_rows["logit"] = test_logits.astype(np.float32)
    test_rows["threshold"] = np.float32(threshold)
    test_rows["y_pred"] = (test_scores >= threshold).astype(np.uint8)
    test_rows["model_id"] = run_id
    prediction_path = predictions_dir / "test_predictions.parquet"
    test_rows.to_parquet(prediction_path, index=False)

    save_plots(plots_dir, history, test_labels, test_scores, threshold)

    driver_info = command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    manifest = {
        "run_id": run_id,
        "created_at_kst": now.isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_branch": git_branch,
        "git_commit": git_commit,
        "dataset_path": preprocessing_metadata["source_path"],
        "dataset_sha256": preprocessing_metadata["source_sha256"],
        "prepared_data_dir": str(data_dir),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "driver_version": driver_info,
        "sklearn_version": sklearn.__version__,
        "seed": config.seed,
        "model_config": config.to_dict(),
        "model_parameter_count": parameter_count,
        "split_definition": preprocessing_metadata["split_definition"],
        "split_counts": preprocessing_metadata["split_counts"],
        "feature_policy": preprocessing_metadata["feature_policy"],
        "feature_names": feature_names,
        "target_name": preprocessing_metadata["target_name"],
        "output_scale": "logit; sigmoid is an uncalibrated risk score",
        "notes": [
            "model selection used validation PR-AUC only",
            "classification threshold was selected on validation only",
            "test was evaluated after model and threshold selection",
        ],
    }
    json_dump(run_dir / "run_manifest.json", manifest)
    json_dump(
        args.runs_dir.resolve() / "latest_run.json",
        {"run_id": run_id, "run_dir": str(run_dir), "test_pr_auc": test_metrics["pr_auc"]},
    )

    log(f"최적 epoch: {best_epoch}, validation PR-AUC={best_pr_auc:.6f}")
    log(f"validation 선택 threshold: {threshold:.6f}")
    log(
        f"TEST | PR-AUC={test_metrics['pr_auc']:.6f} | ROC-AUC={test_metrics['roc_auc']:.6f} | "
        f"F1={test_metrics['f1']:.6f} | Precision={test_metrics['precision']:.6f} | "
        f"Recall={test_metrics['recall']:.6f}"
    )
    log(f"완료 산출물: {run_dir}")


if __name__ == "__main__":
    main()
