"""최신 PJW DNN RUN의 전역·개별 SHAP 예측 기여도를 생성한다.

``src/project_2nd/models/shap``은 읽거나 수정하지 않는다. 외부 ``shap`` Python
패키지의 DeepExplainer를 사용하고 결과는 해당 DNN RUN의 ``shap`` 폴더에 둔다.
SHAP 값은 raw DNN logit 기여도이며 상관계수나 인과효과가 아니다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Sequence
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from deep_mlp import DeepMLP


DL_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = DL_DIR / "saved" / "runs"


class LogitColumn(nn.Module):
    """DeepExplainer용 ``[N, 1]`` logit 출력 wrapper."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features).unsqueeze(1)


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def resolve_latest_run(runs_dir: Path) -> Path:
    with (runs_dir / "latest_run.json").open(encoding="utf-8") as file:
        run_dir = Path(json.load(file)["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"DNN RUN이 없습니다: {run_dir}")
    return run_dir.resolve()


def stratified_sample_indices(
    candidates: np.ndarray,
    all_labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """양성/음성 비율을 유지한 중복 없는 표본을 반환한다."""

    if sample_size < 2 or sample_size > len(candidates):
        raise ValueError("sample_size는 2 이상 후보 행 수 이하여야 합니다.")
    candidate_labels = np.asarray(all_labels[candidates], dtype=np.uint8)
    positive = candidates[candidate_labels == 1]
    negative = candidates[candidate_labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("후보에 양성과 음성이 모두 필요합니다.")
    positive_count = int(round(sample_size * len(positive) / len(candidates)))
    positive_count = min(max(1, positive_count), len(positive), sample_size - 1)
    negative_count = sample_size - positive_count
    rng = np.random.default_rng(seed)
    chosen = np.concatenate(
        [
            rng.choice(negative, size=negative_count, replace=False),
            rng.choice(positive, size=positive_count, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def normalize_shap_values(
    values: object, sample_count: int, feature_count: int
) -> np.ndarray:
    """SHAP 버전별 반환 형식을 ``[N, F]``로 통일한다."""

    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("단일 출력 DNN인데 SHAP 출력이 여러 개입니다.")
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape == (feature_count, sample_count):
        array = array.T
    if array.shape != (sample_count, feature_count):
        raise ValueError(f"SHAP shape 오류: {array.shape}")
    return array.astype(np.float32, copy=False)


def contribution_rows(
    feature_names: Sequence[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    direction: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    if direction == "up":
        candidates = np.flatnonzero(shap_values > 0)
        order = candidates[np.argsort(shap_values[candidates])[::-1]]
    elif direction == "down":
        candidates = np.flatnonzero(shap_values < 0)
        order = candidates[np.argsort(shap_values[candidates])]
    else:
        raise ValueError("direction은 up 또는 down이어야 합니다.")
    return [
        {
            "feature": feature_names[index],
            "feature_value": float(feature_values[index]),
            "shap_value": float(shap_values[index]),
            "direction": direction,
            "rank": rank,
        }
        for rank, index in enumerate(order[:limit], start=1)
    ]


def safe_filename(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))[:100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PJW DNN SHAP 분석")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--background-size", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--waterfall-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 False입니다. CPU로 대체하지 않고 중단합니다.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else resolve_latest_run(args.runs_dir.resolve())
    )
    output_dir = run_dir / "shap"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"기존 SHAP 결과가 있습니다: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_dir / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "run_manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest["feature_policy"] != "pjw_official" or "pmh" in manifest["dataset_path"].lower():
        raise ValueError("최신 RUN이 PJW 공식 DNN 결과가 아닙니다.")
    data_dir = Path(manifest["prepared_data_dir"])
    with (data_dir / "preprocessing_metadata.json").open(encoding="utf-8") as file:
        preprocessing = json.load(file)

    checkpoint = torch.load(
        run_dir / "model" / "dnn_model.pt", map_location=device, weights_only=False
    )
    feature_names = list(checkpoint["feature_names"])
    model = DeepMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["model_config"]["hidden_dims"]),
        dropout=float(checkpoint["model_config"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapped = LogitColumn(model.eval()).to(device).eval()

    features = np.load(data_dir / "X.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(data_dir / "y.npy", mmap_mode="r", allow_pickle=False)
    split = np.load(data_dir / "split.npy", mmap_mode="r", allow_pickle=False)
    rows = pd.read_parquet(data_dir / "row_metadata.parquet")
    if not (len(features) == len(labels) == len(split) == len(rows)):
        raise ValueError("X/y/split/row metadata 행 수 불일치")

    train_indices = np.flatnonzero(np.isin(split, [0, 1, 2]))
    test_indices = np.flatnonzero(split == 4)
    background_indices = stratified_sample_indices(
        train_indices, labels, args.background_size, args.seed
    )
    sample_indices = stratified_sample_indices(
        test_indices, labels, args.sample_size, args.seed + 1
    )
    background = np.ascontiguousarray(features[background_indices], dtype=np.float32)
    sample = np.ascontiguousarray(features[sample_indices], dtype=np.float32)
    background_tensor = torch.from_numpy(background).to(device)
    sample_tensor = torch.from_numpy(sample).to(device)

    print(f"RUN: {run_dir.name}", flush=True)
    print(f"장치: cuda:0 ({torch.cuda.get_device_name(0)})", flush=True)
    print(
        f"DeepExplainer: background={len(background_indices)}, test={len(sample_indices)}",
        flush=True,
    )
    explainer = shap.DeepExplainer(wrapped, background_tensor)
    raw_values = explainer.shap_values(sample_tensor, check_additivity=False)
    shap_values = normalize_shap_values(raw_values, len(sample_indices), len(feature_names))
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    with torch.inference_mode():
        logits = wrapped(sample_tensor).squeeze(1).detach().cpu().numpy()
    error = np.abs(base_value + shap_values.sum(axis=1) - logits)
    additivity = {
        "mean_absolute_error": float(error.mean()),
        "max_absolute_error": float(error.max()),
        "p95_absolute_error": float(np.quantile(error, 0.95)),
        "tolerance": 0.05,
        "passed": bool(float(error.max()) <= 0.05),
    }
    if not additivity["passed"]:
        raise ValueError(f"SHAP 가산성 검증 실패: {additivity}")

    scaler = preprocessing["scaler"]
    display_values = sample * np.asarray(scaler["scale"], dtype=np.float32) + np.asarray(
        scaler["mean"], dtype=np.float32
    )
    importance = np.mean(np.abs(shap_values), axis=0)
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap_logit": importance}
    ).sort_values("mean_abs_shap_logit", ascending=False)
    importance_frame.insert(0, "rank", np.arange(1, len(importance_frame) + 1))
    importance_frame.to_csv(
        output_dir / "shap_global_importance.csv", index=False, encoding="utf-8-sig"
    )

    for plot_type, filename in (
        ("bar", "shap_summary_bar.png"),
        (None, "shap_summary_beeswarm.png"),
    ):
        plt.figure()
        shap.summary_plot(
            shap_values,
            display_values,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=min(20, len(feature_names)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=170, bbox_inches="tight")
        plt.close()

    sample_rows = rows.iloc[sample_indices].reset_index(drop=True)
    risk_scores = (1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))).astype(np.float32)
    local_results: list[dict[str, object]] = []
    for index in range(len(sample_indices)):
        local_results.append(
            {
                "row_id": int(sample_rows.loc[index, "row_id"]),
                "store_id": str(sample_rows.loc[index, "store_id"]),
                "snapshot_date": str(sample_rows.loc[index, "snapshot_date"]),
                "y_true": int(labels[sample_indices[index]]),
                "risk_score_uncalibrated": float(risk_scores[index]),
                "model_logit": float(logits[index]),
                "base_value_logit": base_value,
                "top_risk_up": contribution_rows(
                    feature_names, display_values[index], shap_values[index], "up"
                ),
                "top_risk_down": contribution_rows(
                    feature_names, display_values[index], shap_values[index], "down"
                ),
            }
        )
    json_dump(output_dir / "shap_local_top_features.json", local_results)

    np.savez_compressed(
        output_dir / "shap_values_sample.npz",
        row_id=sample_rows["row_id"].to_numpy(dtype=np.int64),
        sample_indices=sample_indices,
        background_indices=background_indices,
        shap_values=shap_values,
        standardized_feature_values=sample,
        display_feature_values=display_values,
        base_value_logit=np.float32(base_value),
        model_logits=logits.astype(np.float32),
        feature_names=np.asarray(feature_names),
    )

    waterfall_files: list[str] = []
    for index in np.argsort(risk_scores)[::-1][: args.waterfall_count]:
        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=base_value,
            data=display_values[index],
            feature_names=feature_names,
        )
        shap.plots.waterfall(
            explanation, max_display=min(15, len(feature_names)), show=False
        )
        filename = (
            f"waterfall_{safe_filename(sample_rows.loc[index, 'store_id'])}_"
            f"{safe_filename(sample_rows.loc[index, 'snapshot_date'])}.png"
        )
        plt.gcf().savefig(local_dir / filename, dpi=170, bbox_inches="tight")
        plt.close()
        waterfall_files.append(filename)

    metadata = {
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "dnn_run_id": manifest["run_id"],
        "source_pipeline": preprocessing["source_pipeline"],
        "dataset_path": preprocessing["source_path"],
        "device": f"cuda:0 ({torch.cuda.get_device_name(0)})",
        "torch_version": torch.__version__,
        "shap_version": shap.__version__,
        "explainer": "DeepExplainer",
        "output_scale": "raw DNN logit",
        "base_value_logit": base_value,
        "seed": args.seed,
        "background_size": len(background_indices),
        "sample_size": len(sample_indices),
        "background_class_counts": {
            "negative_0": int((labels[background_indices] == 0).sum()),
            "positive_1": int((labels[background_indices] == 1).sum()),
        },
        "sample_class_counts": {
            "negative_0": int((labels[sample_indices] == 0).sum()),
            "positive_1": int((labels[sample_indices] == 1).sum()),
        },
        "feature_names": feature_names,
        "additivity": additivity,
        "waterfall_files": waterfall_files,
        "interpretation_warning": (
            "SHAP is prediction contribution, not correlation or causality; "
            "logit contributions are not probability percentage points."
        ),
    }
    json_dump(output_dir / "shap_metadata.json", metadata)

    top_lines = "\n".join(
        f"| {int(row['rank'])} | {row['feature']} | {row['mean_abs_shap_logit']:.6f} |"
        for _, row in importance_frame.head(10).iterrows()
    )
    report = f"""# PJW DNN SHAP 분석 결과

- DNN RUN: `{manifest['run_id']}`
- 데이터: `{preprocessing['source_path']}`
- 전처리: `{preprocessing['source_pipeline']}`
- Explainer: `DeepExplainer`
- 출력 단위: raw DNN logit
- Background: train 계층표본 {len(background_indices):,}건
- 설명 대상: test 계층표본 {len(sample_indices):,}건
- 가산성 평균/최대 오차: {additivity['mean_absolute_error']:.6f} / {additivity['max_absolute_error']:.6f}
- 가산성 검증: {additivity['passed']}

## 전역 중요도 상위 10개

| 순위 | 피처 | mean(abs(SHAP)) |
|---:|---|---:|
{top_lines}

## 해석 주의사항

- positive/negative SHAP은 모델의 폐업 위험 logit을 올리거나 내린 기여다.
- SHAP은 상관계수나 폐업의 원인이 아니다.
- SHAP 값을 확률 퍼센트포인트로 표현하면 안 된다.
- weighted BCE sigmoid 출력은 미보정 위험점수다.
"""
    (output_dir / "shap_report.md").write_text(report, encoding="utf-8")

    print(f"가산성: {additivity}", flush=True)
    print(f"전역 1위: {importance_frame.iloc[0]['feature']}", flush=True)
    print(f"SHAP 완료: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
