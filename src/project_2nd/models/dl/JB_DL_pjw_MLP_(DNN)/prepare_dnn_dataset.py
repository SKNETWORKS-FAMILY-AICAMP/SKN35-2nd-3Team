"""PJW 정제 CSV를 심층 MLP 학습용 배열과 전처리 JSON으로 변환한다.

모든 결측 대체값, 범주 매핑, 평균과 표준편차는 학습 fold에서만 계산한다.
검증·테스트 데이터의 정보를 전처리에 사용하지 않아 데이터 누수를 막는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
SOURCE = REPO_ROOT / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
OUTPUT_DIR = HERE / "artifacts" / "data"
CONFIG_PATH = HERE / "config.json"

TARGET_COLUMN = "is_closed_next"
FOLD_COLUMN = "fold"
DROP_COLUMNS = ["store_id"]
MISSING_CATEGORY = "__MISSING__"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def json_float(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"입력 파일이 없습니다: {SOURCE}\n"
            "먼저 run_pipeline.ps1과 preprocess_modeling_dataset_pjw.py를 실행하세요."
        )

    config = load_config()
    train_folds = set(config["train_folds"])
    validation_folds = set(config["validation_folds"])
    test_folds = set(config["test_folds"])
    configured_folds = train_folds | validation_folds | test_folds
    if train_folds & validation_folds or train_folds & test_folds or validation_folds & test_folds:
        raise ValueError("train/validation/test fold가 서로 겹치면 안 됩니다.")

    print(f"입력 데이터 로딩: {SOURCE}")
    df = pd.read_csv(SOURCE, low_memory=False)
    required = {TARGET_COLUMN, FOLD_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    fold = pd.to_numeric(df[FOLD_COLUMN], errors="raise").astype(np.int8)
    unexpected_folds = sorted(set(fold.unique()) - configured_folds)
    if unexpected_folds:
        raise ValueError(f"config.json에 정의되지 않은 fold가 있습니다: {unexpected_folds}")

    target = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(np.int8)
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("타깃은 0/1 이진값이어야 합니다.")

    train_mask = fold.isin(train_folds).to_numpy()
    validation_mask = fold.isin(validation_folds).to_numpy()
    test_mask = fold.isin(test_folds).to_numpy()
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("train/validation/test 중 비어 있는 분할이 있습니다.")

    excluded = {TARGET_COLUMN, FOLD_COLUMN, *DROP_COLUMNS}
    feature_columns = [column for column in df.columns if column not in excluded]
    features = df[feature_columns].copy()
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    category_mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = features[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        train_categories = sorted(pd.unique(values.loc[train_mask]).tolist())
        mapping = {value: index for index, value in enumerate(train_categories)}
        features[column] = values.map(mapping).fillna(-1).astype(np.float32)
        category_mappings[column] = mapping

    numeric_imputation: dict[str, float] = {}
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = values.loc[train_mask].median()
        if pd.isna(median):
            median = 0.0
        numeric_imputation[column] = json_float(median)
        features[column] = values.fillna(median).astype(np.float32)

    x = features[feature_columns].to_numpy(dtype=np.float32, copy=True)
    train_x = x[train_mask]
    means = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    stds[~np.isfinite(stds) | (stds < 1e-8)] = 1.0
    means[~np.isfinite(means)] = 0.0
    x -= means
    x /= stds
    if not np.isfinite(x).all():
        raise ValueError("전처리 후 입력 배열에 NaN 또는 무한대가 남았습니다.")

    split = np.full(len(df), -1, dtype=np.int8)
    split[train_mask] = 0
    split[validation_mask] = 1
    split[test_mask] = 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", x, allow_pickle=False)
    np.save(OUTPUT_DIR / "y.npy", target.to_numpy(dtype=np.float32), allow_pickle=False)
    np.save(OUTPUT_DIR / "split.npy", split, allow_pickle=False)

    class_counts = target.value_counts().sort_index().to_dict()
    metadata = {
        "source": str(SOURCE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rows": int(len(df)),
        "input_dimension": int(len(feature_columns)),
        "target_column": TARGET_COLUMN,
        "feature_order": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "category_mappings": category_mappings,
        "numeric_imputation": numeric_imputation,
        "scaling": {
            "mean": {column: json_float(value) for column, value in zip(feature_columns, means)},
            "std": {column: json_float(value) for column, value in zip(feature_columns, stds)},
        },
        "split": {
            "train_folds": sorted(train_folds),
            "validation_folds": sorted(validation_folds),
            "test_folds": sorted(test_folds),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "test_rows": int(test_mask.sum()),
        },
        "target_counts": {str(key): int(value) for key, value in class_counts.items()},
        "unknown_category_value": -1,
        "missing_category_token": MISSING_CATEGORY,
    }
    (OUTPUT_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"전처리 완료: {len(df):,}행 x {len(feature_columns)}피처")
    print(
        "분할: "
        f"train={train_mask.sum():,}, validation={validation_mask.sum():,}, test={test_mask.sum():,}"
    )
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
