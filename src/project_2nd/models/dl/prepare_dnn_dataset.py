"""PJW 정제 CSV를 누수 방지된 DNN 배열로 변환한다.

입력은 ``preprocessing_dataset_pjw``가 생성한
``data/processed/modeling_dataset_refined_pjw.csv`` 한 파일만 사용한다.
``preprocessing_dataset_pmh`` 산출물은 읽지 않는다.

핵심 원칙
---------
1. 타깃, fold, 매장 식별자, 미래 라벨(`transitioned_next`)은 입력하지 않는다.
2. 범주형 8개는 train(fold 0~2)에서만 정수 매핑을 fit한다.
3. 결측치 대체값과 StandardScaler도 train에서만 fit한다.
4. CSV 행과 실제 매장을 다시 연결하도록 row metadata를 함께 저장한다.
5. 날짜 ``202312``는 최초 스냅샷부터의 개월 수로 변환한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "dnn_pjw_official"

TARGET_COLUMN = "is_closed_next"
SPLIT_COLUMN = "fold"
ID_COLUMN = "store_id"
DATE_COLUMN = "snapshot_date"
FEATURE_POLICY = "pjw_official"

CATEGORICAL_COLUMNS = (
    "industry_dae_code",
    "industry_group",
    "industry_jung_code",
    "industry_jung_name",
    "industry_code",
    "industry_name",
    "gu_name",
    "floor_category",
)

HARD_EXCLUDED_COLUMNS = frozenset(
    {TARGET_COLUMN, SPLIT_COLUMN, ID_COLUMN, "transitioned_next"}
)
REQUIRED_COLUMNS = frozenset({TARGET_COLUMN, SPLIT_COLUMN, ID_COLUMN, DATE_COLUMN})
VALID_FOLDS = frozenset({0, 1, 2, 3, 4})


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_feature_plan(
    columns: Sequence[str],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """``(출력명, 원본명, 처리방법)`` 목록과 제외 사유를 반환한다."""

    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")

    plan: list[tuple[str, str, str]] = []
    excluded: dict[str, str] = {}
    for column in columns:
        if column in HARD_EXCLUDED_COLUMNS:
            excluded[column] = "target/split/id/future-label"
        elif column == DATE_COLUMN:
            plan.append(("snapshot_month_index", column, "month_index"))
        elif column in CATEGORICAL_COLUMNS:
            plan.append((f"{column}_enc", column, "train_only_category_encoding"))
        else:
            plan.append((column, column, "numeric"))
    if not plan:
        raise ValueError("선택된 DNN 피처가 없습니다.")
    return plan, excluded


def snapshot_to_month_index(values: Iterable[object]) -> np.ndarray:
    text = pd.Series(values, dtype="string").str.replace(r"\.0$", "", regex=True)
    valid = text.str.fullmatch(r"\d{6}")
    if not bool(valid.all()):
        raise ValueError(f"snapshot_date 형식 오류: {text.loc[~valid].head(5).tolist()}")
    year = text.str[:4].astype(np.int32)
    month = text.str[4:6].astype(np.int16)
    if not bool(month.between(1, 12).all()):
        raise ValueError("snapshot_date에 1~12가 아닌 월이 있습니다.")
    absolute = year.to_numpy(dtype=np.int32) * 12 + month.to_numpy(dtype=np.int32) - 1
    return (absolute - absolute.min()).astype(np.float32)


def encode_categorical_train_only(
    values: pd.Series, train_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, int], int]:
    """train에서만 범주 사전을 만들고 미등록/결측 범주는 -1로 인코딩한다."""

    text = values.astype("string")
    categories = sorted(str(value) for value in text.loc[train_mask].dropna().unique())
    mapping = {value: index for index, value in enumerate(categories)}
    encoded = text.map(mapping).fillna(-1).to_numpy(dtype=np.float32)
    unknown_count = int((encoded == -1).sum())
    return encoded, mapping, unknown_count


def impute_nonfinite_train_median(
    features: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, list[float], list[int]]:
    """NaN/Inf를 피처별 train 중앙값으로 대체한다."""

    result = features.copy()
    medians: list[float] = []
    counts: list[int] = []
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        train_values = column[train_mask]
        finite_train = train_values[np.isfinite(train_values)]
        if finite_train.size == 0:
            raise ValueError(f"피처 {column_index}의 train 유한값이 없습니다.")
        median = float(np.median(finite_train))
        missing = ~np.isfinite(column)
        count = int(missing.sum())
        if count:
            column[missing] = median
        medians.append(median)
        counts.append(count)
    return result, medians, counts


def validate_split_and_target(split: np.ndarray, target: np.ndarray) -> None:
    target_values = set(np.unique(target).tolist())
    if target_values != {0, 1}:
        raise ValueError(f"타깃은 0과 1을 모두 포함해야 합니다: {target_values}")
    fold_values = set(np.unique(split).tolist())
    if fold_values != VALID_FOLDS:
        raise ValueError(f"fold는 정확히 0~4여야 합니다: {fold_values}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PJW 정제본을 DNN NumPy 배열로 변환")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"PJW 정제 파일이 없습니다: {input_path}")
    if "pmh" in input_path.name.lower() or "pmh" in str(input_path.parent).lower():
        raise ValueError("PMH 산출물은 이번 DNN 학습 입력으로 사용할 수 없습니다.")

    output_files = {
        "X": output_dir / "X.npy",
        "y": output_dir / "y.npy",
        "split": output_dir / "split.npy",
        "metadata": output_dir / "preprocessing_metadata.json",
        "rows": output_dir / "row_metadata.parquet",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"기존 산출물이 있습니다. --overwrite 필요: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    plan, excluded = build_feature_plan(columns)
    feature_names = [output for output, _, _ in plan]
    source_features = list(dict.fromkeys(source for _, source, _ in plan))
    usecols = list(
        dict.fromkeys([ID_COLUMN, DATE_COLUMN, SPLIT_COLUMN, TARGET_COLUMN, *source_features])
    )

    dtype: dict[str, str] = {
        column: "float32"
        for column in source_features
        if column not in CATEGORICAL_COLUMNS and column != DATE_COLUMN
    }
    dtype.update({column: "string" for column in CATEGORICAL_COLUMNS})
    dtype[ID_COLUMN] = "string"
    dtype[DATE_COLUMN] = "string"
    dtype[SPLIT_COLUMN] = "uint8"
    dtype[TARGET_COLUMN] = "uint8"

    print(f"PJW CSV 로딩: {input_path}", flush=True)
    print(f"DNN 피처: {len(feature_names)}개", flush=True)
    frame = pd.read_csv(input_path, usecols=usecols, dtype=dtype, low_memory=False)
    row_count = len(frame)
    target = frame[TARGET_COLUMN].to_numpy(dtype=np.uint8, copy=True)
    split = frame[SPLIT_COLUMN].to_numpy(dtype=np.uint8, copy=True)
    validate_split_and_target(split, target)
    train_mask = np.isin(split, [0, 1, 2])
    validation_mask = split == 3
    test_mask = split == 4

    features = np.empty((row_count, len(plan)), dtype=np.float32)
    category_mappings: dict[str, dict[str, int]] = {}
    unknown_counts: dict[str, int] = {}
    for index, (output_name, source_name, method) in enumerate(plan):
        if method == "month_index":
            features[:, index] = snapshot_to_month_index(frame[source_name])
        elif method == "train_only_category_encoding":
            encoded, mapping, unknown_count = encode_categorical_train_only(
                frame[source_name], train_mask
            )
            features[:, index] = encoded
            category_mappings[source_name] = mapping
            unknown_counts[source_name] = unknown_count
        else:
            features[:, index] = frame[source_name].to_numpy(dtype=np.float32, copy=False)

    features, imputation_medians, imputation_counts = impute_nonfinite_train_median(
        features, train_mask
    )
    if not np.isfinite(features).all():
        raise ValueError("결측치 처리 후 DNN 입력에 NaN/Inf가 남아 있습니다.")

    scaler = StandardScaler()
    print(f"StandardScaler fit: train {int(train_mask.sum()):,}행만 사용", flush=True)
    scaler.fit(features[train_mask])
    features = scaler.transform(features).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("스케일링 후 DNN 입력에 NaN/Inf가 있습니다.")

    np.save(output_files["X"], features, allow_pickle=False)
    np.save(output_files["y"], target, allow_pickle=False)
    np.save(output_files["split"], split, allow_pickle=False)
    row_metadata = pd.DataFrame(
        {
            "row_id": np.arange(row_count, dtype=np.int64),
            ID_COLUMN: frame[ID_COLUMN].astype("string"),
            DATE_COLUMN: frame[DATE_COLUMN].astype("string"),
            SPLIT_COLUMN: split,
        }
    )
    row_metadata.to_parquet(output_files["rows"], index=False)

    input_hash = sha256_file(input_path)
    split_counts = {
        "train_fold_0_2": int(train_mask.sum()),
        "validation_fold_3": int(validation_mask.sum()),
        "test_fold_4": int(test_mask.sum()),
    }
    metadata = {
        "schema_version": 2,
        "source_pipeline": "src/project_2nd/preprocessing_dataset_pjw",
        "excluded_pipeline": "src/project_2nd/preprocessing_dataset_pmh",
        "source_path": str(input_path),
        "source_size_bytes": input_path.stat().st_size,
        "source_sha256": input_hash,
        "feature_policy": FEATURE_POLICY,
        "row_count": row_count,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_plan": [
            {"output": output, "source": source, "method": method}
            for output, source, method in plan
        ],
        "excluded_columns": excluded,
        "target_name": TARGET_COLUMN,
        "split_name": SPLIT_COLUMN,
        "split_definition": {"train": [0, 1, 2], "validation": [3], "test": [4]},
        "split_counts": split_counts,
        "class_counts": {
            "negative_0": int((target == 0).sum()),
            "positive_1": int((target == 1).sum()),
        },
        "positive_rate": float(target.mean()),
        "category_mappings_fit_on": "fold 0,1,2 only",
        "category_mappings": category_mappings,
        "category_unknown_counts_all_rows": unknown_counts,
        "imputation": {
            "method": "train median per feature",
            "fit_on": "fold 0,1,2 only",
            "median": imputation_medians,
            "nonfinite_counts_all_rows": imputation_counts,
        },
        "scaler": {
            "type": "StandardScaler",
            "fit_on": "fold 0,1,2 only",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "array_dtypes": {
            "X": str(features.dtype),
            "y": str(target.dtype),
            "split": str(split.dtype),
        },
    }
    with output_files["metadata"].open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("PJW DNN 입력 준비 완료", flush=True)
    print(f"  출력: {output_dir}", flush=True)
    print(f"  행/피처: {row_count:,} / {len(feature_names)}", flush=True)
    print(f"  분할: {split_counts}", flush=True)
    print(f"  원본 SHA256: {input_hash}", flush=True)


if __name__ == "__main__":
    main()
