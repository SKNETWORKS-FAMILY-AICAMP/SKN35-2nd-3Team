"""PMH 전처리 CSV를 누수 방지된 DNN 배열로 변환한다.

핵심 원칙
---------
1. 타깃, fold, 매장 식별자, 미래 라벨(`transitioned_next`)은 입력하지 않는다.
2. 문자열 범주형 대신 PMH가 생성한 ``*_enc`` 열만 사용한다.
3. StandardScaler는 train(fold 0~2)에만 fit한 뒤 validation/test에 적용한다.
4. CSV 행과 실제 매장을 다시 연결할 수 있도록 row metadata를 함께 저장한다.
5. 날짜 ``202312``를 단순 정수로 쓰지 않고 최초 스냅샷부터의 개월 수로 바꾼다.

기본 ``time_safe`` 정책은 인수인계서에서 시간 누수 가능성이 있다고 지적한
피처도 제외한다. 공식 파이프라인 수치를 그대로 재현할 필요가 있을 때만
``--feature-policy official``을 명시한다.
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
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "modeling_dataset_preprocessed_pmh.csv"

TARGET_COLUMN = "is_closed_next"
SPLIT_COLUMN = "fold"
ID_COLUMN = "store_id"
DATE_COLUMN = "snapshot_date"

RAW_CATEGORICAL_COLUMNS = (
    "industry_dae_code",
    "industry_group",
    "industry_jung_code",
    "industry_jung_name",
    "industry_code",
    "industry_name",
    "gu_name",
    "dong_code",
    "floor_category",
)

# 어떤 실험에서도 입력하면 안 되는 열이다. transitioned_next는 예측하려는
# 다음 시점에서만 알 수 있으므로 명백한 타깃 누수다.
HARD_EXCLUDED_COLUMNS = frozenset(
    {
        TARGET_COLUMN,
        SPLIT_COLUMN,
        ID_COLUMN,
        "transitioned_next",
        *RAW_CATEGORICAL_COLUMNS,
    }
)

# 인수인계서가 지적한 "fold-safe일 수는 있으나 time-safe라고 보장할 수 없는"
# 열이다. 안전한 기본 실험에서는 제외하고, official 정책에서만 포함한다.
TIME_UNSAFE_COLUMNS = frozenset(
    {
        "previously_transitioned",
        "keyword_growth_score",
        "industry_historical_rate",
        "dong_historical_rate",
        "dong_industry_historical_rate",
        "korean_pop",
        "foreign_long_pop",
        "foreign_short_pop",
        "total_pop_avg",
        "foreign_short_ratio",
        "tourist_zone_candidate",
        "population_is_proxied",
        "dong_industry_count_growth",
    }
)

REQUIRED_COLUMNS = frozenset({TARGET_COLUMN, SPLIT_COLUMN, ID_COLUMN, DATE_COLUMN})
VALID_FOLDS = frozenset({0, 1, 2, 3, 4})


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """대용량 파일을 메모리에 한꺼번에 올리지 않고 SHA256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_feature_plan(
    columns: Sequence[str], feature_policy: str
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """``(출력 피처명, 원본 열명)`` 목록과 제외 사유를 반환한다."""

    if feature_policy not in {"official", "time_safe"}:
        raise ValueError("feature_policy는 official 또는 time_safe여야 합니다.")

    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")

    plan: list[tuple[str, str]] = []
    excluded: dict[str, str] = {}
    for column in columns:
        if column in HARD_EXCLUDED_COLUMNS:
            excluded[column] = "target/split/id/future-label/original-category"
            continue
        if feature_policy == "time_safe" and column in TIME_UNSAFE_COLUMNS:
            excluded[column] = "potential-time-leakage"
            continue
        if column == DATE_COLUMN:
            plan.append(("snapshot_month_index", DATE_COLUMN))
        else:
            plan.append((column, column))

    if not plan:
        raise ValueError("선택된 DNN 입력 피처가 없습니다.")
    return plan, excluded


def snapshot_to_month_index(values: Iterable[object]) -> np.ndarray:
    """YYYYMM 스냅샷을 최초 시점 기준 0, 6, 12... 개월로 변환한다."""

    text = pd.Series(values, dtype="string").str.replace(r"\.0$", "", regex=True)
    valid = text.str.fullmatch(r"\d{6}")
    if not bool(valid.all()):
        bad = text.loc[~valid].head(5).tolist()
        raise ValueError(f"snapshot_date 형식은 YYYYMM이어야 합니다. 예: {bad}")

    year = text.str[:4].astype(np.int32)
    month = text.str[4:6].astype(np.int16)
    if not bool(month.between(1, 12).all()):
        raise ValueError("snapshot_date에 1~12가 아닌 월이 있습니다.")
    absolute = year.to_numpy(dtype=np.int32) * 12 + month.to_numpy(dtype=np.int32) - 1
    return (absolute - absolute.min()).astype(np.float32)


def validate_split_and_target(split: np.ndarray, target: np.ndarray) -> None:
    """학습/검증/테스트 계약과 이진 타깃을 검사한다."""

    target_values = set(np.unique(target).tolist())
    if not target_values.issubset({0, 1}) or target_values != {0, 1}:
        raise ValueError(f"타깃은 0과 1을 모두 포함해야 합니다: {target_values}")
    fold_values = set(np.unique(split).tolist())
    if fold_values != VALID_FOLDS:
        raise ValueError(f"fold는 정확히 0~4여야 합니다: {fold_values}")
    if not np.isin(split, [0, 1, 2]).any() or not (split == 3).any() or not (split == 4).any():
        raise ValueError("train(0~2), validation(3), test(4)가 모두 필요합니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PMH CSV를 DNN용 NumPy 배열로 변환")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--feature-policy",
        choices=("time_safe", "official"),
        default="time_safe",
        help="기본값 time_safe: 시간 누수 의심 피처도 제외",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="기본값: data/processed/dnn_pmh_<feature-policy>",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else REPO_ROOT / "data" / "processed" / f"dnn_pmh_{args.feature_policy}"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"PMH 전처리 파일이 없습니다: {input_path}")

    output_files = {
        "X": output_dir / "X.npy",
        "y": output_dir / "y.npy",
        "split": output_dir / "split.npy",
        "metadata": output_dir / "preprocessing_metadata.json",
        "rows": output_dir / "row_metadata.parquet",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"기존 산출물이 있습니다. --overwrite가 필요합니다: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"입력 헤더 확인: {input_path}", flush=True)
    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    plan, excluded = build_feature_plan(columns, args.feature_policy)
    feature_names = [output for output, _ in plan]
    source_features = list(dict.fromkeys(source for _, source in plan))
    usecols = list(dict.fromkeys([ID_COLUMN, DATE_COLUMN, SPLIT_COLUMN, TARGET_COLUMN, *source_features]))

    dtype: dict[str, str] = {column: "float32" for column in source_features}
    dtype[ID_COLUMN] = "string"
    dtype[DATE_COLUMN] = "string"
    dtype[SPLIT_COLUMN] = "uint8"
    dtype[TARGET_COLUMN] = "uint8"

    print(f"CSV 로딩: 피처 {len(feature_names)}개, 정책={args.feature_policy}", flush=True)
    frame = pd.read_csv(input_path, usecols=usecols, dtype=dtype, low_memory=False)
    row_count = len(frame)
    if row_count == 0:
        raise ValueError("입력 데이터가 비어 있습니다.")

    target = frame[TARGET_COLUMN].to_numpy(dtype=np.uint8, copy=True)
    split = frame[SPLIT_COLUMN].to_numpy(dtype=np.uint8, copy=True)
    validate_split_and_target(split, target)

    features = np.empty((row_count, len(plan)), dtype=np.float32)
    for index, (output_name, source_name) in enumerate(plan):
        if source_name == DATE_COLUMN:
            features[:, index] = snapshot_to_month_index(frame[source_name])
        else:
            features[:, index] = frame[source_name].to_numpy(dtype=np.float32, copy=False)

    if not np.isfinite(features).all():
        bad_count = int((~np.isfinite(features)).sum())
        raise ValueError(f"스케일링 전 입력에 NaN/Inf가 {bad_count:,}개 있습니다.")

    train_mask = np.isin(split, [0, 1, 2])
    validation_mask = split == 3
    test_mask = split == 4
    scaler = StandardScaler()
    print(f"StandardScaler fit: train {int(train_mask.sum()):,}행만 사용", flush=True)
    scaler.fit(features[train_mask])
    features = scaler.transform(features).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("스케일링 후 입력에 NaN/Inf가 있습니다.")

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
    class_counts = {
        "negative_0": int((target == 0).sum()),
        "positive_1": int((target == 1).sum()),
    }
    metadata = {
        "schema_version": 1,
        "source_path": str(input_path),
        "source_size_bytes": input_path.stat().st_size,
        "source_sha256": input_hash,
        "feature_policy": args.feature_policy,
        "row_count": row_count,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "excluded_columns": excluded,
        "target_name": TARGET_COLUMN,
        "split_name": SPLIT_COLUMN,
        "split_definition": {
            "train": [0, 1, 2],
            "validation": [3],
            "test": [4],
        },
        "split_counts": split_counts,
        "class_counts": class_counts,
        "positive_rate": float(target.mean()),
        "scaler": {
            "type": "StandardScaler",
            "fit_on": "fold 0,1,2 only",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        "array_dtypes": {"X": str(features.dtype), "y": str(target.dtype), "split": str(split.dtype)},
    }
    with output_files["metadata"].open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    del frame, features
    print("DNN 입력 준비 완료", flush=True)
    print(f"  출력: {output_dir}", flush=True)
    print(f"  행/피처: {row_count:,} / {len(feature_names)}", flush=True)
    print(f"  분할: {split_counts}", flush=True)
    print(f"  양성 비율: {metadata['positive_rate']:.6f}", flush=True)
    print(f"  원본 SHA256: {input_hash}", flush=True)


if __name__ == "__main__":
    main()
