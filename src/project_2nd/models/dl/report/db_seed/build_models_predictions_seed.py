"""최종 DNN·SHAP 결과를 models 1건과 predictions 2건 적재 자료로 변환한다.

TiDB 접속 없이도 검토할 수 있도록 CSV, SQL, Markdown 미리보기와 검증 JSON을
생성한다. predictions의 두 행은 SHAP 표본 중 위험점수가 가장 높은 두 행이며,
각 행에는 절대 SHAP 값 기준 상위 5개 피처가 포함된다.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
DL_DIR = OUTPUT_DIR.parents[1]
REPO_ROOT = DL_DIR.parents[3]
RUN_DIR = (
    DL_DIR
    / "26.08.28-dl-e100-p5"
    / "DNN_20260828_094006_7e84a4a_pjw_official"
)
MODELING_DATASET = REPO_ROOT / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
STORES_DATASET = REPO_ROOT / "data" / "features" / "stores.csv"
INDUSTRIES_DATASET = REPO_ROOT / "data" / "features" / "industries.csv"

# TiDB의 실제 information_schema를 읽기 전용으로 조회해 확인한 적재 컬럼이다.
# predictions.prediction_id는 BIGINT AUTO_INCREMENT이므로 입력 파일에서 제외한다.
MODEL_DB_COLUMNS = [
    "model_id",
    "model_name",
    "version",
    "model_type",
    "accuracy",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc",
    "trained_at",
    "is_production",
]
PREDICTION_DB_COLUMNS = [
    "model_id",
    "user_id",
    "query_type",
    "store_id",
    "query_lat",
    "query_lng",
    "industry_code",
    "score",
    "shap_top_features",
    "created_at",
]
TIDB_COLUMN_TYPES = {
    "models": {
        "model_id": "varchar(50)",
        "model_name": "varchar(100)",
        "version": "varchar(20)",
        "model_type": "enum('ML','DL')",
        "accuracy": "decimal(6,5)",
        "precision_score": "decimal(6,5)",
        "recall_score": "decimal(6,5)",
        "f1_score": "decimal(6,5)",
        "roc_auc": "decimal(6,5)",
        "trained_at": "datetime",
        "is_production": "tinyint(1)",
    },
    "predictions": {
        "prediction_id": "bigint AUTO_INCREMENT",
        "model_id": "varchar(50)",
        "user_id": "varchar(30) NULL",
        "query_type": "enum('existing_store','new_location')",
        "store_id": "varchar(30) NULL",
        "query_lat": "decimal(10,7) NULL",
        "query_lng": "decimal(10,7) NULL",
        "industry_code": "varchar(20)",
        "score": "decimal(6,5)",
        "shap_top_features": "json NULL",
        "created_at": "datetime",
    },
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def sql_string(value: object) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def sql_decimal(value: float) -> str:
    return f"{value:.5f}"


def normalize_snapshot(value: object) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def fits_decimal(value: object, precision: int, scale: int) -> bool:
    """값이 DECIMAL(precision, scale)에 손실 없이 들어가는지 확인한다."""
    try:
        decimal_value = Decimal(str(value))
        quantized = decimal_value.quantize(Decimal(1).scaleb(-scale))
    except (InvalidOperation, ValueError):
        return False
    integer_digits = precision - scale
    limit = Decimal(10) ** integer_digits
    return -limit < quantized < limit


def is_datetime_text(value: object) -> bool:
    try:
        datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return True


def top_five_shap(row: dict[str, Any]) -> list[dict[str, object]]:
    contributions = [*row["top_risk_up"], *row["top_risk_down"]]
    contributions.sort(key=lambda item: abs(float(item["shap_value"])), reverse=True)
    return [
        {
            "feature": str(item["feature"]),
            "shap_value": round(float(item["shap_value"]), 8),
            "feature_value": round(float(item["feature_value"]), 8),
        }
        for item in contributions[:5]
    ]


def find_industries(selected: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    targets = {
        (str(row["store_id"]), normalize_snapshot(row["snapshot_date"]))
        for row in selected
    }
    found: dict[tuple[str, str], str] = {}
    for chunk in pd.read_csv(
        MODELING_DATASET,
        usecols=["store_id", "snapshot_date", "industry_code"],
        dtype="string",
        chunksize=200_000,
        low_memory=False,
    ):
        chunk["snapshot_date"] = chunk["snapshot_date"].map(normalize_snapshot)
        keys = list(zip(chunk["store_id"].astype(str), chunk["snapshot_date"].astype(str)))
        mask = pd.Series([key in targets for key in keys], index=chunk.index)
        for store_id, snapshot_date, industry_code in chunk.loc[
            mask, ["store_id", "snapshot_date", "industry_code"]
        ].itertuples(index=False, name=None):
            key = (str(store_id), str(snapshot_date))
            previous = found.get(key)
            if previous is not None and previous != str(industry_code):
                raise ValueError(f"동일 매장·시점에 업종이 여러 개입니다: {key}")
            found[key] = str(industry_code)
        if len(found) == len(targets):
            break
    missing = sorted(targets.difference(found))
    if missing:
        raise ValueError(f"모델링 데이터에서 업종을 찾지 못했습니다: {missing}")
    return found


def build_rows() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    manifest = load_json(RUN_DIR / "run_manifest.json")
    metrics = load_json(RUN_DIR / "metrics" / "metrics.json")
    shap_metadata = load_json(RUN_DIR / "shap" / "shap_metadata.json")
    shap_rows = load_json(RUN_DIR / "shap" / "shap_local_top_features.json")
    selected = sorted(
        shap_rows,
        key=lambda row: float(row["risk_score_uncalibrated"]),
        reverse=True,
    )[:2]
    industries = find_industries(selected)

    model_id = str(manifest["run_id"])
    model = {
        "model_id": model_id,
        "model_name": "PJW DNN 34F E100 P5",
        "version": "2026.08.28-e100p5",
        "model_type": "DL",
        "accuracy": round(float(metrics["test"]["accuracy"]), 5),
        "precision_score": round(float(metrics["test"]["precision"]), 5),
        "recall_score": round(float(metrics["test"]["recall"]), 5),
        "f1_score": round(float(metrics["test"]["f1"]), 5),
        "roc_auc": round(float(metrics["test"]["roc_auc"]), 5),
        "trained_at": datetime.fromisoformat(str(manifest["created_at_kst"])).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        # TiDB 타입이 TINYINT(1)이므로 CSV에도 문자열 False가 아닌 정수 0을 쓴다.
        "is_production": 0,
    }
    if len(model_id) > 50:
        raise ValueError("model_id가 models.model_id VARCHAR(50)을 초과합니다.")

    prediction_created_at = datetime.fromisoformat(
        str(shap_metadata["created_at_kst"])
    ).strftime("%Y-%m-%d %H:%M:%S")
    predictions: list[dict[str, object]] = []
    for row in selected:
        store_id = str(row["store_id"])
        snapshot_date = normalize_snapshot(row["snapshot_date"])
        industry_code = industries[(store_id, snapshot_date)]
        shap_features = top_five_shap(row)
        predictions.append(
            {
                "model_id": model_id,
                "user_id": None,
                "query_type": "existing_store",
                "store_id": store_id,
                "query_lat": None,
                "query_lng": None,
                "industry_code": industry_code,
                "score": round(float(row["risk_score_uncalibrated"]), 5),
                "shap_top_features": json.dumps(
                    shap_features, ensure_ascii=False, separators=(",", ":")
                ),
                "created_at": prediction_created_at,
                "source_snapshot_date": snapshot_date,
                "y_true": int(row["y_true"]),
            }
        )

    store_master = set(
        pd.read_csv(STORES_DATASET, usecols=["store_id"], dtype="string")["store_id"]
        .dropna()
        .astype(str)
    )
    industry_master = set(
        pd.read_csv(INDUSTRIES_DATASET, usecols=["industry_code"], dtype="string")[
            "industry_code"
        ]
        .dropna()
        .astype(str)
    )
    model_types_ok = (
        len(str(model["model_id"])) <= 50
        and len(str(model["model_name"])) <= 100
        and len(str(model["version"])) <= 20
        and model["model_type"] in {"ML", "DL"}
        and all(
            fits_decimal(model[column], 6, 5)
            for column in [
                "accuracy",
                "precision_score",
                "recall_score",
                "f1_score",
                "roc_auc",
            ]
        )
        and is_datetime_text(model["trained_at"])
        and model["is_production"] in {0, 1}
    )
    prediction_types_ok = all(
        len(str(row["model_id"])) <= 50
        and row["user_id"] is None
        and row["query_type"] in {"existing_store", "new_location"}
        and (row["store_id"] is None or len(str(row["store_id"])) <= 30)
        and row["query_lat"] is None
        and row["query_lng"] is None
        and len(str(row["industry_code"])) <= 20
        and fits_decimal(row["score"], 6, 5)
        and isinstance(json.loads(str(row["shap_top_features"])), list)
        and is_datetime_text(row["created_at"])
        for row in predictions
    )
    validation = {
        "models_rows": 1,
        "predictions_rows": len(predictions),
        "verified_database": "seoul_market",
        "verified_tidb_version": "8.0.11-TiDB-v8.5.3-serverless",
        "tidb_schema_checked_read_only": True,
        "models_column_types": TIDB_COLUMN_TYPES["models"],
        "predictions_column_types": TIDB_COLUMN_TYPES["predictions"],
        "models_csv_columns": MODEL_DB_COLUMNS,
        "predictions_csv_columns": PREDICTION_DB_COLUMNS,
        "prediction_id_omitted_because_auto_increment": True,
        "models_values_match_column_types": model_types_ok,
        "predictions_values_match_column_types": prediction_types_ok,
        "shap_non_null_rows": sum(bool(row["shap_top_features"]) for row in predictions),
        "shap_features_per_prediction": [
            len(json.loads(str(row["shap_top_features"]))) for row in predictions
        ],
        "model_id_length_ok": len(model_id) <= 50,
        "store_foreign_keys_exist_locally": all(
            str(row["store_id"]) in store_master for row in predictions
        ),
        "industry_foreign_keys_exist_locally": all(
            str(row["industry_code"]) in industry_master for row in predictions
        ),
        "scores_fit_decimal_6_5": all(
            fits_decimal(row["score"], 6, 5) for row in predictions
        ),
        "database_insert_executed": False,
        "database_insert_note": "TiDB는 스키마 확인용 SELECT만 실행했고 적재는 수행하지 않음",
    }
    if len(predictions) != 2 or validation["shap_non_null_rows"] != 2:
        raise ValueError("predictions 2건 모두에 SHAP 결과가 있어야 합니다.")
    if not validation["store_foreign_keys_exist_locally"]:
        raise ValueError("stores 로컬 마스터에서 store_id를 찾지 못했습니다.")
    if not validation["industry_foreign_keys_exist_locally"]:
        raise ValueError("industries 로컬 마스터에서 industry_code를 찾지 못했습니다.")
    if not model_types_ok or not prediction_types_ok:
        raise ValueError("생성 값 중 실제 TiDB 컬럼 타입에 맞지 않는 값이 있습니다.")
    return model, predictions, validation


def write_csv(model: dict[str, object], predictions: list[dict[str, object]]) -> None:
    # 적재용 CSV에는 실제 테이블 컬럼만 기록한다. 분석 검증용 y_true와
    # source_snapshot_date는 DB 스키마에 없으므로 Markdown 보고서에만 남긴다.
    pd.DataFrame([model], columns=MODEL_DB_COLUMNS).to_csv(
        OUTPUT_DIR / "models_result_1row.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(predictions, columns=PREDICTION_DB_COLUMNS).to_csv(
        OUTPUT_DIR / "predictions_shap_result_2rows.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_sql(model: dict[str, object], predictions: list[dict[str, object]]) -> None:
    model_columns = (
        "model_id, model_name, version, model_type, accuracy, precision_score, "
        "recall_score, f1_score, roc_auc, trained_at, is_production"
    )
    model_values = ", ".join(
        [
            sql_string(model["model_id"]),
            sql_string(model["model_name"]),
            sql_string(model["version"]),
            sql_string(model["model_type"]),
            sql_decimal(float(model["accuracy"])),
            sql_decimal(float(model["precision_score"])),
            sql_decimal(float(model["recall_score"])),
            sql_decimal(float(model["f1_score"])),
            sql_decimal(float(model["roc_auc"])),
            sql_string(model["trained_at"]),
            str(int(model["is_production"])),
        ]
    )
    sql_lines = [
        "-- schema.sql의 models 1건 + predictions(SHAP 포함) 2건 적재 자료",
        "-- predictions는 SHAP 표본 중 위험점수가 높은 두 매장을 사용했다.",
        "START TRANSACTION;",
        "",
        f"INSERT INTO models ({model_columns})",
        f"VALUES ({model_values})",
        "ON DUPLICATE KEY UPDATE",
        "    model_name = VALUES(model_name),",
        "    version = VALUES(version),",
        "    model_type = VALUES(model_type),",
        "    accuracy = VALUES(accuracy),",
        "    precision_score = VALUES(precision_score),",
        "    recall_score = VALUES(recall_score),",
        "    f1_score = VALUES(f1_score),",
        "    roc_auc = VALUES(roc_auc),",
        "    trained_at = VALUES(trained_at),",
        "    is_production = VALUES(is_production);",
        "",
    ]
    columns = (
        "model_id, user_id, query_type, store_id, query_lat, query_lng, "
        "industry_code, score, shap_top_features, created_at"
    )
    for index, row in enumerate(predictions, start=1):
        values = ", ".join(
            [
                sql_string(row["model_id"]),
                "NULL",
                sql_string(row["query_type"]),
                sql_string(row["store_id"]),
                "NULL",
                "NULL",
                sql_string(row["industry_code"]),
                sql_decimal(float(row["score"])),
                sql_string(row["shap_top_features"]),
                sql_string(row["created_at"]),
            ]
        )
        sql_lines.extend(
            [
                f"-- prediction {index}",
                f"INSERT INTO predictions ({columns})",
                f"SELECT {values}",
                "WHERE NOT EXISTS (",
                "    SELECT 1 FROM predictions",
                f"    WHERE model_id = {sql_string(row['model_id'])}",
                f"      AND query_type = {sql_string(row['query_type'])}",
                f"      AND store_id = {sql_string(row['store_id'])}",
                f"      AND industry_code = {sql_string(row['industry_code'])}",
                f"      AND created_at = {sql_string(row['created_at'])}",
                ");",
                "",
            ]
        )
    sql_lines.extend(["COMMIT;", ""])
    (OUTPUT_DIR / "models_predictions_shap_seed.sql").write_text(
        "\n".join(sql_lines), encoding="utf-8"
    )


def write_preview(
    model: dict[str, object],
    predictions: list[dict[str, object]],
    validation: dict[str, object],
) -> None:
    prediction_lines = []
    shap_sections = []
    for index, row in enumerate(predictions, start=1):
        features = json.loads(str(row["shap_top_features"]))
        prediction_lines.append(
            f"| {index} | `{row['store_id']}` | `{row['industry_code']}` | "
            f"{float(row['score']):.5f} | {row['y_true']} | {len(features)}개 |"
        )
        feature_lines = "\n".join(
            f"| {rank} | `{item['feature']}` | {item['feature_value']:.5f} | "
            f"{item['shap_value']:+.6f} |"
            for rank, item in enumerate(features, start=1)
        )
        shap_sections.append(
            f"### 예측 {index}: `{row['store_id']}`\n\n"
            "| 순위 | 피처 | 피처값 | SHAP 값 |\n"
            "|---:|---|---:|---:|\n"
            f"{feature_lines}"
        )

    preview = f"""# models·predictions SHAP 적재 결과

## 처리 순서

1. 최종 DNN의 기존 SHAP 1,000건을 위험점수 순으로 정렬
2. 상위 예측 2건에서 절대 SHAP 값 기준 상위 5개 피처 추출
3. `models` 1건과 `predictions` 2건을 `schema.sql` 컬럼 형식으로 변환
4. 로컬 `stores.csv`, `industries.csv`로 외래키 존재 여부 검증
5. 실제 TiDB 스키마를 읽기 전용으로 확인하고 컬럼명·자료형 검증
6. DB 적재 없이 실행 가능한 SQL·CSV 생성

외래키 때문에 실제 SQL 실행 순서는 `models → predictions`이지만, predictions에 넣을 SHAP 데이터는 먼저 준비했다.

## models 결과 1건

| model_id | 유형 | 버전 | Accuracy | Precision | Recall | F1 | ROC-AUC | 운영 모델 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `{model['model_id']}` | {model['model_type']} | `{model['version']}` | {float(model['accuracy']):.5f} | {float(model['precision_score']):.5f} | {float(model['recall_score']):.5f} | {float(model['f1_score']):.5f} | {float(model['roc_auc']):.5f} | FALSE |

## predictions 결과 2건

`score`는 보정된 폐업 확률이 아니라 DNN의 미보정 폐업 위험점수다.

| 번호 | store_id | industry_code | score | 실제 타깃 | SHAP 피처 |
|---:|---|---|---:|---:|---:|
{chr(10).join(prediction_lines)}

## SHAP 상세

{chr(10).join(shap_sections)}

## 검증 결과

| 항목 | 결과 |
|---|---:|
| models 행 | {validation['models_rows']} |
| predictions 행 | {validation['predictions_rows']} |
| SHAP 비어 있지 않은 행 | {validation['shap_non_null_rows']} |
| 예측별 SHAP 피처 수 | {validation['shap_features_per_prediction']} |
| models 값·타입 일치 | {validation['models_values_match_column_types']} |
| predictions 값·타입 일치 | {validation['predictions_values_match_column_types']} |
| prediction_id 자동 증가로 입력 제외 | {validation['prediction_id_omitted_because_auto_increment']} |
| 로컬 stores FK 확인 | {validation['store_foreign_keys_exist_locally']} |
| 로컬 industries FK 확인 | {validation['industry_foreign_keys_exist_locally']} |
| TiDB INSERT 실행 | {validation['database_insert_executed']} |

## 실제 TiDB 컬럼 타입

### models

| 컬럼 | 타입 |
|---|---|
{chr(10).join(f"| `{column}` | `{column_type}` |" for column, column_type in TIDB_COLUMN_TYPES['models'].items())}

### predictions

| 컬럼 | 타입 |
|---|---|
{chr(10).join(f"| `{column}` | `{column_type}` |" for column, column_type in TIDB_COLUMN_TYPES['predictions'].items())}

`prediction_id`는 `BIGINT AUTO_INCREMENT`이므로 SQL과 CSV에서 입력하지 않는다.

## 생성 파일

- `models_result_1row.csv`
- `predictions_shap_result_2rows.csv`
- `models_predictions_shap_seed.sql`
- `schema_validation.json`
"""
    (OUTPUT_DIR / "DB_models_predictions_SHAP_2건_결과.md").write_text(
        preview, encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, predictions, validation = build_rows()
    write_csv(model, predictions)
    write_sql(model, predictions)
    write_preview(model, predictions, validation)
    with (OUTPUT_DIR / "schema_validation.json").open("w", encoding="utf-8") as file:
        json.dump(validation, file, ensure_ascii=False, indent=2)
    print(f"models 1건, predictions+SHAP 2건 생성 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
