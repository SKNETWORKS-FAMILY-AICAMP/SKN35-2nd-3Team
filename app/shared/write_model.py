# -*- coding: utf-8 -*-
"""
write_model.py — 최종 DB 적재 스크립트. 이 파일 하나만 실행하면
(model_registration.json + predictions.json) 쌍을 models/predictions
테이블에 한 번에 적재함. 팀원 여러 명이 각자 모델을 만들었으면(예: 태민님
dl_tm/, pjw님 official 파이프라인), 각자의 JSON 쌍을 인자로 여러 개
넘기면 한 번의 실행으로 전부 들어감.

=== 1단계: 각자 모델/SHAP 만드는 쪽에서 (DB 접속 불필요) ===

    from write_model import save_model_json
    save_model_json("model_registration.json", model_id=..., ...)

    # predictions는 predictions_csv_to_json.py로 CSV -> JSON 변환
    # (dl_score_tm.py/shap_explain_tm.py가 만든 predictions_for_db.csv 기준)

=== 2단계: 마지막에 이 스크립트 한 번 실행 (DB 담당) ===

    python write_model.py \\
        --model-json models/dl/dl_tm/model_registration.json \\
        --predictions-json data/features/predictions_dnn_tm.json \\
        --model-json models/dl/pjw_official/model_registration.json \\
        --predictions-json data/features/predictions_pjw.json

    (--model-json / --predictions-json 쌍을 순서대로 매칭해서, 모델 개수만큼
     반복해서 적재함 — 모델을 먼저 넣어야 predictions의 model_id FK가 통과되므로
     항상 "그 모델 -> 그 모델의 predictions" 순서로 처리함)

    predictions 없이 모델만 등록하고 싶으면 --predictions-json 없이
    --model-json만 줘도 됨.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sqlalchemy import text

_APP_DIR = str(Path(__file__).resolve().parent / "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

try:
    from shared.db import get_engine  # DB 적재(2단계) 시점에만 필요
except ImportError:
    get_engine = None  # 1단계(JSON 저장)만 쓸 때는 DB 연결 자체가 필요 없음


# =====================================================================
# 1단계: 모델 등록 정보를 JSON으로 저장 (모델 만드는 쪽에서 호출, DB 불필요)
# =====================================================================

def _build_model_dict(
    model_id: str,
    model_name: str,
    version: str,
    model_type: str,
    metrics: Optional[dict] = None,
    trained_at: Optional[str] = None,
    is_production: bool = False,
) -> dict:
    if model_type not in ("ML", "DL"):
        raise ValueError(f"model_type은 'ML' 또는 'DL'만 허용 (받은 값: {model_type!r})")
    metrics = metrics or {}
    return {
        "model_id": model_id,
        "model_name": model_name,
        "version": version,
        "model_type": model_type,
        "accuracy": metrics.get("accuracy"),
        "precision_score": metrics.get("precision_score"),
        "recall_score": metrics.get("recall_score"),
        "f1_score": metrics.get("f1_score"),
        "roc_auc": metrics.get("roc_auc"),
        "trained_at": trained_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_production": is_production,
    }


def save_model_json(
    path: Union[str, Path],
    model_id: str,
    model_name: str,
    version: str,
    model_type: str,
    metrics: Optional[dict] = None,
    trained_at: Optional[str] = None,
    is_production: bool = False,
    append: bool = False,
) -> None:
    """모델 등록 정보를 JSON 파일로 저장 (DB 접속 불필요 — 모델 학습하는 쪽에서 호출)."""
    entry = _build_model_dict(model_id, model_name, version, model_type, metrics, trained_at, is_production)
    path = Path(path)

    entries = []
    if append and path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        entries = existing if isinstance(existing, list) else [existing]
        entries = [e for e in entries if e.get("model_id") != model_id]

    entries.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"모델 등록 정보 저장 완료: {path} ({len(entries)}건)")


# =====================================================================
# 2단계: JSON -> DB 적재 (이 파일을 실행하는 쪽, DB 담당)
# =====================================================================

_UPSERT_MODEL_SQL = text("""
    INSERT INTO models (
        model_id, model_name, version, model_type,
        accuracy, precision_score, recall_score, f1_score, roc_auc,
        trained_at, is_production
    ) VALUES (
        :model_id, :model_name, :version, :model_type,
        :accuracy, :precision_score, :recall_score, :f1_score, :roc_auc,
        :trained_at, :is_production
    )
    ON DUPLICATE KEY UPDATE
        model_name = VALUES(model_name), version = VALUES(version),
        model_type = VALUES(model_type), accuracy = VALUES(accuracy),
        precision_score = VALUES(precision_score), recall_score = VALUES(recall_score),
        f1_score = VALUES(f1_score), roc_auc = VALUES(roc_auc),
        trained_at = VALUES(trained_at), is_production = VALUES(is_production)
""")

_DEMOTE_OTHERS_SQL = text("UPDATE models SET is_production = FALSE WHERE model_id != :model_id")

_INSERT_PREDICTION_SQL = text("""
    INSERT INTO predictions (
        model_id, user_id, query_type, store_id,
        query_lat, query_lng, industry_code, score, shap_top_features
    ) VALUES (
        :model_id, :user_id, :query_type, :store_id,
        :query_lat, :query_lng, :industry_code, :score, :shap_top_features
    )
""")


def _require_engine():
    if get_engine is None:
        raise RuntimeError("DB 적재 기능은 app/shared/db.py를 불러올 수 있는 환경에서만 동작함")
    return get_engine()


def _insert_one_model(engine, entry: dict, demote_others: bool = True) -> None:
    with engine.begin() as conn:
        conn.execute(_UPSERT_MODEL_SQL, entry)
        if entry.get("is_production") and demote_others:
            conn.execute(_DEMOTE_OTHERS_SQL, {"model_id": entry["model_id"]})


def register_model(model_id, model_name, version, model_type, metrics=None,
                    trained_at=None, is_production=False, demote_others=True) -> None:
    """JSON 안 거치고 바로 DB에 모델 1건 등록하고 싶을 때 쓰는 대안 함수."""
    engine = _require_engine()
    entry = _build_model_dict(model_id, model_name, version, model_type, metrics, trained_at, is_production)
    _insert_one_model(engine, entry, demote_others=demote_others)
    print(f"models 테이블에 '{model_id}' 등록 완료" + (" (프로덕션으로 지정됨)" if is_production else ""))


def load_models_json_to_db(json_path: Union[str, Path]) -> None:
    """model_registration.json -> models 테이블."""
    engine = _require_engine()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]

    for entry in entries:
        _insert_one_model(engine, entry, demote_others=True)
        print(f"  [models] '{entry['model_id']}' 적재 완료" + (" (프로덕션)" if entry.get("is_production") else ""))
    print(f"  -> {json_path}: {len(entries)}건")


def _none_if_nan(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _record_to_params(rec: dict) -> dict:
    shap_val = rec.get("shap_top_features")
    if shap_val is not None and not isinstance(shap_val, str):
        shap_val = json.dumps(shap_val, ensure_ascii=False)
    return {
        "model_id": rec["model_id"],
        "user_id": _none_if_nan(rec.get("user_id")),
        "query_type": rec["query_type"],
        "store_id": _none_if_nan(rec.get("store_id")),
        "query_lat": _none_if_nan(rec.get("query_lat")),
        "query_lng": _none_if_nan(rec.get("query_lng")),
        "industry_code": rec["industry_code"],
        "score": float(rec["score"]),
        "shap_top_features": _none_if_nan(shap_val),
    }


def _check_model_registered(engine, model_id: str) -> None:
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM models WHERE model_id = :model_id"),
            {"model_id": model_id},
        ).scalar()
    if not exists:
        raise SystemExit(
            f"\n중단: model_id='{model_id}'가 models 테이블에 없습니다. "
            f"이 predictions.json에 대응하는 --model-json을 먼저(또는 같이) 넘겼는지 확인하세요.\n"
        )


def load_predictions_json_to_db(json_path: Union[str, Path], chunk_size: int = 5000) -> None:
    """predictions.json -> predictions 테이블."""
    engine = _require_engine()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]

    for model_id in {r["model_id"] for r in records if r.get("model_id")}:
        _check_model_registered(engine, model_id)

    n = len(records)
    inserted = 0
    for start in range(0, n, chunk_size):
        chunk = records[start:start + chunk_size]
        params_list = [_record_to_params(r) for r in chunk]
        with engine.begin() as conn:
            conn.execute(_INSERT_PREDICTION_SQL, params_list)
        inserted += len(chunk)
        print(f"  [predictions] {inserted:,}/{n:,} 적재 완료")
    print(f"  -> {json_path}: {inserted:,}건")


# =====================================================================
# CLI: 모델 여러 개를 한 번의 실행으로 전부 적재
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="여러 모델의 (model_registration.json, predictions.json) 쌍을 한 번에 DB에 적재"
    )
    ap.add_argument("--model-json", action="append", default=[],
                     help="모델 등록 JSON 경로 (여러 모델이면 여러 번 반복)")
    ap.add_argument("--predictions-json", action="append", default=[],
                     help="predictions JSON 경로 (--model-json과 순서로 짝지어짐, 없으면 생략 가능)")
    ap.add_argument("--chunk-size", type=int, default=5000)
    args = ap.parse_args()

    if not args.model_json and not args.predictions_json:
        raise SystemExit("--model-json 또는 --predictions-json을 최소 1개 이상 넘겨야 함")

    n_models = len(args.model_json)
    n_preds = len(args.predictions_json)
    if n_preds > n_models:
        raise SystemExit("--predictions-json 개수가 --model-json 개수보다 많습니다 — 순서를 맞춰서 짝지어 주세요")

    for i in range(max(n_models, n_preds)):
        print(f"\n=== [{i+1}/{max(n_models, n_preds)}번째 모델] ===")
        if i < n_models:
            print(f"모델 등록: {args.model_json[i]}")
            load_models_json_to_db(args.model_json[i])
        if i < n_preds:
            print(f"predictions 적재: {args.predictions_json[i]}")
            load_predictions_json_to_db(args.predictions_json[i], chunk_size=args.chunk_size)

    print("\n전체 완료.")


if __name__ == "__main__":
    main()