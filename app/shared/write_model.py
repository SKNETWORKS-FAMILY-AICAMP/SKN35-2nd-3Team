# -*- coding: utf-8 -*-
"""
models/predictions 테이블 write 헬퍼 라이브러리. app/shared/ 밑에서
write_user.py, write_prediction.py 등과 동일한 컨벤션(from .db import get_engine,
상대 import)으로 동작함 — 그래서 이 파일 자체는 직접 실행하지 않고 항상
import해서 씀 (직접 실행하면 상대 import가 깨짐, app.py 등과 동일한 이유).

실제로 "마지막에 실행하는" 스크립트는 프로젝트 루트의 load_models_and_predictions.py임
— 그게 이 파일의 함수들을 shared.write_model로 import해서 씀.

=== 1단계: 각자 모델/SHAP 만드는 쪽에서 (DB 접속 불필요) ===

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
    from shared.write_model import save_model_json

    save_model_json("data/features/model_registration.json", model_id=..., ...)

    (export_model_dl_tm.py / export_model_pjw.py가 이미 이 방식으로 만들어져 있음)

=== 2단계: 마지막에 (DB 담당) ===

    python load_models_and_predictions.py \\
        --model-json data/features/model_registration.json \\
        --predictions-json data/features/predictions_for_db.json
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sqlalchemy import text

from .db import get_engine


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


def _insert_one_model(engine, entry: dict, demote_others: bool = True) -> None:
    with engine.begin() as conn:
        conn.execute(_UPSERT_MODEL_SQL, entry)
        if entry.get("is_production") and demote_others:
            conn.execute(_DEMOTE_OTHERS_SQL, {"model_id": entry["model_id"]})


def register_model(model_id, model_name, version, model_type, metrics=None,
                    trained_at=None, is_production=False, demote_others=True) -> None:
    """JSON 안 거치고 바로 DB에 모델 1건 등록하고 싶을 때 쓰는 대안 함수."""
    engine = get_engine()
    entry = _build_model_dict(model_id, model_name, version, model_type, metrics, trained_at, is_production)
    _insert_one_model(engine, entry, demote_others=demote_others)
    print(f"models 테이블에 '{model_id}' 등록 완료" + (" (프로덕션으로 지정됨)" if is_production else ""))


def load_models_json_to_db(json_path: Union[str, Path]) -> None:
    """model_registration.json -> models 테이블."""
    engine = get_engine()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]

    for entry in entries:
        _insert_one_model(engine, entry, demote_others=True)
        print(f"  [models] '{entry['model_id']}' 적재 완료" + (" (프로덕션)" if entry.get("is_production") else ""))
    print(f"  -> {json_path}: {len(entries)}건")


_VALID_PROMOTE_METRICS = ("roc_auc", "accuracy", "precision_score", "recall_score", "f1_score")


def promote_best_model(metric: str = "roc_auc") -> Optional[str]:
    """DB에 등록된 모든 모델 중 metric 값이 가장 높은 모델을 자동으로 is_production=True로 지정.

    사람이 매번 is_production을 수동으로 정해서 넘기지 않아도, 이 함수를 한 번
    호출하면 그 시점 기준 "제일 성능 좋은 모델"이 자동으로 프로덕션이 됨
    (나머지는 자동으로 False로 내려감). NULL인 모델은 비교 대상에서 제외.

    Args:
        metric: 비교 기준 컬럼. roc_auc / accuracy / precision_score /
                recall_score / f1_score 중 하나 (기본 roc_auc).

    Returns:
        프로덕션으로 지정된 model_id. 비교할 모델이 하나도 없으면(전부 NULL) None.
    """
    if metric not in _VALID_PROMOTE_METRICS:
        raise ValueError(f"metric은 {_VALID_PROMOTE_METRICS} 중 하나여야 함 (받은 값: {metric!r})")

    engine = get_engine()
    with engine.connect() as conn:
        best = conn.execute(text(
            f"SELECT model_id, {metric} FROM models "
            f"WHERE {metric} IS NOT NULL ORDER BY {metric} DESC LIMIT 1"
        )).fetchone()

    if best is None:
        print(f"promote_best_model: {metric} 값이 있는 모델이 하나도 없어서 아무것도 안 함")
        return None

    best_model_id, best_value = best
    with engine.begin() as conn:
        conn.execute(text("UPDATE models SET is_production = FALSE WHERE model_id != :model_id"),
                     {"model_id": best_model_id})
        conn.execute(text("UPDATE models SET is_production = TRUE WHERE model_id = :model_id"),
                     {"model_id": best_model_id})

    print(f"자동 선정: '{best_model_id}' ({metric}={best_value}) -> is_production=True로 지정, 나머지는 False")

    # 안전장치: 선정된 모델의 predictions가 실제로 DB에 있는지 확인
    # (성능 지표만 보고 골랐는데 그 모델 predictions가 아직 안 올라와 있으면
    #  화면에 아무 데이터도 안 뜨는 사고가 남 — 그 전에 경고)
    with engine.connect() as conn:
        pred_count = conn.execute(
            text("SELECT COUNT(*) FROM predictions WHERE model_id = :model_id"),
            {"model_id": best_model_id},
        ).scalar()
    if not pred_count:
        print(f"⚠ 경고: '{best_model_id}'는 predictions 테이블에 데이터가 0건입니다. "
              f"이 모델용 predictions.json을 아직 적재 안 했다면, 지금 화면엔 이 모델 데이터가 "
              f"하나도 안 뜹니다 — load_predictions_json_to_db()로 먼저 적재하세요.")
    else:
        print(f"  (predictions 테이블에 이 모델 데이터 {pred_count:,}건 확인됨 — 정상)")

    return best_model_id


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
    engine = get_engine()
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