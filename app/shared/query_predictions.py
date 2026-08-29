# -*- coding: utf-8 -*-
"""
app.py 렌더링 쪽에서 "지금 프로덕션 모델이 뭔지" + "그 모델 기준 예측값"을
쉽게 가져다 쓸 수 있게 만든 조회 헬퍼.

핵심 원칙: model_id를 절대 하드코딩하지 않는다. models.is_production=TRUE인
모델이 항상 "지금 화면에 보여줘야 할 모델"이고, 이건 write_model.py의
promote_best_model()이 바뀔 때마다 자동으로 달라진다 — 그래서 이 파일의
함수들을 거치기만 하면, 나중에 더 좋은 모델로 교체돼도 app.py 코드는
전혀 안 고쳐도 됨.
"""

# TiDB 연결 주의사항:
# - 팀원마다 각자의 로컬 .env를 사용하므로 실제로 연결되는 TiDB가 서로 다를 수 있다.
# - DB 주소·계정과 model_id는 이 파일에 하드코딩하지 않는다.
# - 이 파일은 조회만 담당하며 promote_best_model()을 자동 실행하거나 DB 값을 바꾸지 않는다.
#   나중에 현재 TiDB의 모델이 is_production=TRUE로 승격되면 그 모델을 자동 조회한다.
# - 아래 함수의 결과는 현재 실행 환경의 models.is_production 및 predictions 데이터에
#   따라 달라지며, 프로덕션 모델이나 예측값이 없으면 None 또는 빈 결과를 반환한다.
from typing import Any, Optional

from sqlalchemy import text

from .db import get_engine

_GET_PRODUCTION_MODEL_SQL = text("""
    SELECT model_id FROM models WHERE is_production = TRUE LIMIT 1
""")

_GET_STORE_PREDICTION_SQL = text("""
    SELECT p.score, p.shap_top_features, p.created_at
    FROM predictions p
    WHERE p.store_id = :store_id
      AND p.model_id = (SELECT model_id FROM models WHERE is_production = TRUE LIMIT 1)
      AND p.query_type = 'existing_store'
    ORDER BY p.created_at DESC
    LIMIT 1
""")

_GET_DONG_AVG_SCORE_SQL = text("""
    SELECT s.dong_code, AVG(p.score) AS avg_score, COUNT(*) AS n_stores
    FROM predictions p
    JOIN stores s ON s.store_id = p.store_id
    WHERE p.model_id = (SELECT model_id FROM models WHERE is_production = TRUE LIMIT 1)
      AND p.query_type = 'existing_store'
    GROUP BY s.dong_code
""")

_GET_HIGH_RISK_STORES_SQL = text("""
    SELECT p.store_id, p.score, p.shap_top_features, s.dong_code, s.current_industry_code
    FROM predictions p
    JOIN stores s ON s.store_id = p.store_id
    WHERE p.model_id = (SELECT model_id FROM models WHERE is_production = TRUE LIMIT 1)
      AND p.query_type = 'existing_store'
    ORDER BY p.score DESC
    LIMIT :limit
""")


def get_production_model_id() -> Optional[str]:
    """지금 is_production=TRUE인 모델의 model_id. 없으면 None(모델 아직 등록 전 등)."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(_GET_PRODUCTION_MODEL_SQL).fetchone()
    return row[0] if row else None


def get_prediction_for_store(store_id: str) -> Optional[dict[str, Any]]:
    """기존점주 패널 - 내 매장 폐업 위험점수. 프로덕션 모델 기준, 없으면 None."""
    if not store_id:
        return None
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(_GET_STORE_PREDICTION_SQL, {"store_id": store_id}).fetchone()
    if row is None:
        return None
    return {"score": row[0], "shap_top_features": row[1], "created_at": row[2]}


def get_dong_avg_scores() -> dict[str, dict[str, Any]]:
    """GUEST 메인화면 지도 - 동별 평균 폐업 위험점수 (지금까지 _dong_survival_proxy()로
    임시 대체하던 부분을 이걸로 교체하면 실제 모델 기반이 됨).

    반환: {dong_code: {"avg_score": ..., "n_stores": ...}, ...}
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_GET_DONG_AVG_SCORE_SQL).fetchall()
    return {r[0]: {"avg_score": float(r[1]), "n_stores": int(r[2])} for r in rows}


def get_high_risk_stores(limit: int = 50) -> list[dict[str, Any]]:
    """관리자 대시보드 - 고위험 상권 모니터링용, score 상위 매장 리스트."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_GET_HIGH_RISK_STORES_SQL, {"limit": limit}).fetchall()
    return [
        {"store_id": r[0], "score": r[1], "shap_top_features": r[2],
         "dong_code": r[3], "industry_code": r[4]}
        for r in rows
    ]
