"""
app/shared/write_prediction.py

predictions 테이블에 예측 요청 결과(+ SHAP 설명)를 기록한다.
기존점주 위험도 조회 / 예비창업자 위치+업종 조회 양쪽 다 이 함수 하나로 처리한다.
"""
import json
from datetime import datetime
from sqlalchemy import text

from .db import get_engine


def log_prediction(model_id: str, query_type: str, industry_code: str, score: float,
                    user_id: str | None = None, store_id: str | None = None,
                    query_lat: float | None = None, query_lng: float | None = None,
                    shap_top_features: list | None = None):
    """
    query_type: 'existing_store' | 'new_location'
      - 'existing_store'면 store_id를 채우고 query_lat/query_lng는 None
      - 'new_location'이면 query_lat/query_lng를 채우고 store_id는 None
    shap_top_features 예: [{"feature": "store_age_months", "shap_value": -0.12, "feature_value": 6}, ...]
    """
    engine = get_engine()
    values = {
        'model_id': model_id,
        'user_id': user_id,
        'query_type': query_type,
        'store_id': store_id,
        'query_lat': query_lat,
        'query_lng': query_lng,
        'industry_code': industry_code,
        'score': score,
        # JSON 컬럼은 드라이버가 dict/list -> JSON 자동 변환을 안 해주므로 문자열로 직접 변환해서 넣는다.
        'shap_top_features': json.dumps(shap_top_features, ensure_ascii=False) if shap_top_features else None,
        'created_at': datetime.now(),
    }
    sql = text(
        "INSERT INTO predictions (model_id, user_id, query_type, store_id, query_lat, query_lng, "
        "industry_code, score, shap_top_features, created_at) "
        "VALUES (:model_id, :user_id, :query_type, :store_id, :query_lat, :query_lng, "
        ":industry_code, :score, :shap_top_features, :created_at)"
    )
    with engine.begin() as conn:
        conn.execute(sql, values)


# 사용 예 (기존점주 위험도 조회 화면):
#   from .write_prediction import log_prediction   # (상대 import, 전엔 app.shared.write_prediction)
#
#   proba = model.predict_proba(X_one_store)[0][1]
#   shap_values = explainer(X_one_store)
#   top5 = sorted(zip(feature_names, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
#   shap_json = [{"feature": n, "shap_value": float(v), "feature_value": float(X_one_store[n])} for n, v in top5]
#
#   log_prediction(model_id='lightgbm_v1', query_type='existing_store', industry_code=store_industry,
#                  score=float(proba), user_id=current_user_id, store_id=store_id,
#                  shap_top_features=shap_json)