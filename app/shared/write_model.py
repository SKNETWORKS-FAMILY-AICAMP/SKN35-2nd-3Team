"""
app/shared/write_model.py

models 테이블에 학습된 모델 정보를 등록한다.
학습 스크립트(예: train_baseline.py) 맨 끝에서 평가지표 계산이 끝나면 이 함수 하나만 호출하면 된다.
"""
from datetime import datetime
from sqlalchemy import text
from app.shared.db import get_engine


def register_model(model_id: str, model_name: str, version: str, model_type: str,
                    accuracy: float, precision: float, recall: float, f1: float, roc_auc: float,
                    is_production: bool = False):
    """
    model_type: 'ML' | 'DL'
    accuracy/precision/recall/f1/roc_auc: sklearn 등으로 계산한 값을 그대로 전달
    """
    engine = get_engine()
    values = {
        'model_id': model_id,
        'model_name': model_name,
        'version': version,
        'model_type': model_type,
        'accuracy': accuracy,
        'precision_score': precision,
        'recall_score': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'trained_at': datetime.now(),
        'is_production': is_production,
    }
    sql = text(
        "INSERT INTO models (model_id, model_name, version, model_type, accuracy, "
        "precision_score, recall_score, f1_score, roc_auc, trained_at, is_production) "
        "VALUES (:model_id, :model_name, :version, :model_type, :accuracy, "
        ":precision_score, :recall_score, :f1_score, :roc_auc, :trained_at, :is_production)"
    )
    with engine.begin() as conn:
        conn.execute(sql, values)


