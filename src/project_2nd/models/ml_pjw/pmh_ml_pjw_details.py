"""pmh 데이터 + ML(ExtraTrees 튜닝) 모델을 TiDB `models` 테이블에 후보로 등록.

schema.sql의 models 테이블 컬럼에 맞춘 값 - 실제 모델 파일(857MB, .joblib)은
용량 문제로 git/DB에 안 올리고, 여기 성적표(메타데이터)만 등록한다.
is_production=FALSE로 넣음 - 4개 후보(pjw+ML/DL, pmh+ML/DL) 중 팀이 승자를
정하기 전까지는 후보 상태로만 유지.

수치 출처: models/ml_pjw/compare_output/finalize_extratrees_5fold_report_pmh.md
(5-fold OOF, 추천 임계값 0.655 기준)
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import text

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from app.shared.db import get_engine  # noqa: E402

MODEL_DETAILS = {
    "model_id": "pmh_ml_extratrees_pjw_v1",
    "model_name": "ExtraTreesClassifier (Optuna tuned)",
    "version": "1.0",
    "model_type": "ML",
    # 5-fold OOF, 추천 임계값(0.655) 기준 성능
    "accuracy": 0.8881,
    "precision_score": 0.4668,
    "recall_score": 0.3578,
    "f1_score": 0.4051,
    "roc_auc": 0.7486,
    "trained_at": datetime(2026, 8, 28, 11, 59, 22),
    "is_production": False,
}


def insert_model_details(details: dict = MODEL_DETAILS) -> None:
    engine = get_engine()
    sql = text(
        "INSERT INTO models (model_id, model_name, version, model_type, accuracy, "
        "precision_score, recall_score, f1_score, roc_auc, trained_at, is_production) "
        "VALUES (:model_id, :model_name, :version, :model_type, :accuracy, "
        ":precision_score, :recall_score, :f1_score, :roc_auc, :trained_at, :is_production)"
    )
    with engine.begin() as conn:
        conn.execute(sql, details)
    print(f"등록 완료: {details['model_id']}")


if __name__ == "__main__":
    insert_model_details()
