# -*- coding: utf-8 -*-
"""
pmh 데이터 + ML(ExtraTrees, Optuna 튜닝) 결과를 data/features/model_registration.json에
추가한다 (export_model_pjw.py/export_model_dl_tm.py와 동일한 컨벤션).

수치 출처: src/project_2nd/models/ml_pjw/compare_output/finalize_extratrees_5fold_report_pmh.md
(5-fold OOF, 추천 임계값 0.655 기준)

실행 (프로젝트 루트에서):
    python export_model_pmh_ml_pjw.py
    python export_model_pmh_ml_pjw.py --is-production   # 이 모델을 프로덕션으로 지정할 때
"""
import argparse
import sys
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parent / "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from shared.write_model import save_model_json  # noqa: E402

METRICS = {
    "accuracy": 0.8881,
    "precision_score": 0.4668,
    "recall_score": 0.3578,
    "f1_score": 0.4051,
    "roc_auc": 0.7486,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/features/model_registration_pjw.json")
    ap.add_argument("--model-id", default="pmh_ml_extratrees_pjw_v1")
    ap.add_argument("--model-name", default="ExtraTreesClassifier (Optuna tuned, pmh)")
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--is-production", action="store_true")
    args = ap.parse_args()

    save_model_json(
        args.out,
        model_id=args.model_id,
        model_name=args.model_name,
        version=args.version,
        model_type="ML",
        metrics=METRICS,
        is_production=args.is_production,
        append=True,
    )
    print("참고: 이 모델의 추천 분류 임계값은 0.655 (F1 최적화 기준) — "
          "models 테이블 스키마엔 임계값 컬럼이 없어서 등록 안 됨, "
          "predictions.json 만들 때 이 임계값 기준으로 점수를 해석할 것.")


if __name__ == "__main__":
    main()
