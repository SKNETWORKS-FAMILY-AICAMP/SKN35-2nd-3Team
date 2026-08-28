# -*- coding: utf-8 -*-
"""
프로젝트 루트에서 실행하는 최종 DB 적재 스크립트.

app/shared/write_model.py(함수만 있는 라이브러리)를 shared.write_model로
import해서 씀 — write_model.py 자체는 app/shared/ 밑에서 상대 import
(from .db import get_engine)를 쓰기 때문에 직접 실행이 안 되고, 이렇게
루트의 별도 스크립트에서 불러다 써야 함 (app.py 등과 동일한 이유:
"app" is not a package 문제 회피를 위해 app/ 폴더를 sys.path에 넣는 방식).

실행 (프로젝트 루트에서):
    python load_models_and_predictions.py \\
        --model-json data/features/model_registration.json \\
        --predictions-json data/features/predictions_for_db.json

    여러 모델이면 --model-json/--predictions-json을 여러 번 반복해서 순서대로
    짝지어 넘기면 됨 (모델을 먼저 등록해야 predictions의 model_id FK가 통과되므로,
    항상 "그 모델 -> 그 모델의 predictions" 순서로 처리함):

    python load_models_and_predictions.py \\
        --model-json data/features/model_registration.json \\
        --predictions-json data/features/predictions_for_db_dnn_tm.json \\
        --predictions-json data/features/predictions_for_db_pjw.json

    (model_registration.json 하나에 모델이 여러 건 들어있어도 됨 — export_model_*.py들이
     전부 같은 파일에 append하도록 만들어져 있으므로, 보통 --model-json은 1번만 줘도 됨)

    predictions 없이 모델만 등록하고 싶으면 --predictions-json 없이
    --model-json만 줘도 됨.

    적재 후 "지금 등록된 모델 중 성능이 제일 좋은 모델"을 자동으로
    is_production=True로 지정하고 싶으면 --auto-promote-best 추가
    (기본 비교 기준: roc_auc, --promote-metric으로 다른 지표 선택 가능):

    python load_models_and_predictions.py \\
        --model-json data/features/model_registration.json \\
        --auto-promote-best

    이미 모델이 다 등록/적재된 상태에서, 새로 아무것도 안 넣고 그냥
    "지금 DB에 있는 모델들 중에서 다시 골라줘"만 하고 싶으면
    --model-json/--predictions-json 없이 이렇게만 실행해도 됨:

    python load_models_and_predictions.py --auto-promote-best
"""
import argparse
import sys
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parent / "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from shared.write_model import (  # noqa: E402
    load_models_json_to_db, load_predictions_json_to_db, promote_best_model,
)


def main():
    ap = argparse.ArgumentParser(
        description="model_registration.json / predictions.json을 models/predictions 테이블에 적재"
    )
    ap.add_argument("--model-json", action="append", default=[],
                     help="모델 등록 JSON 경로 (여러 파일이면 여러 번 반복, 보통 1개로 충분)")
    ap.add_argument("--predictions-json", action="append", default=[],
                     help="predictions JSON 경로 (모델별로 여러 번 반복 가능)")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--auto-promote-best", action="store_true",
                     help="적재 후 DB에 등록된 모델 중 성능이 제일 좋은 모델을 자동으로 프로덕션 지정")
    ap.add_argument("--promote-metric", default="roc_auc",
                     choices=["roc_auc", "accuracy", "precision_score", "recall_score", "f1_score"],
                     help="--auto-promote-best 사용 시 비교 기준 (기본 roc_auc)")
    args = ap.parse_args()

    if not args.model_json and not args.predictions_json and not args.auto_promote_best:
        raise SystemExit("--model-json / --predictions-json / --auto-promote-best 중 최소 1개는 넘겨야 함")

    n_models = len(args.model_json)
    n_preds = len(args.predictions_json)

    for i in range(max(n_models, n_preds)):
        print(f"\n=== [{i+1}/{max(n_models, n_preds)}] ===")
        if i < n_models:
            print(f"모델 등록: {args.model_json[i]}")
            load_models_json_to_db(args.model_json[i])
        if i < n_preds:
            print(f"predictions 적재: {args.predictions_json[i]}")
            load_predictions_json_to_db(args.predictions_json[i], chunk_size=args.chunk_size)

    if args.auto_promote_best:
        print(f"\n=== 자동 프로덕션 선정 (기준: {args.promote_metric}) ===")
        promote_best_model(metric=args.promote_metric)

    print("\n전체 완료.")


if __name__ == "__main__":
    main()