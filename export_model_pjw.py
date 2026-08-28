# -*- coding: utf-8 -*-
"""
dl_train_tm.py가 만든 ensemble_config.json에서 성능 지표를 뽑아
data/features/model_registration.json에 이 모델 1건을 추가함.

여러 명이 각자 이런 export 스크립트를 자기 모델용으로 실행하면, 전부 같은
파일(data/features/model_registration.json)에 이어붙여져서 결국 하나로
모임 (write_model.py의 save_model_json(append=True) 덕분 — 같은 model_id면
덮어쓰고, 다른 model_id면 그냥 추가됨).

실행 (프로젝트 루트에서):
    python export_model_dl_tm.py
    python export_model_dl_tm.py --is-production   # 이 모델을 프로덕션으로 지정할 때
"""
import argparse
import json
import sys
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parent / "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from shared.write_model import save_model_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-config", default="models/dl/saved/ensemble_config.json")
    ap.add_argument("--out", default="data/features/model_registration.json")
    ap.add_argument("--model-id", default="dnn_mlp_v2")
    ap.add_argument("--model-name", default="DNN (5-fold ensemble, dl_tm)")
    ap.add_argument("--version", default="2.0")
    ap.add_argument("--is-production", action="store_true")
    args = ap.parse_args()

    with open(args.ensemble_config, encoding="utf-8") as f:
        econfig = json.load(f)
    cv = econfig.get("cv_summary", {})

    # 이 파이프라인은 5-fold CV 평균 ROC-AUC만 뽑아뒀음(accuracy/precision/recall/f1은
    # 따로 계산 안 해서 None -> DB에 NULL로 들어감, 스키마상 허용됨).
    save_model_json(
        args.out,
        model_id=args.model_id,
        model_name=args.model_name,
        version=args.version,
        model_type="DL",
        metrics={"roc_auc": cv.get("roc_auc_mean")},
        is_production=args.is_production,
        append=True,
    )
    print(f"참고: PR-AUC={cv.get('pr_auc_mean')}, Top5%Lift={cv.get('top5pct_lift_mean')} "
          f"(models 테이블 스키마엔 이 두 항목이 없어서 등록 안 됨 — 필요하면 리포트 문서에 별도 기재)")


if __name__ == "__main__":
    main()