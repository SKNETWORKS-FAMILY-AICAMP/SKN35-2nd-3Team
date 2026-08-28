# pmh + ML(ExtraTrees 튜닝) — 실행 방법

담당: pjw. 데이터: minha님 정제본(pmh). 최종 모델: ExtraTreesClassifier(Optuna 튜닝).
최종 성능(5-fold OOF): ROC-AUC 0.7486, PR-AUC 0.4123 — 자세한 내용은 [ml_report_pjw.md](ml_report_pjw.md) 참고.

---

## A. 우리 결과만 내 TiDB에 반영하고 싶을 때 (가장 빠름, 재학습 불필요)

이미 만들어둔 JSON 2개(Drive로 전달됨)를 받아서 프로젝트 루트의 `data/features/`에 넣고:

```bash
python load_models_and_predictions.py \
    --model-json data/features/model_registration_pjw.json \
    --predictions-json data/features/predictions_for_db_pmh_ml_pjw.json \
    --auto-promote-best
```

- `model_registration_pjw.json`: 모델 성적표 1건 (`pmh_ml_extratrees_pjw_v1`)
- `predictions_for_db_pmh_ml_pjw.json`: 최신 실제 스냅샷(202512, 2025년 12월) 기준 매장 50건의 실제 예측+SHAP 값
- `--auto-promote-best`: DB에 등록된 모델 중 ROC-AUC 제일 높은 걸 자동으로 `is_production=TRUE`로 지정 (다른 사람 모델까지 같이 등록했으면 그중에서 비교됨)

## B. 우리 파이프라인을 처음부터 재현하고 싶을 때

전부 프로젝트 루트에서 실행. `data/processed/modeling_dataset_preprocessed_pmh.csv`가 먼저 있어야 함(minha님 전처리 결과 — 없으면 `src/project_2nd/preprocessing_dataset_pmh/preprocess_output/README_pmh.md` 참고).

```bash
# 1. 기본 하이퍼파라미터로 XGBoost/ExtraTrees vs LightGBM/CatBoost 비교 (5-fold)
python src/project_2nd/models/ml_pjw/compare_output/compare_xgb_extratrees_pmh.py

# 2. Optuna 튜닝 (XGBoost 40 trials + ExtraTrees 15 trials, 시간 꽤 걸림 - 1시간 이상 가능)
python src/project_2nd/models/ml_pjw/compare_output/tune_xgb_extratrees_pmh.py

# 3. 튜닝된 하이퍼파라미터로 5-fold 재검증 (최종 확정 수치)
python src/project_2nd/models/ml_pjw/compare_output/finalize_extratrees_5fold_pmh.py

# 4. 최종 모델 학습 + 저장(.joblib, 857MB - git에 없음, 로컬에서 새로 생성됨) + SHAP 전역 중요도
python src/project_2nd/models/ml_pjw/shap_output/build_final_model_and_shap_pmh.py

# 5. 모델 등록 JSON 생성
python export_model_pmh_ml_pjw.py

# 6. 실제 예측 JSON 생성 (4번에서 만든 .joblib 모델 필요, 기본 50건, 최신 스냅샷 자동 선택)
python export_predictions_pmh_ml_pjw.py

# 7. DB 적재
python load_models_and_predictions.py \
    --model-json data/features/model_registration_pjw.json \
    --predictions-json data/features/predictions_for_db_pmh_ml_pjw.json \
    --auto-promote-best
```

## 참고

- `explain_prediction.py`/`explain_batch()`(`../shap/explain_prediction.py`)는 트리 기반 모델이면 어떤 모델이든 재사용 가능한 공용 SHAP 함수 — 다른 사람 트리 모델(RandomForest, LightGBM 등)에도 그대로 쓸 수 있음
- `export_predictions_pmh_ml_pjw.py`는 `--n-samples`(기본 50), `--snapshot-date`(기본: 데이터 내 최신 스냅샷 자동), `--seed`(기본 42) 옵션으로 조정 가능
- 모델 파일(`saved/best_model_extratrees_pmh.joblib`, 857MB)은 git에 안 올라가 있음 — 4번 스크립트를 실행해야 로컬에 새로 생성됨
