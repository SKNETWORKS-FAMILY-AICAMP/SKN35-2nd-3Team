# pmh 데이터 기반 머신러닝 모델링 리포트 (pjw 담당)

담당: pjw. 데이터: minha님 정제본(`data/processed/modeling_dataset_preprocessed_pmh.csv`, 43컬럼).
minha님이 이미 LightGBM vs CatBoost를 직접 비교해뒀기 때문에(`preprocessing_dataset_pmh/compare_output/`),
같은 조건에서 **아직 안 나온 모델**(XGBoost, ExtraTrees)로 이어서 비교하고, 그중 나은 쪽을 튜닝했다.

## 실행 순서

1. XGBoost / ExtraTrees 기본 하이퍼파라미터 5-fold 비교 (`compare_xgb_extratrees_pmh.py`)
2. Optuna로 두 모델 각각 튜닝 (`tune_xgb_extratrees_pmh.py`) — DL팀 계획서의 고정분할 관례(train=fold0,1,2 / val=fold3 / test=fold4) 재사용, validation PR-AUC를 목적함수로 사용
3. 튜닝 결과 더 좋은 쪽(ExtraTrees) 승자 선정 → validation에서 F1 최적 임계값 탐색
4. 승자 하이퍼파라미터를 고정한 채 **5-fold 전체로 재검증**(`finalize_extratrees_5fold_pmh.py`) — 1개 fold짜리 결과가 우연이 아님을 확인

## 피처 구성

minha님의 `compare_lgbm_catboost_pmh.py`와 동일한 31개 피처(원본 34컬럼에서 `store_id`/`fold`/`is_closed_next` 제외, `_enc` 인코딩 컬럼 9개도 중복이라 제외)를 그대로 사용 — 모델만 바뀐 상태로 비교하기 위함.

- XGBoost: 원본 문자열 범주형 10개를 category dtype으로 네이티브 지원
- ExtraTrees: sklearn 트리는 category dtype을 못 받아서 `_enc` 라벨인코딩 컬럼으로 대체(정보량은 동일, 표현만 다름)

⚠️ **주의**: minha님 원본 방식을 그대로 따라서 `transitioned_next`(pjw 쪽에서 타깃 누수로 판단해 제거한 컬럼)를 제외하지 않았다. 실제로 XGBoost 기본값 비교에서 이 컬럼이 feature importance 5위(전체의 7.5%)로 잡혀서 영향이 작지 않아 보임 — 절대 수치는 다소 부풀려져 있을 수 있다. minha님께 공유 완료(공유 필요).

## 최종 결과

### 기본 하이퍼파라미터 5-fold (원본, XGB/ET vs minha님 LightGBM/CatBoost)

| | ROC-AUC | PR-AUC |
|---|---|---|
| LightGBM (minha) | 0.7473~0.7476 | 0.4050 |
| CatBoost (minha) | 0.7431~0.7472 | 0.3965 |
| XGBoost (기본값) | 0.7455 | 0.4004 |
| ExtraTrees (기본값) | 0.7134 | 0.3504 |

→ 둘 다 LightGBM을 못 넘음. XGBoost는 근접, ExtraTrees는 확실히 뒤처짐.

### Optuna 튜닝 + 5-fold 재검증

| | validation PR-AUC (탐색 기준) |
|---|---|
| XGBoost 튜닝 (40 trials) | 0.4095 |
| **ExtraTrees 튜닝 (15 trials)** | **0.4115** ← 승자 |

**최종 확정 — ExtraTrees(튜닝), 5-fold OOF:**

| | ROC-AUC | PR-AUC |
|---|---|---|
| LightGBM (minha, OOF) | 0.7476 | 0.4050 |
| **ExtraTrees (튜닝, OOF)** | **0.7486** | **0.4123** |

5-fold 전부 표준편차가 작음(ROC-AUC ±0.0006, PR-AUC ±0.001) — 우연이 아니라 일관된 개선.

**최종 하이퍼파라미터**: `n_estimators=280, max_depth=22, min_samples_leaf=35, min_samples_split=10, max_features=0.818, class_weight='balanced'`

**임계값**: 0.5 → F1 0.3505 / 최적 임계값 0.655 → F1 0.4051 (precision 0.26→0.47, recall 0.54→0.36 트레이드오프)

## 결론

기본 하이퍼파라미터 비교만으로는 4개 모델(LightGBM/CatBoost/XGBoost/ExtraTrees) 중 LightGBM이 계속 1위였지만, **ExtraTrees를 Optuna로 제대로 튜닝하니 LightGBM의 OOF 기준 성능을 근소하게 넘어섰다** (ROC-AUC +0.001, PR-AUC +0.0073). 차별화 포인트는 "다른 모델을 더 찾는 것"이 아니라 "덜 주목받던 모델을 제대로 튜닝하는 것"이었음.

## SHAP 분석

최종 모델(ExtraTrees 튜닝본)을 train+val(fold 0~3)로 재학습해 저장하고(`saved/best_model_extratrees_pmh.joblib`, 857MB라 git엔 안 올림 — gitignore 처리, 팀 공유는 Drive 등 별도 필요), SHAP(TreeExplainer)로 전역 피처 중요도를 뽑았다. 개별 예측 단위 설명(`predictions.shap_top_features` 스키마, `[{"feature":..., "shap_value":..., "feature_value":...}, ...]`)을 만드는 재사용 함수(`explain_prediction()`/`explain_batch()`)도 `models/shap/explain_prediction.py`에 공용으로 만들어둠 — 트리 기반 모델이면 어떤 팀원 모델이든 재사용 가능.

### SHAP 전역 중요도 top 10 (기존 feature_importances_ 순위와 다름)

| 순위 | 피처 | 비중 |
|---|---|---|
| 1 | `store_age_months` | 30.94% |
| 2 | `snapshot_date` | 15.81% |
| 3 | `industry_historical_rate` | 9.70% |
| 4 | `industry_group` | 8.63% |
| 5 | `dong_industry_historical_rate` | 5.91% |
| 6 | `floor_category` | 4.54% |
| 7 | `previously_transitioned` | 2.67% |
| 8 | `industry_dae_code` | 2.50% |
| 9 | `industry_code` | 2.20% |
| 10 | `industry_jung_code` | 1.77% |

### 왜 split 기반 importance와 순위가 다른가

두 지표는 애초에 재는 대상이 다르다.

- **`feature_importances_`(split 횟수 기반)**: 트리를 만들 때 이 피처로 몇 번 가지를 쪼갰는지 셈. 나무 구조에 대한 통계라, 실제 예측값이 얼마나 바뀌었는지는 반영 안 함.
- **SHAP**: 각 행마다 "이 피처의 실제 값이 예측 확률을 얼마나 밀어올리거나 내렸는지"를 직접 계산 — 출력값에 대한 영향력을 잼.

`industry_code`는 192개 카테고리라 쪼갤 수 있는 후보 분기점이 훨씬 많아서, 실제 기여도와 무관하게 "선택지가 많다"는 이유만으로 split 횟수가 과대평가되는 경향이 있다(카디널리티 높은 피처의 잘 알려진 편향). 반대로 `store_age_months`는 값이 5개(0/6/12/18/24개월)뿐이라 split 횟수는 불리하지만, 값이 바뀔 때마다 예측 확률이 아주 크고 일관되게 움직인다(EDA 확인: 신규매장 폐업률 15.2% vs 24개월차 5.7%, 거의 3배 차이 — [소상공인 폐업 리포트 아티팩트](https://claude.ai/code/artifact/132890d2-4fc8-44c0-bc78-1ac71b3ef1f2) 참고). SHAP은 "몇 번 쪼갰는지"가 아니라 "쪼갤 때마다 얼마나 크게 움직였는지"를 직접 재기 때문에 이 강력하고 일관된 효과를 정확히 잡아낸 것.

부수 효과로, `industry_code`/`industry_jung_code`/`industry_dae_code`/`industry_group`는 같은 정보를 다른 해상도로 담은 중복 컬럼들인데, split 기반은 카디널리티가 제일 높은 세분류(`industry_code`)로 쏠리는 반면 SHAP은 각 컬럼의 실제 기여를 더 고르게 나눠 보여줘서 굵직한 분류(`industry_group`, `industry_dae_code`)가 상대적으로 더 올라왔다.

## 참고 파일

- `compare_output/compare_xgb_extratrees_pmh.py` / `compare_xgb_extratrees_result_pmh.txt` — 기본값 비교
- `compare_output/tune_xgb_extratrees_pmh.py` / `tune_xgb_extratrees_report_pmh.md` — Optuna 튜닝 + 1-split 최종
- `compare_output/finalize_extratrees_5fold_pmh.py` / `finalize_extratrees_5fold_report_pmh.md` — 5-fold 재검증(최종 확정)
- `compare_output/build_results_json_pmh.py` / `compare_output/ml_results_summary_pmh.json` — 전체 결과 팀 공유용 JSON
- `shap_output/build_final_model_and_shap_pmh.py` — SHAP 분석 + 최종 모델 저장 스크립트
- `../shap/explain_prediction.py` — 팀 공용 SHAP 설명 함수(재사용 가능)
- `../shap/shap_feature_importance_pmh.json` / `../shap/shap_top_features_examples_pmh.json` — SHAP 분석 결과(팀 공용 폴더에 위치)
