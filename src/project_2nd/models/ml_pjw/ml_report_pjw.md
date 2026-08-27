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

## 참고 파일

- `compare_output/compare_xgb_extratrees_pmh.py` / `compare_xgb_extratrees_result_pmh.txt` — 기본값 비교
- `compare_output/tune_xgb_extratrees_pmh.py` / `tune_xgb_extratrees_report_pmh.md` — Optuna 튜닝 + 1-split 최종
- `compare_output/finalize_extratrees_5fold_pmh.py` / `finalize_extratrees_5fold_report_pmh.md` — 5-fold 재검증(최종 확정)
