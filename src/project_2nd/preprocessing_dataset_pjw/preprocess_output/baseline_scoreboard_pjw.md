# 베이스라인 스코어보드 (ML/DL 팀 공유용)

전처리(pjw) 파트에서 5-fold LightGBM으로 검증한 공식 베이스라인 수치. ML/DL 모델이
이 숫자보다 유의미하게 좋아야 "전처리가 아니라 모델링이 성능을 끌어올렸다"고 말할 수 있다.

## 검증 방법

- 모델: `lightgbm.LGBMClassifier(random_state=42)`, 기본 하이퍼파라미터(튜닝 없음)
- 검증: GroupKFold 5-fold — `fold` 컬럼 기준(store_id를 md5 해시해 배정, 같은 매장은 항상 같은 fold라 누수 없음)
- 타깃: `is_closed_next` (다음 스냅샷 폐업 여부, 전체 평균 약 10.6%)
- 지표: Accuracy / Precision / Recall / F1 / **ROC-AUC**(주 지표)
- 재현: `compare_baseline_5fold_pjw.py` (원본 vs 정제본), `feature_importance_full_pjw.py` (전체 피처 중요도)

## 최종 스코어 (2026-08-26, 팀 최신 데이터 기준)

| | 원본(`modeling_dataset.csv`, 34컬럼) | 정제본(`modeling_dataset_refined_pjw.csv`, 37컬럼) |
|---|---|---|
| **ROC-AUC** | 0.747793 ± 0.0007 | **0.748440 ± 0.0009** |
| Accuracy | 0.906171 | 0.906280 |
| Precision | 0.766215 | 0.765020 |
| Recall | 0.171059 | 0.173039 |
| F1 | 0.279678 | 0.282237 |

전처리로 얻은 개선폭은 **+0.00065 (ROC-AUC)** — 5-fold 전부 일관된 방향이라 노이즈는 아니지만
절대적으로는 작은 수치. 즉 **이 데이터셋 자체의 트리 모델 기준 상한이 ROC-AUC 0.75 부근**이라는
뜻이므로, ML/DL 모델이 이보다 크게 낮게 나온다면 전처리보다는 모델/학습 설정을 먼저 의심해볼 것.

## 참고: 정제본 사용 권장 컬럼

정제본(`modeling_dataset_refined_pjw.csv`)이 원본보다 다음 이유로 더 안전하다:
- `gu_name`/생활인구 결측 관련 데이터 완전성 이슈가 해소된 상태(팀원분 상류 수정 + 우리 검증 이중 확인)
- 타깃 누수 컬럼(`transitioned_next`) 제거됨
- `store_age_months`의 좌측절단 문제, 트렌드 키워드의 이진 신호 손실 문제를 보완하는 플래그 포함

컬럼별 의미/결측률은 [data_dictionary_pjw.md](data_dictionary_pjw.md), 컬럼별 중요도 순위는
[feature_importance_full_result_pjw.txt](feature_importance_full_result_pjw.txt) 참고.

## 히스토리 (참고용)

| 버전 | 데이터 기준 | 원본 ROC-AUC | 정제본 ROC-AUC | 비고 |
|---|---|---|---|---|
| v4 | 팀원 floor_category/coord_cluster_size 반영 직후 | 0.747896 | 0.748261 (+0.00037) | gu_name 버그 우리가 발견/수정 |
| v5 | 팀원분이 gu_name/population 상류 수정 후 | 0.747793 | 0.748440 (+0.00065) | 현재 최종 |
