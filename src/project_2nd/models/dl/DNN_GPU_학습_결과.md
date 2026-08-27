# DNN GPU 학습 결과 (PJW 전처리 기준)

## 1. 실행 기준

- 전처리 기준: `src/project_2nd/preprocessing_dataset_pjw`
- 학습 데이터: `data/processed/modeling_dataset_refined_pjw.csv`
- 제외한 전처리: `src/project_2nd/preprocessing_dataset_pmh`
- GPU: NVIDIA GeForce RTX 3070
- 학습 횟수: 7 Epoch
- 데이터 분할: Train fold 0~2 / Validation fold 3 / Test fold 4
- 난수 시드: 42
- 입력 피처: 34개
- 모델 파라미터: 51,073개

학습 데이터 SHA-256:

```text
9180DA5F763FC811A08DC8C1C2069DEC2DEA4D319B404686B8BC129668857C5E
```

## 2. 전처리 결과

| 항목 | 결과 |
|---|---:|
| 전체 행 | 1,889,582 |
| 원본 열 | 37 |
| DNN 입력 피처 | 34 |
| 입력 배열 형태 | `(1,889,582, 34)` |
| NaN/무한대 | 0 |

- 범주형 8개 열은 Train fold만 사용하여 정수 인코딩했습니다.
- 숫자형 결측치는 Train fold의 중앙값으로 채웠습니다.
- 수치 스케일러도 Train fold에만 적합하여 Validation/Test 정보가 학습에 섞이지 않도록 했습니다.
- `snapshot_date`는 월 단위 인덱스로 변환했습니다.
- `is_closed_next`(타깃), `fold`, `store_id`, `transitioned_next`는 입력 피처에서 제외했습니다.

## 3. 모델 구조

```text
입력 34개
  → Dense 256 + BatchNorm + ReLU + Dropout(0.3)
  → Dense 128 + BatchNorm + ReLU + Dropout(0.3)
  → Dense 64  + BatchNorm + ReLU + Dropout(0.3)
  → Dense 1
```

- 손실 함수: 클래스 불균형 가중치를 적용한 BCEWithLogitsLoss
- 최적화 함수: AdamW
- 학습률: 0.001
- Weight decay: 0.0001
- 배치 크기: 4,096
- 최적 모델 선정 기준: Validation PR-AUC

## 4. Epoch별 학습 결과

| Epoch | Validation PR-AUC | Validation ROC-AUC |
|---:|---:|---:|
| 1 | 0.3246 | 0.7220 |
| 2 | 0.3392 | 0.7274 |
| 3 | 0.3554 | 0.7322 |
| 4 | 0.3666 | 0.7351 |
| 5 | 0.3718 | 0.7363 |
| 6 | 0.3765 | 0.7377 |
| 7 | **0.3802** | **0.7385** |

최적 모델은 7 Epoch 모델입니다.

## 5. 최종 평가 결과

| 항목 | 결과 |
|---|---:|
| Validation PR-AUC | 0.3802 |
| Test PR-AUC | 0.3781 |
| Test ROC-AUC | 0.7377 |
| Test F1 | 0.3798 |
| Precision | 0.4260 |
| Recall | 0.3427 |
| Accuracy | 0.8812 |
| 판정 임계값 | 0.6937 |
| Brier score | 0.1928 |
| ECE | 0.3205 |
| Precision@Top 5% | 0.5507 |
| Recall@Top 5% | 0.2594 |
| Precision@Top 10% | 0.3907 |
| Recall@Top 10% | 0.3681 |

Test 혼동행렬:

```text
TN 318,520 | FP 18,473
FN  26,304 | TP 13,712
```

## 6. SHAP 분석 결과

SHAP 실행 코드는 참고 폴더인 `models/shap`을 수정하지 않고 `models/dl/run_shap_dnn.py`에 작성했습니다. 결과 역시 해당 DNN 실행 폴더 안에 저장했습니다.

SHAP 전역 중요도 상위 피처:

| 순위 | 피처 | 평균 절대 SHAP 값 |
|---:|---|---:|
| 1 | `store_age_months` | 0.2453 |
| 2 | `is_left_censored_age` | 0.1552 |
| 3 | `industry_historical_rate` | 0.1549 |
| 4 | `industry_group_enc` | 0.1323 |
| 5 | `dong_industry_historical_rate` | 0.0922 |
| 6 | `is_mass_reclass_window` | 0.0821 |
| 7 | `snapshot_month_index` | 0.0749 |
| 8 | `previously_transitioned` | 0.0564 |
| 9 | `floor_category_enc` | 0.0534 |
| 10 | `dong_historical_rate` | 0.0322 |

- SHAP 배경 데이터: Train 256건
- SHAP 분석 데이터: Test 1,000건
- Additivity 평균 오차: `0.0000001004`
- Additivity 최대 오차: `0.0000014305`
- 허용 오차 0.05 기준: 통과

SHAP 중요도는 모델이 판단할 때 각 피처를 얼마나 사용했는지를 설명하며, 실제 폐업 원인이나 인과관계를 증명하지는 않습니다.

## 7. 산출물 위치

```text
src/project_2nd/models/dl/saved/runs/DNN_20260828_004659_cc977ca_pjw_official
```

주요 산출물:

- 학습 모델 가중치
- 실행 환경 및 데이터 출처가 기록된 manifest
- Validation/Test 평가 지표
- Test 예측 결과
- 학습 곡선 및 평가 그래프
- SHAP 전역 중요도, beeswarm, 개별 예측 waterfall 결과

## 8. 단위 테스트

PJW 데이터 출처 확인, PMH 경로 차단, 누수 피처 제외, 배열 정합성, DNN 구조, 평가 지표, SHAP 결과 형식 등을 포함한 단위 테스트 12개를 모두 통과했습니다.

## 9. 해석 시 주의사항

- 모델 출력값은 보정된 폐업 확률이 아니라 **폐업 위험점수**입니다.
- 이번 결과는 PJW 전처리 결과의 피처를 공식 입력으로 사용한 비교용 학습 결과입니다.
- 미래 정보 사용 여부가 별도로 검증되지 않은 누적·과거 비율 계열 피처는 실제 서비스 배포 전에 생성 시점과 데이터 기준일을 다시 확인해야 합니다.
- 임계값 0.6937은 Validation 데이터에서 F1이 높도록 정한 값이며, 실제 운영에서는 오탐 비용과 미탐 비용에 맞춰 다시 조정할 수 있습니다.
