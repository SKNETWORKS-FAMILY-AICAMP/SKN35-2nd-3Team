# DNN GPU 학습 결과 — Epoch 100 / Patience 5

## 실행 조건

| 항목 | 설정 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 |
| 최대 Epoch | 100 |
| Early stopping patience | 5 |
| 실제 종료 Epoch | 86 |
| 최적 Epoch | 81 |
| 입력 피처 | 34개 |
| 모델 파라미터 | 51,073개 |
| Batch size | 4,096 |
| Seed | 42 |

- 전처리 기준: `src/project_2nd/preprocessing_dataset_pjw`
- 학습 데이터: `data/processed/modeling_dataset_refined_pjw.csv`
- 제외한 전처리: `src/project_2nd/preprocessing_dataset_pmh`
- 데이터 SHA-256: `9180DA5F763FC811A08DC8C1C2069DEC2DEA4D319B404686B8BC129668857C5E`
- Train: 1,137,077행 / Validation: 375,496행 / Test: 377,009행

## 최종 성능

| 항목 | Validation | Test |
|---|---:|---:|
| PR-AUC | 0.4074 | 0.4051 |
| ROC-AUC | 0.7470 | 0.7458 |
| F1 | 0.4013 | 0.3988 |
| Precision | 0.4530 | 0.4492 |
| Recall | 0.3601 | 0.3586 |
| Accuracy | 0.8847 | 0.8852 |

- Validation에서 선택한 판정 임계값: `0.6890`
- Test Precision@Top 5%: `0.5836`
- Test Recall@Top 5%: `0.2749`
- Test Precision@Top 10%: `0.4080`
- Test Recall@Top 10%: `0.3844`

Test 혼동행렬:

```text
TN 319,398 | FP 17,595
FN  25,668 | TP 14,348
```

## 7 Epoch 결과와 비교

| 항목 | 7 Epoch | 100 Epoch·Patience 5 | 변화 |
|---|---:|---:|---:|
| Test PR-AUC | 0.3781 | 0.4051 | +0.0270 |
| Test ROC-AUC | 0.7377 | 0.7458 | +0.0081 |
| Test F1 | 0.3798 | 0.3988 | +0.0190 |
| Precision | 0.4260 | 0.4492 | +0.0231 |
| Recall | 0.3427 | 0.3586 | +0.0159 |

## SHAP 결과

SHAP DeepExplainer 분석을 Test 계층표본 1,000건에 수행했습니다. 가산성 검증 최대 오차는 `0.0000038147`로 허용 오차 `0.05` 이내입니다.

| 순위 | 피처 | 평균 절대 SHAP 값 |
|---:|---|---:|
| 1 | `store_age_months` | 0.1517 |
| 2 | `industry_historical_rate` | 0.1237 |
| 3 | `is_left_censored_age` | 0.1081 |
| 4 | `industry_group_enc` | 0.1080 |
| 5 | `previously_transitioned` | 0.1072 |

## 주의사항

- 최대 100 Epoch로 설정했지만 Validation PR-AUC가 5회 연속 개선되지 않아 86 Epoch에서 정상 조기 종료됐습니다.
- 저장된 모델은 Validation PR-AUC가 가장 높았던 81 Epoch 모델입니다.
- 출력값은 보정된 폐업 확률이 아니라 폐업 위험점수입니다.
- SHAP은 모델의 예측 기여도를 설명하며 실제 폐업 원인이나 인과관계를 증명하지 않습니다.
