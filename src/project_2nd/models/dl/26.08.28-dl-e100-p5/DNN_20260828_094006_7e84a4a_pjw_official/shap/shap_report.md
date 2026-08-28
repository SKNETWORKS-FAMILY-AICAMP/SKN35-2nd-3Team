# PJW DNN SHAP 분석 결과

- DNN RUN: `DNN_20260828_094006_7e84a4a_pjw_official`
- 데이터: `C:\codes\SKN35-2nd-3Team\data\processed\modeling_dataset_refined_pjw.csv`
- 전처리: `src/project_2nd/preprocessing_dataset_pjw`
- Explainer: `DeepExplainer`
- 출력 단위: raw DNN logit
- Background: train 계층표본 256건
- 설명 대상: test 계층표본 1,000건
- 가산성 평균/최대 오차: 0.000000 / 0.000004
- 가산성 검증: True

## 전역 중요도 상위 10개

| 순위 | 피처 | mean(abs(SHAP)) |
|---:|---|---:|
| 1 | store_age_months | 0.151674 |
| 2 | industry_historical_rate | 0.123664 |
| 3 | is_left_censored_age | 0.108116 |
| 4 | industry_group_enc | 0.107998 |
| 5 | previously_transitioned | 0.107214 |
| 6 | snapshot_month_index | 0.082061 |
| 7 | is_mass_reclass_window | 0.065619 |
| 8 | floor_category_enc | 0.062733 |
| 9 | dong_industry_historical_rate | 0.040068 |
| 10 | industry_code_enc | 0.034934 |

## 해석 주의사항

- positive/negative SHAP은 모델의 폐업 위험 logit을 올리거나 내린 기여다.
- SHAP은 상관계수나 폐업의 원인이 아니다.
- SHAP 값을 확률 퍼센트포인트로 표현하면 안 된다.
- weighted BCE sigmoid 출력은 미보정 위험점수다.
