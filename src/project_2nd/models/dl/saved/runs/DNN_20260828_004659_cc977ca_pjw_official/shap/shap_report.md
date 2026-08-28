# PJW DNN SHAP 분석 결과

- DNN RUN: `DNN_20260828_004659_cc977ca_pjw_official`
- 데이터: `C:\codes\SKN35-2nd-3Team\data\processed\modeling_dataset_refined_pjw.csv`
- 전처리: `src/project_2nd/preprocessing_dataset_pjw`
- Explainer: `DeepExplainer`
- 출력 단위: raw DNN logit
- Background: train 계층표본 256건
- 설명 대상: test 계층표본 1,000건
- 가산성 평균/최대 오차: 0.000000 / 0.000001
- 가산성 검증: True

## 전역 중요도 상위 10개

| 순위 | 피처 | mean(abs(SHAP)) |
|---:|---|---:|
| 1 | store_age_months | 0.245308 |
| 2 | is_left_censored_age | 0.155211 |
| 3 | industry_historical_rate | 0.154891 |
| 4 | industry_group_enc | 0.132318 |
| 5 | dong_industry_historical_rate | 0.092156 |
| 6 | is_mass_reclass_window | 0.082136 |
| 7 | snapshot_month_index | 0.074887 |
| 8 | previously_transitioned | 0.056384 |
| 9 | floor_category_enc | 0.053356 |
| 10 | dong_historical_rate | 0.032226 |

## 해석 주의사항

- positive/negative SHAP은 모델의 폐업 위험 logit을 올리거나 내린 기여다.
- SHAP은 상관계수나 폐업의 원인이 아니다.
- SHAP 값을 확률 퍼센트포인트로 표현하면 안 된다.
- weighted BCE sigmoid 출력은 미보정 위험점수다.
