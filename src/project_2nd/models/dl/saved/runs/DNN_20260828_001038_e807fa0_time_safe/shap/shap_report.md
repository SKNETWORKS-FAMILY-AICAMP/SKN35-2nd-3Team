# DNN SHAP 분석 결과

- DNN RUN: `DNN_20260828_001038_e807fa0_time_safe`
- Explainer: `DeepExplainer`
- 출력 단위: raw DNN logit
- Background: train 계층표본 256건
- 설명 대상: test 계층표본 1,000건
- 가산성 평균/최대 절대오차: 0.000000 / 0.000001
- 가산성 허용오차 통과: True

## 전역 중요도 상위 10개

| 순위 | 피처 | mean(abs(SHAP)) |
|---:|---|---:|
| 1 | store_age_months | 0.446938 |
| 2 | industry_group_enc | 0.223713 |
| 3 | snapshot_month_index | 0.221980 |
| 4 | industry_code_enc | 0.100757 |
| 5 | floor_category_enc | 0.057041 |
| 6 | industry_jung_name_enc | 0.047561 |
| 7 | industry_jung_code_enc | 0.036683 |
| 8 | industry_name_enc | 0.031576 |
| 9 | industry_dae_code_enc | 0.028100 |
| 10 | dong_code_enc | 0.024076 |

## 해석 주의사항

- positive SHAP은 해당 예측의 폐업 위험 logit을 올린 기여다.
- negative SHAP은 해당 예측의 폐업 위험 logit을 내린 기여다.
- SHAP은 모델이 사용한 예측 기여도이며 상관계수나 폐업의 원인이 아니다.
- SHAP +0.18을 폐업확률 18%p 증가로 표현하면 안 된다.
- DNN sigmoid 출력은 weighted BCE로 학습된 미보정 위험점수다.
- 전체 데이터가 아니라 계층표본을 설명했으므로 표본 오차가 존재한다.
