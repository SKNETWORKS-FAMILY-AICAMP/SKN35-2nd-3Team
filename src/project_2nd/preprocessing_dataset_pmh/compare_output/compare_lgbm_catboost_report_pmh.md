
# LightGBM vs CatBoost 5-fold 비교 리포트 (pmh)

- 데이터: `data/processed/modeling_dataset_preprocessed_pmh.csv` (1,889,582행)
- 피처: 31개 (원본 문자열 범주형 10개는 category 타입으로 사용, `_enc` 정수 컬럼은 중복이라 제외)
- 검증: `fold` 컬럼 기준 5-fold, 매번 4개 fold로 학습 → 나머지 1개 fold로 평가
- 원본 스크립트: [compare_lgbm_catboost_pmh.py](compare_lgbm_catboost_pmh.py)
- 원본 결과: [compare_lgbm_catboost_result_pmh.txt](compare_lgbm_catboost_result_pmh.txt)

## 5-fold 평균 비교 요약

| 지표 | LightGBM | CatBoost |
|---|---|---|
| accuracy | 0.9063 | 0.9032 |
| precision | 0.7590 | 0.6385 |
| recall | 0.1759 | 0.2103 |
| f1 | 0.2856 | 0.3163 |
| roc_auc | 0.7473 ± 0.0009 | 0.7472 ± 0.0027 |

## 해석

1. **ROC-AUC는 사실상 동률** — 두 모델 다 0.747x이고 차이(0.0001)가 표준편차(±0.0009~0.0027)보다 훨씬 작다. 랭킹/확률 기반 분류력 자체는 두 모델이 거의 같다는 뜻.
2. **하지만 기본 임계값(0.5) 기준 precision/recall/f1은 꽤 다르다** — LightGBM은 precision 0.76 / recall 0.18로 "확실할 때만 폐업으로 예측"하는 쪽에 가깝고, CatBoost는 precision 0.64 / recall 0.21로 좀 더 적극적으로 폐업을 잡아낸다(f1은 CatBoost가 0.316으로 더 높음). 이건 모델 성능 차이라기보다 **두 모델이 뽑아내는 확률 분포(캘리브레이션)가 달라서 생기는 임계값 효과**일 가능성이 커서, 나중에 임계값을 직접 조정하면 격차가 줄어들 수 있다 — "CatBoost가 더 낫다"는 결론은 아직 이르고, 확인이 더 필요하다.
3. **feature importance가 완전히 다른 그림을 보여준다.**
   - LightGBM: `dong_code`(1276), `industry_code`(904)처럼 고카디널리티 범주형이 압도적 1·2위. 트리 분기가 이 두 컬럼에 크게 의존.
   - CatBoost: `store_age_months`(17.9), `previously_transitioned`(17.5), `snapshot_date`(10.9)처럼 수치형/저카디널리티 피처가 상위권이고 `dong_code`(3.5)·`industry_code`(3.2)는 한참 아래. CatBoost의 ordered boosting이 고카디널리티 범주형에 대한 과적합성 분기를 덜 만든다는 특성과 일치.
   - 두 모델의 importance 상위권이 이렇게 갈리는 건 앙상블(두 모델 예측 평균)을 시도해볼 여지가 있다는 신호이기도 하다.

## 다음 단계 제안

- 지금은 기본 하이퍼파라미터 그대로 비교한 baseline이라, 임계값(threshold)을 0.5 고정이 아니라 PR-AUC나 F1 최적 지점 기준으로 다시 비교하면 두 모델의 실질적 우위를 더 정확히 판단할 수 있다.
- pjw 버전(원본 0.7479 / 정제본 0.7483, LightGBM 기준)과 비교하면 이번 pmh 버전 LightGBM 결과(0.7473)가 근소하게 낮은데, 피처 구성 차이(pjw는 `transitioned_next` 등 제거, `is_left_censored_age`/`is_trend_keyword_match` 추가) 때문일 가능성이 높다 — 필요하면 같은 피처 조정을 pmh 쪽에도 적용해 재비교해볼 수 있다.
