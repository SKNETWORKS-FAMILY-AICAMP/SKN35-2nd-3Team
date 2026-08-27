
# LightGBM vs CatBoost 임계값 최적화 비교 리포트 (pmh)

- 데이터: `data/processed/modeling_dataset_preprocessed_pmh.csv` (1,889,582행), 피처 31개
- 방식: 5-fold OOF(out-of-fold) 확률을 모아 임계값 무관 지표(ROC-AUC, PR-AUC)로 비교하고,
  OOF 전체에서 F1을 최대화하는 임계값을 찾아 그 기준으로 다시 fold별 성능 비교
- train의 10%를 검증셋으로 떼어 early stopping(30라운드 무개선 시 중단)을 두 모델에 동일 적용
- 스크립트: [compare_lgbm_catboost_threshold_pmh.py](compare_lgbm_catboost_threshold_pmh.py)
- 원본 결과: [compare_lgbm_catboost_threshold_result_pmh.txt](compare_lgbm_catboost_threshold_result_pmh.txt)

## 임계값 무관 지표 (OOF 전체)

| 지표 | LightGBM | CatBoost |
|---|---|---|
| roc_auc | 0.7476 | 0.7431 |
| pr_auc (average precision) | 0.4050 | 0.3965 |

## 기본 임계값(0.5) vs F1 최적 임계값

| | LightGBM (0.5) | LightGBM (최적 0.206) | CatBoost (0.5) | CatBoost (최적 0.259) |
|---|---|---|---|---|
| precision | 0.756 | 0.458 | 0.712 | 0.444 |
| recall | 0.177 | 0.354 | 0.195 | 0.340 |
| f1 | 0.287 | **0.399** | 0.306 | 0.385 |

## 해석

1. **임계값을 튜닝하니 결론이 뒤집혔다.** 지난번 기본 0.5 임계값 비교에서는 CatBoost의 F1(0.306)이
   LightGBM(0.287)보다 높아 보였는데, 그건 두 모델 다 0.5라는 임계값이 안 맞아서 생긴 착시였다.
   폐업 비율이 10.6%로 낮다 보니 두 모델의 F1 최적 임계값도 0.5와 한참 떨어진 0.21~0.26 근처였고,
   그 지점에서 다시 재보니 **LightGBM이 precision·recall·f1 전부에서 CatBoost를 앞선다**
   (f1 0.399 vs 0.385).
2. **ROC-AUC/PR-AUC 같은 임계값 무관 지표에서도 이번엔 LightGBM이 앞섰다** (roc_auc 0.7476 vs
   0.7431, pr_auc 0.405 vs 0.397). 지난번 기본 비교에서는 두 값이 사실상 동률(0.7473 vs 0.7472)이었는데,
   이번엔 차이가 좀 더 벌어졌다 — early stopping을 위해 train의 10%를 검증셋으로 떼어내면서
   실제 학습에 쓴 데이터가 줄었고, 그 영향을 CatBoost가 더 크게 받았을 가능성이 있다(1회성 분할이라
   노이즈일 수도 있음, 딱 한 번의 train/val split이라 완전히 확정적인 차이라 보긴 이르다).
3. **결론: 이번 데이터·피처 구성에서는 LightGBM 손을 들어주는 게 맞아 보인다.** 임계값 무관 지표와
   임계값 최적화 후 지표 둘 다 LightGBM이 우세했고, 학습 속도도 CatBoost보다 훨씬 빠르다
   (CatBoost는 고카디널리티 범주형 때문에 폴드당 최대 1000회 반복을 다 채우는 경우가 많아 폴드 하나에
   20분 이상, 5-fold 전체에 2시간 넘게 걸렸다).

## 다음 단계 제안

- 서빙/평가 시 기본 0.5가 아니라 F1 최적 임계값(LightGBM 기준 약 0.206)을 쓰는 걸 검토할 것.
  다만 비즈니스 목적(재현율을 더 중시할지, 정밀도를 더 중시할지)에 따라 임계값을 다시 조정해야 할 수 있음.
- CatBoost는 고카디널리티 범주형(`dong_code`, `industry_code`) 처리 비용이 커서 이 프로젝트 규모(189만 행)에는
  비효율적 — 이후 실험은 LightGBM 중심으로 진행하는 게 시간 대비 효율이 좋다.
