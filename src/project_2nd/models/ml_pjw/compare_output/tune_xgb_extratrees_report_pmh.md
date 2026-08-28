# XGBoost/ExtraTrees Optuna 튜닝 + 임계값 최적화 (pmh 데이터)

train=fold(0,1,2) 1,137,077행 · validation=fold(3) 375,496행 · test=fold(4) 377,009행

목적함수: validation PR-AUC (팀 표준 1순위 지표)

## 튜닝 결과 (validation PR-AUC 기준)

- XGBoost: 40 trials, best val PR-AUC = 0.4095
  - best params: {'n_estimators': 1197, 'max_depth': 8, 'learning_rate': 0.024697621004280745, 'subsample': 0.9746143876826797, 'colsample_bytree': 0.7678121320177449, 'min_child_weight': 13, 'reg_lambda': 0.06113763603077497, 'reg_alpha': 0.09882889536433334, 'scale_pos_weight': 1.0962295661396568}
- ExtraTrees: 15 trials, best val PR-AUC = 0.4115
  - best params: {'n_estimators': 280, 'max_depth': 22, 'min_samples_leaf': 35, 'min_samples_split': 10, 'max_features': 0.8175392099040567}

## 승자: **extratrees** (validation PR-AUC 더 높은 쪽)

## 임계값 최적화 (validation, F1 기준)

- 최적 임계값: **0.655** (validation F1=0.4047)

## 최종 테스트(fold 4, 최초 1회 확인) — 0.5 임계값 vs 최적 임계값

| 지표 | 0.5 임계값 | 최적 임계값(0.655) |
|---|---|---|
| accuracy | 0.7883 | 0.8884 |
| precision | 0.2590 | 0.4662 |
| recall | 0.5341 | 0.3576 |
| f1 | 0.3488 | 0.4048 |
| roc_auc | 0.7479 | 0.7479 |
| pr_auc | 0.4117 | 0.4117 |
