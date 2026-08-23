"""
models/ml/03_tune_threshold.py

02번에서 학습한 모델의 기본 0.5 임계값 대신, 검증셋에서 F1이 최대가 되는
임계값을 찾아 테스트셋에 적용해본다. 클래스 불균형(폐업 비율 ~9%) 상황에서는
임계값에 따라 Precision/Recall 트레이드오프가 크게 갈리므로, 팀에서 정한
비즈니스 기준(Recall 우선 vs Precision-Recall 균형)에 맞춰 최종 임계값을 정한다.

참고용으로 "상위 N% 위험군만 본다면" 관점의 지표도 같이 출력한다 —
관리자가 케어할 수 있는 인원이 제한적일 때 더 실전적인 기준이 될 수 있다.

입력: data/features/modeling_dataset.csv
출력: models/ml/saved/lightgbm_tuned.txt
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, precision_recall_curve)

SRC = 'data/features/modeling_dataset.csv'
MODEL_OUT = 'models/ml/saved/lightgbm_tuned.txt'

df = pd.read_csv(SRC, dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                              'industry_dae_code': str, 'industry_jung_code': str,
                              'snapshot_date': str})

cat_cols = ['industry_dae_code', 'industry_jung_code', 'industry_code', 'gu_name', 'dong_code']
for c in cat_cols:
    df[c] = df[c].astype('category')

feature_cols = ['industry_dae_code', 'industry_jung_code', 'industry_code', 'gu_name', 'dong_code',
                'lng', 'lat', 'same_industry_count_300m', 'total_count_300m',
                'nearest_same_industry_distance_m', 'dong_industry_count',
                'store_age_months', 'previously_transitioned', 'keyword_growth_score',
                'korean_pop', 'foreign_long_pop', 'foreign_short_pop', 'total_pop_avg',
                'foreign_short_ratio', 'tourist_zone_candidate',
                'industry_historical_rate', 'dong_historical_rate', 'dong_industry_historical_rate']

val_fold, test_fold = 3, 4
train = df[~df['fold'].isin([val_fold, test_fold])]
val = df[df['fold'] == val_fold]
test = df[df['fold'] == test_fold]

X_train, y_train = train[feature_cols], train['is_closed_next']
X_val, y_val = val[feature_cols], val['is_closed_next']
X_test, y_test = test[feature_cols], test['is_closed_next']

model = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
    class_weight='balanced', random_state=42, verbosity=-1
)
model.fit(X_train, y_train, categorical_feature=cat_cols,
          eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])

val_proba = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f1s[:-1])]
print(f"검증셋 기준 최적 임계값(F1 최대화): {best_threshold:.4f}")

test_proba = model.predict_proba(X_test)[:, 1]
for th, label in [(0.5, "기본 임계값 0.5"), (best_threshold, f"튜닝 임계값 {best_threshold:.3f}")]:
    pred = (test_proba >= th).astype(int)
    print(f"\n=== {label} ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1       : {f1_score(y_test, pred):.4f}")

print(f"\nROC-AUC (임계값 무관): {roc_auc_score(y_test, test_proba):.4f}")

# 상위 N% 위험군만 케어 가능하다는 가정의 실전적 지표
top10_cut = np.quantile(test_proba, 0.90)
top10_mask = test_proba >= top10_cut
print(f"\n상위 10% 위험군 기준: Precision={y_test[top10_mask].mean():.4f}, "
      f"Recall={y_test[top10_mask].sum()/y_test.sum():.4f}")

model.booster_.save_model(MODEL_OUT)
print(f"\nsaved {MODEL_OUT}")
