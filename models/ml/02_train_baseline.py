"""
models/ml/02_train_baseline.py

modeling_dataset.csv로 LightGBM 베이스라인을 학습하고 평가한다.
GroupKFold(store_id 기준) fold 4를 테스트셋으로 고정해서 사용한다.

입력: data/features/modeling_dataset.csv
출력: models/ml/saved/lightgbm_baseline.txt
"""
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

SRC = 'data/features/modeling_dataset.csv'
MODEL_OUT = 'models/ml/saved/lightgbm_baseline.txt'

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

test_fold = 4
train = df[df['fold'] != test_fold]
test = df[df['fold'] == test_fold]

X_train, y_train = train[feature_cols], train['is_closed_next']
X_test, y_test = test[feature_cols], test['is_closed_next']

print(f"train={len(X_train):,}  test={len(X_test):,}")
print(f"train/test store_id 중복(0이어야 함): {len(set(train['store_id']) & set(test['store_id']))}")

model = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, class_weight='balanced', random_state=42, verbosity=-1
)
model.fit(X_train, y_train, categorical_feature=cat_cols)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
base_rate = y_test.mean()

print()
print(f"기저 폐업률(base rate): {base_rate:.4f}")
print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"Precision: {precision_score(y_test, pred):.4f}")
print(f"Recall   : {recall_score(y_test, pred):.4f}")
print(f"F1       : {f1_score(y_test, pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")
print(f"PR-AUC   : {average_precision_score(y_test, proba):.4f} (무작위 기준선={base_rate:.4f})")

imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print()
print("Feature importance:")
print(imp)

model.booster_.save_model(MODEL_OUT)
print(f"\nsaved {MODEL_OUT}")
