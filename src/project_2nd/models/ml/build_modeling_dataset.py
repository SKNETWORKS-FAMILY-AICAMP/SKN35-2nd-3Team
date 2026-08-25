"""
models/ml/build_modeling_dataset.py

지금까지 만든 모든 피처를 합쳐서 학습용 최종 데이터셋을 조립한다.
  - store_snapshots (라벨 available=True만)
  - spatial_density_features (같은 store_id+snapshot_date로 조인)
  - population_features (dong_code로 조인)
  - industries (업종명 붙이기)
  - GroupKFold(store_id 해시 기반, K=5) fold 배정
  - fold-safe 타겟 인코딩: industry_historical_rate, dong_historical_rate,
    dong_industry_historical_rate (각 fold의 값은 해당 fold를 제외한 데이터로만 계산 -> 데이터 누수 방지)
  - store_age_months (stores.csv의 first_seen_snapshot 기준)
  - previously_transitioned (industry_transitions.csv 기준)
  - keyword_growth_score (trend_keywords.csv의 growth_rate를 상호명 매칭에 반영)

입력: data/features/{store_snapshots, spatial_density_features, population_features,
      industries, stores, industry_transitions, trend_keywords}.csv
출력: data/features/modeling_dataset.csv
"""
import pandas as pd
import numpy as np
import hashlib

FEATURES_DIR = 'data/features'
ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']
snap_idx = {s: i for i, s in enumerate(ORDER)}


def fold_of(store_id, k=5):
    h = hashlib.md5(store_id.encode()).hexdigest()
    return int(h, 16) % k


print("loading base tables ...")
snapshots = pd.read_csv(f'{FEATURES_DIR}/store_snapshots.csv',
                         dtype={'store_id': str, 'dong_code': str,
                                'industry_code': str, 'snapshot_date': str})
snapshots = snapshots[snapshots['label_available'] == True].reset_index(drop=True)

spatial = pd.read_csv(f'{FEATURES_DIR}/spatial_density_features.csv',
                       dtype={'store_id': str, 'snapshot_date': str})
pop = pd.read_csv(f'{FEATURES_DIR}/population_features.csv', dtype={'dong_code': str})
industries = pd.read_csv(f'{FEATURES_DIR}/industries.csv', dtype=str)
stores = pd.read_csv(f'{FEATURES_DIR}/stores.csv', dtype={'store_id': str})
transitions = pd.read_csv(f'{FEATURES_DIR}/industry_transitions.csv', dtype=str)
trend_kw = pd.read_csv(f'{FEATURES_DIR}/trend_keywords.csv')

df = snapshots.merge(spatial, on=['store_id', 'snapshot_date'], how='left')
df = df.merge(pop.drop(columns=['dong_name']), on='dong_code', how='left')
df = df.merge(industries[['industry_code', 'industry_name', 'industry_jung_code',
                           'industry_jung_name', 'industry_dae_code', 'custom_group']],
              on='industry_code', how='left')
df = df.rename(columns={'custom_group': 'industry_group'})

# 매장나이
first_seen_map = dict(zip(stores['store_id'], stores['first_seen_snapshot']))
df['first_seen'] = df['store_id'].map(first_seen_map).astype(str)
df['store_age_months'] = (df['snapshot_date'].map(snap_idx) - df['first_seen'].map(snap_idx)) * 6
df = df.drop(columns=['first_seen'])

# 과거 업종전환 이력 여부
transitioned_ever = set(transitions['store_id'].unique())
df['previously_transitioned'] = df['store_id'].isin(transitioned_ever).astype(int)

# 트렌드 키워드 증가율 점수
kw_growth = dict(zip(trend_kw['keyword'], trend_kw['growth_rate'].fillna(0)))
kw_score = pd.Series(0.0, index=df.index)
for kw, growth in kw_growth.items():
    if pd.isna(growth):
        continue
    is_match = df['store_name'].str.contains(kw, na=False, regex=False)
    kw_score = np.where(is_match & (growth > kw_score), growth, kw_score)
df['keyword_growth_score'] = kw_score

# fold 배정
df['fold'] = df['store_id'].apply(fold_of)
df['is_closed_next'] = df['is_closed_next'].astype(str).map({'True': 1, 'False': 0}).fillna(df['is_closed_next']).astype(int)

# fold-safe 과거 폐업률 (industry / dong / dong+industry 조합)
global_rate = df['is_closed_next'].mean()
df['industry_historical_rate'] = np.nan
df['dong_historical_rate'] = np.nan
df['dong_industry_historical_rate'] = np.nan
df['dong_industry_key'] = df['dong_code'] + '_' + df['industry_code']

for k in sorted(df['fold'].unique()):
    train_mask = df['fold'] != k
    test_mask = df['fold'] == k

    ind_rate = df.loc[train_mask].groupby('industry_code')['is_closed_next'].mean()
    dong_rate = df.loc[train_mask].groupby('dong_code')['is_closed_next'].mean()
    di_rate = df.loc[train_mask].groupby('dong_industry_key')['is_closed_next'].mean()
    di_count = df.loc[train_mask].groupby('dong_industry_key')['is_closed_next'].count()
    di_rate_reliable = di_rate.where(di_count >= 30)  # 표본 30건 미만은 신뢰도 낮음 -> 업종 전체 비율로 대체

    df.loc[test_mask, 'industry_historical_rate'] = df.loc[test_mask, 'industry_code'].map(ind_rate)
    df.loc[test_mask, 'dong_historical_rate'] = df.loc[test_mask, 'dong_code'].map(dong_rate)
    df.loc[test_mask, 'dong_industry_historical_rate'] = df.loc[test_mask, 'dong_industry_key'].map(di_rate_reliable)

df['industry_historical_rate'] = df['industry_historical_rate'].fillna(global_rate)
df['dong_historical_rate'] = df['dong_historical_rate'].fillna(global_rate)
df['dong_industry_historical_rate'] = df['dong_industry_historical_rate'].fillna(df['industry_historical_rate'])
df = df.drop(columns=['dong_industry_key', 'label_available'])

# 스코프 확정: 과학·기술/부동산/시설관리·임대는 소비자 대면 업종과 폐업 패턴이 이질적이고
# (폐업률이 전체 평균의 절반 수준) 서울시 매출 데이터에도 커버리지가 없어 제외한다.
# 외부 경제지표 4종(매출/상권임대료/표준지공시지가/개별공시지가)은 각각 검증했으나
# dong_code + dong_industry_historical_rate와 중복되어 성능 개선이 없어 최종안에서 제외했다.
EXCLUDED_GROUPS = ['과학·기술', '부동산', '시설관리·임대']
before = len(df)
df = df[~df['industry_group'].isin(EXCLUDED_GROUPS)].reset_index(drop=True)
print(f"스코프 제외: {before:,} -> {len(df):,} ({before - len(df):,}행 제거, {EXCLUDED_GROUPS})")

final_cols = ['snapshot_date', 'store_id', 'industry_dae_code', 'industry_group',
              'industry_jung_code', 'industry_jung_name', 'industry_code', 'industry_name',
              'gu_name', 'dong_code', 'lng', 'lat', 'floor_category',
              'same_industry_count_300m', 'total_count_300m', 'nearest_same_industry_distance_m',
              'dong_industry_count', 'coord_cluster_size', 'store_age_months', 'previously_transitioned',
              'keyword_growth_score', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
              'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate',
              'industry_historical_rate', 'dong_historical_rate', 'dong_industry_historical_rate',
              'transitioned_next', 'fold', 'is_closed_next']

df[final_cols].to_csv(f'{FEATURES_DIR}/modeling_dataset.csv', index=False, encoding='utf-8-sig')
print(f"saved modeling_dataset.csv, rows={len(df):,}, cols={len(final_cols)}")