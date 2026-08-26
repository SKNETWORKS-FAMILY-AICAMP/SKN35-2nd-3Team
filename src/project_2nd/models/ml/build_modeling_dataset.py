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
from sklearn.neighbors import BallTree

FEATURES_DIR = 'data/features'
ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']
snap_idx = {s: i for i, s in enumerate(ORDER)}

# 서울시 25개 자치구 표준 코드 (행정동코드 앞 5자리). population_features.csv는
# 생활인구 원본 3개 CSV에 등장하는 행정동만 포함하므로, 상가업소 데이터에는
# 있지만 생활인구 원본에는 없는 행정동(실제 검증: 12개 동, 약 3만행)은 population과의
# 조인이 통째로 실패해 gu_name까지 결측이 된다. gu_name은 dong_code만으로도
# 코드 체계상 100% 확정되는 값이라, 조인 결과와 무관하게 이 표로 보강한다.
GU_CODE_MAP = {
    '11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구',
    '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구',
    '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구',
    '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구',
    '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구',
    '11740': '강동구',
}


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

# gu_name 보강: population_features.csv에 없는 행정동이라 조인이 실패해도,
# dong_code 앞 5자리 -> 자치구 매핑은 항상 성립하므로 여기서 100% 채운다.
n_gu_missing_before = df['gu_name'].isna().sum()
df['gu_name'] = df['gu_name'].fillna(df['dong_code'].str[:5].map(GU_CODE_MAP))
n_gu_missing_after = df['gu_name'].isna().sum()
if n_gu_missing_before > 0:
    print(f"gu_name 결측 {n_gu_missing_before:,}건 중 "
          f"{n_gu_missing_before - n_gu_missing_after:,}건을 자치구코드로 보강 "
          f"(남은 결측 {n_gu_missing_after:,}건)")

# 생활인구 피처(korean_pop 등)는 dong_code 자체가 population_features.csv에 없으면
# 코드로 보강할 수 없는 진짜 결측이다. features/spatial/build_spatial_features.py에서
# 이미 쓰고 있는 BallTree(haversine) 방식을 재사용해서, 인구 데이터가 없는 동의
# 매장 좌표 중심점(centroid)을 구한 뒤 "가장 가까운, 인구 데이터가 있는 동"의
# 값을 근사치로 빌려온다(정밀한 폴리곤 매칭은 아니지만, 인접 동의 인구 수준을
# 대리값으로 쓰는 것은 합리적인 근사다).
POP_FEATURE_COLS = ['korean_pop', 'foreign_long_pop', 'foreign_short_pop',
                     'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
pop_missing_mask = df['korean_pop'].isna()
df['population_is_proxied'] = False

if pop_missing_mask.any():
    missing_dongs = sorted(df.loc[pop_missing_mask, 'dong_code'].unique())
    known_dongs = sorted(df.loc[~pop_missing_mask, 'dong_code'].unique())
    print(f"⚠ 생활인구 피처 결측 {pop_missing_mask.sum():,}행 "
          f"({len(missing_dongs)}개 동, population_features.csv에 해당 dong_code 없음): "
          f"{missing_dongs}")

    # 동별 매장 좌표 중심점(centroid) 계산 (라디안 변환, haversine용)
    centroids = df.groupby('dong_code')[['lat', 'lng']].mean()
    known_centroids = centroids.loc[known_dongs]
    missing_centroids = centroids.loc[missing_dongs]

    tree = BallTree(np.radians(known_centroids[['lat', 'lng']].values), metric='haversine')
    dist, idx = tree.query(np.radians(missing_centroids[['lat', 'lng']].values), k=1)
    nearest_dong_map = dict(zip(missing_dongs, known_centroids.index[idx.flatten()]))
    nearest_dist_km = dict(zip(missing_dongs, (dist.flatten() * 6371)))  # 지구 반경(km)

    # 동별 인구 피처값(known dong 기준 1행)을 조회용 테이블로 준비
    pop_by_dong = df.loc[~pop_missing_mask].drop_duplicates('dong_code').set_index('dong_code')[POP_FEATURE_COLS]

    for missing_dong, nearest_dong in nearest_dong_map.items():
        row_mask = df['dong_code'] == missing_dong
        df.loc[row_mask, POP_FEATURE_COLS] = pop_by_dong.loc[nearest_dong].values
        df.loc[row_mask, 'population_is_proxied'] = True
        print(f"  {missing_dong} -> 최근접 {nearest_dong} 값으로 대체 "
              f"(중심점간 거리 {nearest_dist_km[missing_dong]:.2f}km, {row_mask.sum():,}행)")

    print(f"population_is_proxied=True: {df['population_is_proxied'].sum():,}행 "
          f"(최근접 동 값으로 대체한 근사치 — 결과서에 한계로 명시 필요)")

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
              'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate', 'population_is_proxied',
              'industry_historical_rate', 'dong_historical_rate', 'dong_industry_historical_rate',
              'transitioned_next', 'fold', 'is_closed_next']

df[final_cols].to_csv(f'{FEATURES_DIR}/modeling_dataset.csv', index=False, encoding='utf-8-sig')
print(f"saved modeling_dataset.csv, rows={len(df):,}, cols={len(final_cols)}")