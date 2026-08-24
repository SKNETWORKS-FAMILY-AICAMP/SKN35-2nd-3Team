"""
features/spatial/build_spatial_features.py

BallTree(haversine)로 반경 300m 내 밀도 피처 4종을 계산한다:
  - same_industry_count_300m           : 반경 300m 내 동일업종 매장 수
  - total_count_300m                   : 반경 300m 내 전체 업종 매장 수
  - nearest_same_industry_distance_m   : 가장 가까운 동일업종 매장까지 거리(m)
  - dong_industry_count                : 행정동 전체 기준 동일업종 매장 수 (반경 아님)

기존점주 앱에서는 경쟁 리스크 피처로, 예비창업자 앱에서는 추천도 피처로
same_industry_count_300m 하나를 그대로 재사용한다.

입력: data/features/store_snapshots.csv (db/etl/02_build_store_snapshots.py 산출물)
출력: data/features/spatial_density_features.csv
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

SRC = 'data/features/store_snapshots.csv'
DEST = 'data/features/spatial_density_features.csv'

RADIUS_M = 300
EARTH_R = 6371000
RADIUS_RAD = RADIUS_M / EARTH_R

ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']

first_write = True
for snap in ORDER:
    df = pd.read_csv(SRC, dtype={'store_id': str, 'dong_code': str, 'snapshot_date': str})
    df = df[df['snapshot_date'] == snap].reset_index(drop=True)
    if len(df) == 0:
        continue

    # 전체 밀도 (업종 무관)
    all_rad = np.radians(df[['lat', 'lng']].to_numpy())
    tree_all = BallTree(all_rad, metric='haversine')
    df['total_count_300m'] = tree_all.query_radius(all_rad, r=RADIUS_RAD, count_only=True) - 1

    # 업종별 그룹: 동일업종 카운트 + 최근접 거리
    same_counts = np.zeros(len(df), dtype=np.int32)
    nearest_dist = np.full(len(df), np.nan)
    for code, idx in df.groupby('industry_code').groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        sub_rad = np.radians(df.loc[idx, ['lat', 'lng']].to_numpy())
        tree = BallTree(sub_rad, metric='haversine')
        cnt = tree.query_radius(sub_rad, r=RADIUS_RAD, count_only=True)
        same_counts[idx] = cnt - 1
        dist, _ = tree.query(sub_rad, k=2)
        nearest_dist[idx] = dist[:, 1] * EARTH_R
    df['same_industry_count_300m'] = same_counts
    df['nearest_same_industry_distance_m'] = nearest_dist

    # 행정동 단위 동일업종 카운트
    df['dong_industry_count'] = df.groupby(['dong_code', 'industry_code'])['store_id'].transform('count')

    out_cols = ['store_id', 'snapshot_date', 'same_industry_count_300m', 'total_count_300m',
                'nearest_same_industry_distance_m', 'dong_industry_count']
    df[out_cols].to_csv(DEST, mode='w' if first_write else 'a',
                         header=first_write, index=False, encoding='utf-8-sig')
    first_write = False
    print(f"{snap}: {len(df):,} rows written")
    del df

print("done")
