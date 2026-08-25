"""
features/spatial/build_spatial_features.py

BallTree(haversine)로 반경 300m 내 밀도 피처 4종을 계산하고,
DBSCAN(haversine)으로 "같은 건물/복합상가 안에 몇 개 매장이 있는지"를 계산한다:
  - same_industry_count_300m           : 반경 300m 내 동일업종 매장 수
  - total_count_300m                   : 반경 300m 내 전체 업종 매장 수
  - nearest_same_industry_distance_m   : 가장 가까운 동일업종 매장까지 거리(m)
  - dong_industry_count                : 행정동 전체 기준 동일업종 매장 수 (반경 아님)
  - coord_cluster_size                 : 반경 20m 이내로 서로 연결된 매장들을 하나의
                                          클러스터(=같은 건물/복합상가로 추정)로 묶어 그 안의
                                          유니크 매장 수를 센다 (DBSCAN, min_samples=1)

기존점주 앱에서는 경쟁 리스크 피처로, 예비창업자 앱에서는 추천도 피처로
same_industry_count_300m 하나를 그대로 재사용한다.

주의: 원본 CSV를 좌표 그대로 정확히 일치(exact match)시켜 묶으면, 스냅샷마다
지오코딩이 살짝씩 달라져서(예: 1~2m 차이) 사실은 같은 건물인데 다른 클러스터로
쪼개지는 문제가 있었다(실제 검증: 송파구 한 대형 복합상가에서 발견). DBSCAN은
정확히 일치하지 않아도 반경 안에 있으면 하나로 묶어주므로 이 문제에서 자유롭다.

입력: data/features/store_snapshots.csv (db/etl/02_build_store_snapshots.py 산출물)
출력: data/features/spatial_density_features.csv
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

SRC = 'data/features/store_snapshots.csv'
DEST = 'data/features/spatial_density_features.csv'

RADIUS_M = 300
EARTH_R = 6371000
RADIUS_RAD = RADIUS_M / EARTH_R

CLUSTER_RADIUS_M = 20
CLUSTER_EPS_RAD = CLUSTER_RADIUS_M / EARTH_R

ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']

first_write = True
for snap in ORDER:
    df = pd.read_csv(SRC, dtype={'store_id': str, 'dong_code': str, 'snapshot_date': str})
    df = df[df['snapshot_date'] == snap].reset_index(drop=True)
    if len(df) == 0:
        continue

    all_rad = np.radians(df[['lat', 'lng']].to_numpy())

    # 전체 밀도 (업종 무관)
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

    # 같은 건물/복합상가 매장 수 (DBSCAN, 반경 20m 이내 연결된 점들을 하나의 클러스터로)
    db = DBSCAN(eps=CLUSTER_EPS_RAD, min_samples=1, metric='haversine').fit(all_rad)
    df['building_cluster_id'] = db.labels_
    df['coord_cluster_size'] = df.groupby('building_cluster_id')['store_id'].transform('nunique')

    out_cols = ['store_id', 'snapshot_date', 'same_industry_count_300m', 'total_count_300m',
                'nearest_same_industry_distance_m', 'dong_industry_count', 'coord_cluster_size']
    df[out_cols].to_csv(DEST, mode='w' if first_write else 'a',
                         header=first_write, index=False, encoding='utf-8-sig')
    first_write = False
    print(f"{snap}: {len(df):,} rows written")
    del df

print("done")