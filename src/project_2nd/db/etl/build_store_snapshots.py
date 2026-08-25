"""
db/etl/02_build_store_snapshots.py

원본 6개 스냅샷 CSV + 01번 스크립트의 폐업/전환 라벨을 합쳐서
store_snapshots(스냅샷 단위 이력)와 stores(매장 단위 마스터)를 만든다.

입력: data/raw/*.csv, data/features/closed_ids_by_snap.pkl, transition_by_snap.pkl
출력: data/features/store_snapshots.csv, data/features/stores.csv
"""
import pandas as pd
import pickle
import os
import re

RAW_DIR = 'data/raw'
OUT_DIR = 'data/features'

RAW_FILES = {
    '202312': f'{RAW_DIR}/seoul_202312.csv',
    '202406': f'{RAW_DIR}/seoul_202406.csv',
    '202412': f'{RAW_DIR}/seoul_202412.csv',
    '202506': f'{RAW_DIR}/seoul_202506.csv',
    '202512': f'{RAW_DIR}/seoul_202512.csv',
    '202606': f'{RAW_DIR}/seoul_202606.csv',
}
ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']

ID_COL = '상가업소번호'
NAME_COL = '상호명'
SO_CODE = '상권업종소분류코드'
DONG_CODE = '행정동코드'
LNG_COL, LAT_COL = '경도', '위도'
FLOOR_COL = '층정보'

USE_COLS = [ID_COL, NAME_COL, SO_CODE, DONG_CODE, LNG_COL, LAT_COL, FLOOR_COL]


def categorize_floor(x):
    """
    원본 층정보 문자열을 5개 카테고리로 단순화한다.
    - '1' -> 1층
    - 'B'로 시작 -> 지하
    - 1~2자리 숫자(1 제외) -> 2층이상
    - 그 외(4자리 이상 코드, '지' 등 애매한 값) -> 기타
    - 결측 -> 결측 (실제 검증 결과 폐업률이 가장 높게 나온 카테고리라 별도 유지)
    """
    if pd.isna(x):
        return '결측'
    if x.startswith('B'):
        return '지하'
    if x == '1':
        return '1층'
    if re.fullmatch(r'\d{1,2}', x) and x != '1':
        return '2층이상'
    return '기타'


with open(f'{OUT_DIR}/closed_ids_by_snap.pkl', 'rb') as f:
    closed_ids_by_snap = pickle.load(f)
with open(f'{OUT_DIR}/transition_by_snap.pkl', 'rb') as f:
    transition_by_snap = pickle.load(f)

first_write = True
for snap in ORDER:
    df = pd.read_csv(RAW_FILES[snap], usecols=USE_COLS, dtype=str)
    df[LNG_COL] = df[LNG_COL].astype(float)
    df[LAT_COL] = df[LAT_COL].astype(float)
    df['floor_category'] = df[FLOOR_COL].apply(categorize_floor)

    closed_set = closed_ids_by_snap.get(snap, set())
    trans_map = transition_by_snap.get(snap, {})
    df['is_closed_next'] = df[ID_COL].isin(closed_set)
    df['transitioned_next'] = df[ID_COL].isin(trans_map.keys())
    df['label_available'] = snap != ORDER[-1]
    df['snapshot_date'] = snap

    out = df.rename(columns={ID_COL: 'store_id', NAME_COL: 'store_name',
                              SO_CODE: 'industry_code', DONG_CODE: 'dong_code',
                              LNG_COL: 'lng', LAT_COL: 'lat'})
    cols = ['snapshot_date', 'store_id', 'store_name', 'industry_code', 'dong_code',
            'lng', 'lat', 'floor_category', 'is_closed_next', 'transitioned_next', 'label_available']
    out[cols].to_csv(f'{OUT_DIR}/store_snapshots.csv', mode='w' if first_write else 'a',
                      header=first_write, index=False, encoding='utf-8-sig')
    first_write = False
    print(f"{snap}: {len(out):,} rows written")
    del df, out

# stores 마스터: store_snapshots를 store_id로 집계
panel = pd.read_csv(f'{OUT_DIR}/store_snapshots.csv',
                     usecols=['snapshot_date', 'store_id', 'industry_code', 'dong_code', 'is_closed_next'],
                     dtype=str)
panel_sorted = panel.sort_values(['store_id', 'snapshot_date'])
first_seen = panel_sorted.groupby('store_id')['snapshot_date'].first().rename('first_seen_snapshot')
last_row = panel_sorted.groupby('store_id').last()
n_observed = panel.groupby('store_id')['snapshot_date'].nunique().rename('n_snapshots_observed')

stores = pd.concat([first_seen,
                     last_row['snapshot_date'].rename('last_seen_snapshot'),
                     last_row['industry_code'].rename('current_industry_code'),
                     last_row['dong_code'].rename('dong_code'),
                     n_observed], axis=1).reset_index()

# 폐업 판정: 마지막 스냅샷(202606)까지 살아있지 않으면 폐업
stores['is_closed'] = stores['last_seen_snapshot'] != ORDER[-1]

# 일시적 공백(중간에 사라졌다 재등장) 탐지
snap_idx = {s: i for i, s in enumerate(ORDER)}
stores['first_idx'] = stores['first_seen_snapshot'].map(snap_idx)
stores['last_idx'] = stores['last_seen_snapshot'].map(snap_idx)
stores['expected_snapshots'] = stores['last_idx'] - stores['first_idx'] + 1
stores['had_temporary_gap'] = stores['n_snapshots_observed'] < stores['expected_snapshots']
stores = stores.drop(columns=['first_idx', 'last_idx', 'expected_snapshots'])

stores.to_csv(f'{OUT_DIR}/stores.csv', index=False, encoding='utf-8-sig')
print(f"stores.csv: {len(stores):,} rows, 폐업 {stores['is_closed'].sum():,}건")