"""
db/etl/01_build_closure_transitions.py

원본 소상공인 CSV 6개 스냅샷을 순차 비교해서:
  - 폐업 라벨 (다음 스냅샷에서 사라졌는지)
  - 업종전환 이력 (다음 스냅샷으로 갈 때 업종 코드가 바뀌었는지)
을 계산한다. 메모리 절약을 위해 store_id -> industry_code 매핑만 유지하며
인접한 두 스냅샷씩 롤링으로 비교한다 (전체 스냅샷을 동시에 메모리에 올리지 않음).

입력: data/raw/*.csv (6개 스냅샷)
출력:
  - data/features/closed_ids_by_snap.pkl   (다음 스냅샷용 중간 산출물)
  - data/features/transition_by_snap.pkl
  - data/features/industry_transitions.csv
"""
import pandas as pd
import pickle
import os

RAW_DIR = 'data/raw'
OUT_DIR = 'data/features'
os.makedirs(OUT_DIR, exist_ok=True)

# 실제 파일명에 맞게 수정
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
SO_CODE = '상권업종소분류코드'

closed_ids_by_snap = {}
transition_by_snap = {}
transitions_records = []

prev_map = None
prev_snap = None
for snap in ORDER:
    df = pd.read_csv(RAW_FILES[snap], usecols=[ID_COL, SO_CODE], dtype=str)
    cur_map = dict(zip(df[ID_COL], df[SO_CODE]))
    del df

    if prev_map is not None:
        prev_ids = set(prev_map.keys())
        cur_ids = set(cur_map.keys())
        closed_at_prev = prev_ids - cur_ids
        closed_ids_by_snap[prev_snap] = closed_at_prev

        common = prev_ids & cur_ids
        trans_map = {}
        for sid in common:
            fc, tc = prev_map[sid], cur_map[sid]
            if fc != tc:
                trans_map[sid] = (fc, tc)
                transitions_records.append({
                    'store_id': sid, 'from_snapshot': prev_snap, 'to_snapshot': snap,
                    'from_industry_code': fc, 'to_industry_code': tc
                })
        transition_by_snap[prev_snap] = trans_map
        print(f"  {prev_snap}->{snap}: closed={len(closed_at_prev):,} transitioned={len(trans_map):,}")

    prev_map = cur_map
    prev_snap = snap

# 마지막 스냅샷은 다음이 없어 라벨 불가 (서빙 전용)
closed_ids_by_snap[ORDER[-1]] = set()
transition_by_snap[ORDER[-1]] = {}

transitions_df = pd.DataFrame(transitions_records)
transitions_df.insert(0, 'transition_id', range(1, len(transitions_df) + 1))
transitions_df.to_csv(f'{OUT_DIR}/industry_transitions.csv', index=False, encoding='utf-8-sig')
print(f"saved industry_transitions.csv rows={len(transitions_df):,}")

with open(f'{OUT_DIR}/closed_ids_by_snap.pkl', 'wb') as f:
    pickle.dump(closed_ids_by_snap, f)
with open(f'{OUT_DIR}/transition_by_snap.pkl', 'wb') as f:
    pickle.dump(transition_by_snap, f)
print("done")
