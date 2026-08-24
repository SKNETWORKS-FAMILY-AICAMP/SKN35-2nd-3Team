"""
features/survival_transition/build_survival_stats.py

industry_transitions.csv를 (from_industry, to_industry) 조합별로 집계해서
전환 생존율을 계산한다. "생존"의 기준은 전환 직후 다음 스냅샷까지
살아있었는지(단기 생존 프록시)이며, 관측 기간이 짧다는 한계가 있다.
표본이 적은 조합은 sample_size로 걸러서 써야 한다.

입력: data/features/industry_transitions.csv, data/features/store_snapshots.csv
출력: data/features/industry_survival_stats.csv
"""
import pandas as pd

FEATURES_DIR = 'data/features'

transitions = pd.read_csv(f'{FEATURES_DIR}/industry_transitions.csv', dtype=str)

panel_small = pd.read_csv(f'{FEATURES_DIR}/store_snapshots.csv',
                           usecols=['snapshot_date', 'store_id', 'is_closed_next', 'label_available'],
                           dtype={'store_id': str, 'snapshot_date': str})

merged = transitions.merge(panel_small, left_on=['store_id', 'to_snapshot'],
                            right_on=['store_id', 'snapshot_date'], how='left')

# 라벨 불가 시점(마지막 스냅샷)으로 전환된 건은 생존 여부를 알 수 없어 제외
merged = merged[merged['label_available'].astype(str) == 'True']
merged['is_closed_next'] = merged['is_closed_next'].astype(str) == 'True'
merged['survived'] = ~merged['is_closed_next']

survival_stats = merged.groupby(['from_industry_code', 'to_industry_code']).agg(
    sample_size=('store_id', 'count'),
    survival_rate=('survived', 'mean')
).reset_index().sort_values('sample_size', ascending=False)

survival_stats.to_csv(f'{FEATURES_DIR}/industry_survival_stats.csv', index=False, encoding='utf-8-sig')
print(f"industry_survival_stats.csv: {len(survival_stats):,}개 조합")
