"""
features/trend_keywords/build_trend_keywords.py

상호명 실데이터 기반으로 키워드별 스냅샷 매장수와 증가율을 계산한다.
더미/가짜 검색량 데이터는 쓰지 않는다 — 엄밀성 리스크 때문에 이번 프로젝트에서
명시적으로 배제하기로 한 방식이다.

키워드 목록은 예시일 뿐, 팀에서 자유롭게 추가/교체 가능. 음식류에 국한하지 않고
무인매장/오락/카페 콘셉트 등 다양한 업태를 포함하는 게 좋다 (예: 인형뽑기가
실제로 가장 가파른 증가세를 보였음).

입력: data/features/store_snapshots.csv
출력: data/features/trend_keywords.csv
"""
import pandas as pd
import numpy as np

SRC = 'data/features/store_snapshots.csv'
DEST = 'data/features/trend_keywords.csv'

ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']

KEYWORDS = [
    '탕후루', '버터', '마라탕', '흑당', '요거트',      # 음식/디저트
    '무인', '셀프',                                    # 무인/셀프 매장
    '인형뽑기', '인생네컷', '방탈출', '스크린골프',       # 오락/여가
    '만화카페', '애견카페', '스터디카페',                 # 카페 콘셉트
]

counts = {kw: {snap: 0 for snap in ORDER} for kw in KEYWORDS}

chunksize = 300_000
reader = pd.read_csv(SRC, usecols=['snapshot_date', 'store_name'],
                      dtype={'snapshot_date': str}, chunksize=chunksize)

for chunk in reader:
    names = chunk['store_name'].fillna('')
    for kw in KEYWORDS:
        match = names.str.contains(kw, regex=False)
        if match.any():
            g = chunk.loc[match, 'snapshot_date'].value_counts()
            for snap, c in g.items():
                counts[kw][snap] = counts[kw].get(snap, 0) + int(c)

rows = []
for kw in KEYWORDS:
    row = {'keyword': kw}
    row.update(counts[kw])
    rows.append(row)

trend_df = pd.DataFrame(rows).set_index('keyword').reindex(columns=ORDER)
trend_df['growth_rate'] = (trend_df[ORDER[-1]] - trend_df[ORDER[0]]) / trend_df[ORDER[0]].replace(0, np.nan)
trend_df = trend_df.reset_index().sort_values('growth_rate', ascending=False)
trend_df.to_csv(DEST, index=False, encoding='utf-8-sig')
print(trend_df.to_string(index=False))
