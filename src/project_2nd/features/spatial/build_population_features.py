"""
features/spatial/build_population_features.py

생활인구 3개 CSV(내국인/외국인장기/외국인단기)를 행정동 단위로 평균낸다.
실제 데이터를 열어보면 하루치가 아니라 한 달치(31일 x 24시간대) 데이터이므로,
전체 시간대·전체 일자를 평균해서 "평상시 생활인구" 대표값을 만든다.

주의: 원본 CSV는 헤더보다 실제 데이터 컬럼이 1개 더 많다(끝에 빈 컬럼).
pandas 기본 설정으로 읽으면 컬럼이 한 칸씩 밀려서 잘못 매칭되므로
반드시 index_col=False를 지정해야 한다.

입력: data/raw/*.csv
출력: data/features/population_features.csv
"""
import pandas as pd

RAW_DIR = 'data/raw'
DEST = 'data/features/population_features.csv'

FILES = {
    'korean_pop': f'{RAW_DIR}/local_pop.csv',
    'foreign_long_pop': f'{RAW_DIR}/longf_pop.csv',
    'foreign_short_pop': f'{RAW_DIR}/tempf_pop.csv',
}

# 소상공인 원본 6개 스냅샷 (db/etl/build_store_snapshots.py, build_closure_transitions.py,
# features/spatial/build_spatial_features.py 등 다른 스크립트와 동일한 RAW_FILES/ORDER 관례).
# 동 이름 조회를 스냅샷 1개(예: 최신 202606)에만 의존하면, 그 시점에 매장이 하나도
# 없던(폐업/신설 등) 행정동은 이름을 못 붙인다. 6개 스냅샷을 전부 합치면 어느 한
# 시점에라도 매장이 있었던 동은 이름을 확보할 수 있다.
RAW_FILES = {
    '202312': f'{RAW_DIR}/seoul_202312.csv',
    '202406': f'{RAW_DIR}/seoul_202406.csv',
    '202412': f'{RAW_DIR}/seoul_202412.csv',
    '202506': f'{RAW_DIR}/seoul_202506.csv',
    '202512': f'{RAW_DIR}/seoul_202512.csv',
    '202606': f'{RAW_DIR}/seoul_202606.csv',
}
ORDER = ['202312', '202406', '202412', '202506', '202512', '202606']

# 서울시 25개 자치구 표준 코드 (행정동코드 앞 5자리). 6개 스냅샷을 다 합쳐도
# 소상공인 데이터에 한 번도 등장하지 않는 행정동은 gu_name을 못 붙이므로,
# 표준 코드표로 fallback 채운다. dong_name(동 이름)까지는 코드만으로
# 유추 불가능하지만, 파이프라인 어디서도 조인키로 쓰지 않아 결측으로 남겨도 무방하다.
GU_CODE_MAP = {
    '11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구',
    '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구',
    '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구',
    '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구',
    '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구',
    '11740': '강동구',
}

dfs = {}
for col_name, path in FILES.items():
    df = pd.read_csv(path, usecols=['행정동코드', '총생활인구수'], dtype={'행정동코드': str},
                      encoding='utf-8-sig', index_col=False)
    df['총생활인구수'] = df['총생활인구수'].astype(float)
    dfs[col_name] = df.groupby('행정동코드')['총생활인구수'].mean().rename(col_name)
    del df

pop = pd.concat(dfs.values(), axis=1).reset_index().rename(columns={'행정동코드': 'dong_code'})
pop['total_pop_avg'] = pop['korean_pop'] + pop['foreign_long_pop'].fillna(0) + pop['foreign_short_pop'].fillna(0)
pop['foreign_short_ratio'] = pop['foreign_short_pop'] / pop['total_pop_avg']

# 관광특구 후보 플래그: 단기외국인 비율 상위 10% (임계값은 팀에서 조정 가능)
threshold = pop['foreign_short_ratio'].quantile(0.90)
pop['tourist_zone_candidate'] = pop['foreign_short_ratio'] >= threshold

# 행정동명/구명 붙이기 (소상공인 원본 CSV 6개 스냅샷 전체에서 조회)
# 스냅샷 1개만 쓰면 그 시점에 매장이 없던 동이 누락되므로, 전체 스냅샷을
# 합쳐 "어느 한 시점에라도 등장한 적 있는" 행정동코드-이름 매핑을 최대한 확보한다.
dong_name_frames = []
for snap in ORDER:
    frame = pd.read_csv(RAW_FILES[snap], usecols=['행정동코드', '행정동명', '시군구명'], dtype=str)
    dong_name_frames.append(frame)

dong_names = pd.concat(dong_name_frames, ignore_index=True)
dong_names = dong_names.drop_duplicates(subset=['행정동코드']).rename(
    columns={'행정동코드': 'dong_code', '행정동명': 'dong_name', '시군구명': 'gu_name'})

print(f"동 이름 조회에 스냅샷 {len(ORDER)}개 사용, "
      f"고유 행정동코드 {dong_names['dong_code'].nunique()}개 확보")

pop = pop.merge(dong_names, on='dong_code', how='left')

# 6개 스냅샷을 다 합쳐도 매칭 안 되는 행정동(소상공인 데이터에 한 번도 등장하지
# 않은 동)만 표준 코드표로 gu_name을 fallback 채운다.
n_missing_before = pop['gu_name'].isna().sum()
pop['gu_name'] = pop['gu_name'].fillna(pop['dong_code'].str[:5].map(GU_CODE_MAP))
n_missing_after = pop['gu_name'].isna().sum()
if n_missing_before > 0:
    print(f"gu_name 결측 {n_missing_before}건 중 {n_missing_before - n_missing_after}건을 "
          f"표준 자치구코드로 채움 (남은 결측 {n_missing_after}건)")

n_dong_name_missing = pop['dong_name'].isna().sum()
if n_dong_name_missing > 0:
    print(f"dong_name 결측 {n_dong_name_missing}건 남음 "
          f"(6개 스냅샷 전체에도 등장하지 않은 행정동 — 조인키로 미사용이라 무방)")

cols = ['dong_code', 'dong_name', 'gu_name', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
        'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
pop = pop[cols].sort_values('foreign_short_ratio', ascending=False)
pop.to_csv(DEST, index=False, encoding='utf-8-sig')
print(f"population_features.csv: {len(pop)}개 행정동, 관광특구 후보 임계값={threshold:.4f}")