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

# 서울시 25개 자치구 표준 코드 (행정동코드 앞 5자리). 생활인구 원본의 행정동코드가
# 소상공인 원본 CSV의 행정동 목록에 없는 경우(실제 검증: 9개 동) gu_name을
# 못 붙이므로, 표준 코드표로 fallback 채운다. dong_name(동 이름)까지는 코드만으로
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

# 행정동명/구명 붙이기 (소상공인 원본 CSV에서 조회)
dong_names = pd.read_csv('data/raw/seoul_202606.csv',
                          usecols=['행정동코드', '행정동명', '시군구명'], dtype=str)
dong_names = dong_names.drop_duplicates(subset=['행정동코드']).rename(
    columns={'행정동코드': 'dong_code', '행정동명': 'dong_name', '시군구명': 'gu_name'})

pop = pop.merge(dong_names, on='dong_code', how='left')

# 소상공인 목록에 없어 gu_name을 못 붙인 행정동은 표준 코드표로 fallback 채우기
n_missing_before = pop['gu_name'].isna().sum()
pop['gu_name'] = pop['gu_name'].fillna(pop['dong_code'].str[:5].map(GU_CODE_MAP))
n_missing_after = pop['gu_name'].isna().sum()
if n_missing_before > 0:
    print(f"gu_name 결측 {n_missing_before}건 중 {n_missing_before - n_missing_after}건을 "
          f"표준 자치구코드로 채움 (남은 결측 {n_missing_after}건)")

cols = ['dong_code', 'dong_name', 'gu_name', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
        'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
pop = pop[cols].sort_values('foreign_short_ratio', ascending=False)
pop.to_csv(DEST, index=False, encoding='utf-8-sig')
print(f"population_features.csv: {len(pop)}개 행정동, 관광특구 후보 임계값={threshold:.4f}")