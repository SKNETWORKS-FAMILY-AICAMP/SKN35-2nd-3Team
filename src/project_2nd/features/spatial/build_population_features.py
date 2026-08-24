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

cols = ['dong_code', 'dong_name', 'gu_name', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
        'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
pop = pop[cols].sort_values('foreign_short_ratio', ascending=False)
pop.to_csv(DEST, index=False, encoding='utf-8-sig')
print(f"population_features.csv: {len(pop)}개 행정동, 관광특구 후보 임계값={threshold:.4f}")
