"""
features/spatial/build_population_features.py

생활인구 3개 CSV(내국인/외국인장기/외국인단기)를 행정동 단위로 평균낸다.
실제 데이터를 열어보면 하루치가 아니라 한 달치(31일 x 24시간대) 데이터이므로,
전체 시간대·전체 일자를 평균해서 "평상시 생활인구" 대표값을 만든다.

주의: 원본 CSV는 헤더보다 실제 데이터 컬럼이 1개 더 많다(끝에 빈 컬럼).
pandas 기본 설정으로 읽으면 컬럼이 한 칸씩 밀려서 잘못 매칭되므로
반드시 index_col=False를 지정해야 한다.

행정동 이름 마스터(administrative_dongs.csv)도 여기서 같이 만든다. 예전엔 population
쪽 dong_code에만 이름을 붙였는데, 그러면 "생활인구 데이터엔 없지만 상권(매장) 데이터엔
있는 동"(실측 12개, DB FK 적재 단계에서 발견됨)의 이름이 원본 상권 스냅샷에 멀쩡히
있는데도 그냥 버려지고 DB 쪽에서 '(미상)' placeholder로 채워지는 문제가 있었다.
population 쪽 dong_code와 상권 6개 스냅샷 쪽 dong_code를 합집합으로 모아서 마스터를
만들면 그 12개도 진짜 이름을 쓸 수 있다. gu_name은 어느 쪽 데이터에 있든 없든 dong_code
앞 5자리로 100% 결정되는 값이라(models/ml/build_modeling_dataset.py와 동일한 원칙)
조인 성공 여부와 무관하게 항상 코드 기준으로 채운다.

입력: data/raw/*.csv
출력: data/features/population_features.csv, data/features/administrative_dongs.csv
"""
import pandas as pd

RAW_DIR = 'data/raw'
POP_DEST = 'data/features/population_features.csv'
DONGS_DEST = 'data/features/administrative_dongs.csv'

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

# 서울시 25개 자치구 표준 코드 (행정동코드 앞 5자리). gu_name은 이 표로 100% 결정되는
# 값이라 원본 데이터에 그 동이 등장하는지 여부와 무관하게 항상 이 매핑으로 채운다
# (models/ml/build_modeling_dataset.py에서 이미 쓰고 있는 것과 동일한 원칙 재사용).
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

# 행정동명 조회용 마스터 (소상공인 원본 CSV 6개 스냅샷 전체에서 조회)
# 스냅샷 1개만 쓰면 그 시점에 매장이 없던 동이 누락되므로, 전체 스냅샷을
# 합쳐 "어느 한 시점에라도 등장한 적 있는" 행정동코드-이름 매핑을 최대한 확보한다.
dong_name_frames = []
for snap in ORDER:
    frame = pd.read_csv(RAW_FILES[snap], usecols=['행정동코드', '행정동명'], dtype=str)
    dong_name_frames.append(frame)

dong_names = pd.concat(dong_name_frames, ignore_index=True)
dong_names = dong_names.drop_duplicates(subset=['행정동코드']).rename(
    columns={'행정동코드': 'dong_code', '행정동명': 'dong_name'})

print(f"동 이름 조회에 스냅샷 {len(ORDER)}개 사용, "
      f"고유 행정동코드 {dong_names['dong_code'].nunique()}개 확보")

# --- administrative_dongs.csv: 생활인구 dong_code ∪ 상권 6개 스냅샷 dong_code -------
# 두 원본이 서로 다른 동 목록을 가지고 있어서(생활인구 쪽에만 있는 동, 상권 쪽에만
# 있는 동이 각각 존재) population 쪽 dong_code만 기준으로 이름을 붙이면 상권에만
# 있는 동은 이름이 있는데도 그냥 버려진다. 합집합으로 마스터를 만들어서 이 문제를
# 소스 단계에서 아예 없앤다 — DB 적재(load_to_tidb.py) 쪽에서 더 이상 placeholder로
# 땜질할 필요 없이 이 파일 하나가 완전한 정답이 되도록.
all_dong_codes = sorted(set(pop['dong_code']) | set(dong_names['dong_code']))
admin = pd.DataFrame({'dong_code': all_dong_codes})
admin = admin.merge(dong_names, on='dong_code', how='left')
admin['gu_name'] = admin['dong_code'].str[:5].map(GU_CODE_MAP)

n_gu_unmapped = admin['gu_name'].isna().sum()
if n_gu_unmapped:
    # gu_name은 코드 체계상 항상 결정되는 값이라 정상 데이터라면 0건이어야 함.
    # 여기 걸리면 GU_CODE_MAP에 없는 낯선 접두사가 섞여 있다는 뜻이라 원본 확인이 필요함.
    print(f"⚠ GU_CODE_MAP에 없는 자치구 코드 접두사 {n_gu_unmapped}건 발견 — 데이터 확인 필요: "
          f"{admin.loc[admin['gu_name'].isna(), 'dong_code'].tolist()}")

# dong_name은 gu_name과 달리 "코드 -> 이름" 전국 공통 매핑표가 없다(동은 자치구마다
# 고유해서 25개짜리 GU_CODE_MAP 같은 걸 만들 수 없음). 생활인구+상권 6개 스냅샷을
# 다 합쳐도 안 나온다는 건 그 동에 지난 4년간 등록 매장이 한 번도 없었고 생활인구
# 조사 대상으로만 잡혔다는 뜻이라, 우리가 가진 데이터로는 더 이상 복구할 방법이 없음.
n_dong_name_missing = admin['dong_name'].isna().sum()
if n_dong_name_missing > 0:
    missing_codes = admin.loc[admin['dong_name'].isna(), 'dong_code'].tolist()
    print(f"dong_name 결측 {n_dong_name_missing}건 "
          f"(생활인구엔 있지만 상권 6개 스냅샷 전체에도 등장하지 않는 행정동): {missing_codes}")
    print("     -> 원본 데이터로는 이름을 알아낼 방법이 없어 '(미상)'으로 채움. "
          "필요하면 서울시 행정동코드 공식 목록으로 직접 확인해서 보완 권장.")
    admin['dong_name'] = admin['dong_name'].fillna('(미상)')

admin = admin[['dong_code', 'dong_name', 'gu_name']].sort_values('dong_code').reset_index(drop=True)
admin.to_csv(DONGS_DEST, index=False, encoding='utf-8-sig')
print(f"administrative_dongs.csv: {len(admin)}개 행정동 "
      f"(생활인구 {pop['dong_code'].nunique()}개 ∪ 상권 스냅샷 {dong_names['dong_code'].nunique()}개)")

# --- population_features.csv: 이름은 위에서 만든 완전한 admin 마스터에서 붙임 -------
# admin이 pop의 dong_code를 전부 포함하는 상위집합이라(합집합으로 만들었으므로)
# 이 merge는 항상 전부 매칭되고 dong_name/gu_name 결측이 남지 않는다.
pop = pop.merge(admin, on='dong_code', how='left')
assert pop['dong_name'].notna().all() and pop['gu_name'].notna().all(), \
    "admin 마스터가 pop의 상위집합이어야 하는데 매칭 안 된 행이 있음 — 로직 확인 필요"

cols = ['dong_code', 'dong_name', 'gu_name', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
        'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
pop = pop[cols].sort_values('foreign_short_ratio', ascending=False)
pop.to_csv(POP_DEST, index=False, encoding='utf-8-sig')
print(f"population_features.csv: {len(pop)}개 행정동, 관광특구 후보 임계값={threshold:.4f}")