"""
preprocessing_dataset_pmh/preprocess_output/preprocess_modeling_dataset_pmh.py

data/features/modeling_dataset.csv(EDA 완료본)를 모델 학습에 바로 넣을 수 있는
형태로 정리한다. data/raw, data/features의 원본 파일은 전혀 수정하지 않고,
새 파일(modeling_dataset_preprocessed_pmh.csv)로만 결과를 저장한다.

처리 내용:
  1. gu_name 결측 복구
     - build_population_features.py가 seoul_202606.csv 한 스냅샷에서만
       dong_code -> gu_name 매핑을 만들어서 놓친 경우다. 실제로는 결측 dong_code
       전부 다른 5개 스냅샷 중 한 곳에는 존재해서(행정구역 개편/오탈자로 최신
       스냅샷에서만 빠짐), data/raw의 6개 스냅샷 전체에서 매핑을 만들면
       읽기 전용 조회만으로 전부 복구된다.
  2. 생활인구 6개 컬럼 결측 대체
     - 이 결측은 population_features.csv 자체에 없는 행정동이라 원본에서 복구 불가.
       gu_name 복구 후, 같은 gu_name 안의 중앙값으로 대체한다(전체 중앙값보다
       지역 편차를 덜 왜곡함). tourist_zone_candidate는 이진 플래그라 최빈값(False)으로 채운다.
  3. nearest_same_industry_distance_m 결측 방어적 처리
     - 이 값은 "해당 스냅샷에서 동일업종 매장이 자기 자신뿐"일 때만 결측이며,
       이번 데이터에는 결측이 0건이지만 향후 파이프라인 재실행 시 재발할 수 있어
       "근처에 동일업종이 없다"는 의미로 큰 상수(9999.0)를 채워 넣는다.
  4. bool 컬럼(transitioned_next, tourist_zone_candidate)을 0/1 정수로 변환
  5. 카디널리티가 있는 범주형 9개(industry_dae_code/group/jung_code/jung_name/
     code/name, gu_name, dong_code, floor_category)를 정수 라벨로 인코딩해 `_enc`
     컬럼으로 추가한다(원본 문자열 컬럼은 그대로 유지). 매핑은 encoders_pmh.json에
     저장해서 앱 서빙 단계에서도 동일한 인코딩을 재사용할 수 있게 한다.

의도적으로 하지 않은 것:
  - 수치형 스케일링(StandardScaler 등): 트리 기반 모델(LightGBM/XGBoost/CatBoost)은
    스케일링이 불필요하고, MLP처럼 스케일링이 필요한 모델은 fold별로 train 데이터에만
    fit해야 누수가 없다. 그래서 스케일링은 이 공용 전처리 단계가 아니라 각 모델의
    학습 스크립트(models/ml, models/dl) 안에서 fold-safe하게 처리하는 게 맞다.
  - industry_historical_rate 등 fold-safe 타겟 인코딩 컬럼은 build_modeling_dataset.py에서
    이미 fold별로 누수 없이 계산되어 있어 그대로 둔다.

입력: data/features/modeling_dataset.csv, data/raw/seoul_*.csv (gu_name 조회용, 읽기 전용)
출력: data/processed/modeling_dataset_preprocessed_pmh.csv
      src/project_2nd/preprocessing_dataset_pmh/preprocess_output/encoders_pmh.json
      src/project_2nd/preprocessing_dataset_pmh/preprocess_output/preprocess_report_pmh.md
"""
import pandas as pd
import numpy as np
import json
import os

SRC = 'data/features/modeling_dataset.csv'
RAW_DIR = 'data/raw'
DEST = 'data/processed/modeling_dataset_preprocessed_pmh.csv'
OUTPUT_DIR = 'src/project_2nd/preprocessing_dataset_pmh/preprocess_output'
ENCODERS_PATH = f'{OUTPUT_DIR}/encoders_pmh.json'
REPORT_PATH = f'{OUTPUT_DIR}/preprocess_report_pmh.md'

RAW_SNAPSHOTS = ['seoul_202312', 'seoul_202406', 'seoul_202412',
                  'seoul_202506', 'seoul_202512', 'seoul_202606']

POP_NUMERIC_COLS = ['korean_pop', 'foreign_long_pop', 'foreign_short_pop',
                     'total_pop_avg', 'foreign_short_ratio']
CATEGORICAL_COLS = ['industry_dae_code', 'industry_group', 'industry_jung_code',
                     'industry_jung_name', 'industry_code', 'industry_name',
                     'gu_name', 'dong_code', 'floor_category']
DIST_FILL_VALUE = 9999.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
report = []


def h(title, level=2):
    report.append(f"\n{'#' * level} {title}\n")


def p(text):
    report.append(text)


print("loading modeling_dataset.csv ...")
df = pd.read_csv(SRC, dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                              'industry_jung_code': str, 'industry_dae_code': str,
                              'snapshot_date': str})
n_before = len(df)
n_before_cols = df.shape[1]
print(f"loaded: {n_before:,} rows x {n_before_cols} cols")

h("modeling_dataset.csv 전처리 리포트 (pmh)", level=1)
p(f"- 입력 행 수: {n_before:,}")

# ---------------------------------------------------------------
# 1. gu_name 결측 복구 (data/raw 6개 스냅샷 전체에서 dong_code -> gu_name 매핑)
# ---------------------------------------------------------------
h("1. gu_name 결측 복구")

gu_before = df['gu_name'].isna().sum()
dong_gu_maps = []
for snap in RAW_SNAPSHOTS:
    raw = pd.read_csv(f'{RAW_DIR}/{snap}.csv', usecols=['행정동코드', '시군구명'], dtype=str)
    dong_gu_maps.append(raw.drop_duplicates(subset=['행정동코드']))
dong_to_gu = (pd.concat(dong_gu_maps)
              .drop_duplicates(subset=['행정동코드'])
              .set_index('행정동코드')['시군구명'])

missing_mask = df['gu_name'].isna()
df.loc[missing_mask, 'gu_name'] = df.loc[missing_mask, 'dong_code'].map(dong_to_gu)
gu_after = df['gu_name'].isna().sum()

p(f"- 복구 전 gu_name 결측: {gu_before:,}행")
p(f"- 복구 후 gu_name 결측: {gu_after:,}행 (data/raw 6개 스냅샷 전체 조회로 복구, data/raw는 읽기만 함)")
if gu_after > 0:
    still_missing_dongs = sorted(df.loc[df['gu_name'].isna(), 'dong_code'].unique().tolist())
    df['gu_name'] = df['gu_name'].fillna('UNKNOWN')
    p(f"- data/raw에서도 못 찾은 dong_code {len(still_missing_dongs)}개는 'UNKNOWN'으로 채움: {still_missing_dongs}")

# ---------------------------------------------------------------
# 2. 생활인구 6개 컬럼 결측 대체 (gu_name 그룹 중앙값)
# ---------------------------------------------------------------
h("2. 생활인구 결측 대체")

pop_missing_before = df[POP_NUMERIC_COLS].isna().any(axis=1).sum()
gu_median = df.groupby('gu_name')[POP_NUMERIC_COLS].transform('median')
for col in POP_NUMERIC_COLS:
    df[col] = df[col].fillna(gu_median[col])
    df[col] = df[col].fillna(df[col].median())  # 그룹 내 전체 결측인 극단적 경우 대비

df['tourist_zone_candidate'] = df['tourist_zone_candidate'].map({'True': True, 'False': False, True: True, False: False})
tourist_missing = df['tourist_zone_candidate'].isna().sum()
df['tourist_zone_candidate'] = df['tourist_zone_candidate'].fillna(False)

p(f"- 생활인구 5개 수치 컬럼 결측(대체 전): {pop_missing_before:,}행 -> gu_name 그룹 중앙값으로 대체")
p(f"- tourist_zone_candidate 결측(대체 전): {tourist_missing:,}행 -> False(최빈값)로 대체")

# ---------------------------------------------------------------
# 3. nearest_same_industry_distance_m 방어적 결측 처리
# ---------------------------------------------------------------
h("3. nearest_same_industry_distance_m 결측 처리")

dist_missing = df['nearest_same_industry_distance_m'].isna().sum()
df['nearest_same_industry_distance_m'] = df['nearest_same_industry_distance_m'].fillna(DIST_FILL_VALUE)
p(f"- 결측 {dist_missing:,}행을 {DIST_FILL_VALUE}(동일업종 없음을 의미하는 상수)로 대체")

# ---------------------------------------------------------------
# 4. bool 컬럼 -> 0/1 정수
# ---------------------------------------------------------------
h("4. bool 컬럼 정수 변환")

df['transitioned_next'] = df['transitioned_next'].map({'True': 1, 'False': 0, True: 1, False: 0}).astype(int)
df['tourist_zone_candidate'] = df['tourist_zone_candidate'].astype(int)
p("- transitioned_next, tourist_zone_candidate -> 0/1 정수로 변환")

# ---------------------------------------------------------------
# 5. 범주형 라벨 인코딩 (_enc 컬럼 추가, 매핑은 encoders_pmh.json에 저장)
# ---------------------------------------------------------------
h("5. 범주형 라벨 인코딩")

encoders = {}
for col in CATEGORICAL_COLS:
    codes, uniques = pd.factorize(df[col], sort=True)
    df[f'{col}_enc'] = codes
    encoders[col] = {str(v): i for i, v in enumerate(uniques)}
    p(f"- {col} -> {col}_enc (카디널리티={len(uniques)})")

with open(ENCODERS_PATH, 'w', encoding='utf-8') as f:
    json.dump(encoders, f, ensure_ascii=False, indent=2)
p(f"\n인코딩 매핑 저장: {ENCODERS_PATH} (서빙 시 동일 매핑 재사용 가능, 사전에 없는 값은 -1)")

# ---------------------------------------------------------------
# 저장
# ---------------------------------------------------------------
assert len(df) == n_before, "행 수가 변하면 안 된다 (필터링 없이 결측치 대체만 수행)"
assert df.isna().sum().sum() == 0, "결측치가 남아있으면 안 된다"

df.to_csv(DEST, index=False, encoding='utf-8-sig')

h("결과")
p(f"- 출력 행 수: {len(df):,} (입력과 동일, 행 제거 없음)")
p(f"- 출력 컬럼 수: {df.shape[1]} (원본 {n_before_cols} + 인코딩 {len(CATEGORICAL_COLS)})")
p(f"- 최종 결측치: {df.isna().sum().sum()}건")
p(f"- 저장 위치: {DEST}")

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"전처리 완료: {len(df):,} rows x {df.shape[1]} cols")
print(f"출력: {DEST}")
print(f"리포트: {REPORT_PATH}")
print(f"인코더: {ENCODERS_PATH}")
