"""
preprocessing_dataset_pmh/preprocess_output/preprocess_modeling_dataset_pmh.py

data/features/modeling_dataset.csv(EDA 완료본)를 모델 학습에 바로 넣을 수 있는
형태로 정리한다. data/raw, data/features의 원본 파일은 전혀 읽지도/수정하지도
않는다 — v2(2026-08-26)부터 gu_name/생활인구 결측이 파이프라인 쪽에서
population_is_proxied 플래그 + 대체값으로 이미 해소되어 data/raw 조회가
필요 없어졌다. 새 파일(modeling_dataset_preprocessed_pmh.csv)로만 결과를 저장한다.

v2 변경 사항 (이전 버전 대비):
  - 원본에 새 컬럼 population_is_proxied(bool)가 추가되고 gu_name/생활인구
    6개 컬럼의 결측(1.6%)이 전부 대체값으로 채워져 들어옴 -> data/raw에서
    dong_code -> gu_name을 조회해 복구하던 로직, gu_name 그룹 중앙값으로
    생활인구를 채우던 로직을 모두 제거. 대신 population_is_proxied를 다른
    bool 컬럼과 함께 0/1 정수로만 변환한다.
  - 이 변경으로 이 스크립트는 data/features/modeling_dataset.csv 한 파일만
    읽으며, data/raw는 아예 열지 않는다.

처리 내용:
  1. nearest_same_industry_distance_m 결측 방어적 처리
     - 이 값은 "해당 스냅샷에서 동일업종 매장이 자기 자신뿐"일 때만 결측이며,
       이번 데이터에는 결측이 0건이지만 향후 파이프라인 재실행 시 재발할 수 있어
       "근처에 동일업종이 없다"는 의미로 큰 상수(9999.0)를 채워 넣는다.
  2. bool 컬럼(transitioned_next, tourist_zone_candidate, population_is_proxied)을
     0/1 정수로 변환
  3. 카디널리티가 있는 범주형 9개(industry_dae_code/group/jung_code/jung_name/
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

입력: data/features/modeling_dataset.csv (읽기 전용, data/raw는 읽지 않음)
출력: data/processed/modeling_dataset_preprocessed_pmh.csv
      src/project_2nd/preprocessing_dataset_pmh/preprocess_output/encoders_pmh.json
      src/project_2nd/preprocessing_dataset_pmh/preprocess_output/preprocess_report_pmh.md
"""
import pandas as pd
import numpy as np
import json
import os

SRC = 'data/features/modeling_dataset.csv'
DEST = 'data/processed/modeling_dataset_preprocessed_pmh.csv'
OUTPUT_DIR = 'src/project_2nd/preprocessing_dataset_pmh/preprocess_output'
ENCODERS_PATH = f'{OUTPUT_DIR}/encoders_pmh.json'
REPORT_PATH = f'{OUTPUT_DIR}/preprocess_report_pmh.md'

BOOL_COLS = ['transitioned_next', 'tourist_zone_candidate', 'population_is_proxied']
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
p("- data/raw는 읽지 않음 (원본에 gu_name/생활인구 결측이 이미 없어 조회 불필요)")

# ---------------------------------------------------------------
# 1. 원본 결측치 현황
# ---------------------------------------------------------------
h("1. 원본 결측치 현황")

na_total = int(df.isna().sum().sum())
p(f"- 전체 결측치: {na_total:,}건")
if na_total > 0:
    na_cols = df.isna().sum()
    na_cols = na_cols[na_cols > 0]
    p(na_cols.to_string())

if 'population_is_proxied' in df.columns:
    proxied_ratio = df['population_is_proxied'].astype(bool).mean() * 100
    p(f"- population_is_proxied=True 비율: {proxied_ratio:.2f}% (생활인구가 대체값으로 채워진 행, 원본 파이프라인 단계에서 이미 처리됨)")

# ---------------------------------------------------------------
# 2. nearest_same_industry_distance_m 방어적 결측 처리
# ---------------------------------------------------------------
h("2. nearest_same_industry_distance_m 결측 처리")

dist_missing = df['nearest_same_industry_distance_m'].isna().sum()
df['nearest_same_industry_distance_m'] = df['nearest_same_industry_distance_m'].fillna(DIST_FILL_VALUE)
p(f"- 결측 {dist_missing:,}행을 {DIST_FILL_VALUE}(동일업종 없음을 의미하는 상수)로 대체")

# ---------------------------------------------------------------
# 3. bool 컬럼 -> 0/1 정수
# ---------------------------------------------------------------
h("3. bool 컬럼 정수 변환")

for col in BOOL_COLS:
    df[col] = df[col].map({'True': 1, 'False': 0, True: 1, False: 0}).astype(int)
p(f"- {', '.join(BOOL_COLS)} -> 0/1 정수로 변환")

# ---------------------------------------------------------------
# 4. 범주형 라벨 인코딩 (_enc 컬럼 추가, 매핑은 encoders_pmh.json에 저장)
# ---------------------------------------------------------------
h("4. 범주형 라벨 인코딩")

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
