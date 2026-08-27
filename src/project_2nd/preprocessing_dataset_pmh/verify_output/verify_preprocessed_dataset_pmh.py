"""
preprocessing_dataset_pmh/verify_output/verify_preprocessed_dataset_pmh.py

preprocess_modeling_dataset_pmh.py가 만든 data/processed/modeling_dataset_preprocessed_pmh.csv가
의도대로 잘 만들어졌는지 검증한다. 읽기 전용으로만 동작하며 data/raw, data/features,
data/processed의 어떤 파일도 수정하지 않는다.

다루는 내용:
  1. 기본 형태 (shape, 결측치) + 원본(modeling_dataset.csv) 대비 행 수 일치
  2. 인코딩(_enc) 컬럼 무결성: 원본 범주값 하나가 코드 하나로만 매핑되는지
  3. encoders_pmh.json에 저장된 매핑과 실제 _enc 값이 전부 일치하는지
  4. 타겟(is_closed_next) 비율이 전처리 전후로 동일한지 (행 필터링이 없었는지)
  5. nearest_same_industry_distance_m 결측 대체값(9999.0) 개수
  6. gu_name 복구 후 UNKNOWN 잔존 여부
  7. 수치형 컬럼의 inf/음수 등 이상값

입력: data/features/modeling_dataset.csv (읽기 전용, 원본 비교용)
      data/processed/modeling_dataset_preprocessed_pmh.csv (읽기 전용, 검증 대상)
      src/project_2nd/preprocessing_dataset_pmh/preprocess_output/encoders_pmh.json (읽기 전용)
출력: src/project_2nd/preprocessing_dataset_pmh/verify_output/verify_report_pmh.md
"""
import pandas as pd
import numpy as np
import json
import os

SRC_ORIGINAL = 'data/features/modeling_dataset.csv'
SRC_PREPROCESSED = 'data/processed/modeling_dataset_preprocessed_pmh.csv'
ENCODERS_PATH = 'src/project_2nd/preprocessing_dataset_pmh/preprocess_output/encoders_pmh.json'
OUTPUT_DIR = 'src/project_2nd/preprocessing_dataset_pmh/verify_output'
REPORT_PATH = f'{OUTPUT_DIR}/verify_report_pmh.md'

DIST_FILL_VALUE = 9999.0
NUMERIC_COLS = ['korean_pop', 'foreign_long_pop', 'foreign_short_pop', 'total_pop_avg',
                'foreign_short_ratio', 'same_industry_count_300m', 'total_count_300m',
                'nearest_same_industry_distance_m', 'dong_industry_count',
                'coord_cluster_size', 'store_age_months', 'keyword_growth_score']

os.makedirs(OUTPUT_DIR, exist_ok=True)
report = []
all_ok = True


def h(title, level=2):
    report.append(f"\n{'#' * level} {title}\n")


def p(text):
    report.append(text)


def check(name, ok, detail):
    global all_ok
    status = "PASS" if ok else "FAIL"
    line = f"- [{status}] {name}: {detail}"
    print(line)
    p(line)
    if not ok:
        all_ok = False


print("loading modeling_dataset.csv (원본) ...")
df_orig = pd.read_csv(SRC_ORIGINAL, dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                                            'industry_jung_code': str, 'industry_dae_code': str,
                                            'snapshot_date': str})
print("loading modeling_dataset_preprocessed_pmh.csv ...")
df = pd.read_csv(SRC_PREPROCESSED, dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                                           'industry_jung_code': str, 'industry_dae_code': str,
                                           'snapshot_date': str})
encoders = json.load(open(ENCODERS_PATH, encoding='utf-8'))

h("modeling_dataset_preprocessed_pmh.csv 검증 리포트", level=1)
p(f"- 원본(modeling_dataset.csv): {len(df_orig):,}행 x {df_orig.shape[1]}열")
p(f"- 전처리 결과(modeling_dataset_preprocessed_pmh.csv): {len(df):,}행 x {df.shape[1]}열")

# ---------------------------------------------------------------
# 1. 기본 형태
# ---------------------------------------------------------------
h("1. 기본 형태")

check("행 수 원본과 일치 (필터링 없이 결측치 대체만 수행됐는지)",
      len(df) == len(df_orig), f"원본 {len(df_orig):,}행 vs 전처리 {len(df):,}행")
check("결측치 0건", df.isna().sum().sum() == 0, f"{df.isna().sum().sum()}건")

# ---------------------------------------------------------------
# 2. 인코딩(_enc) 컬럼 무결성
# ---------------------------------------------------------------
h("2. 인코딩(_enc) 컬럼 무결성")

enc_cols = [c for c in df.columns if c.endswith('_enc')]
for col in enc_cols:
    base = col[:-4]
    n_map = df.groupby(base)[col].nunique()
    bad = int((n_map > 1).sum())
    check(f"{col}: 원본값 1개 -> 코드 1개로만 매핑",
          bad == 0, f"코드가 2개 이상으로 흩어진 원본값 {bad}건")

# ---------------------------------------------------------------
# 3. encoders_pmh.json과 실제 _enc 값 일치
# ---------------------------------------------------------------
h("3. encoders_pmh.json 매핑과 실제 _enc 값 일치")

for col in enc_cols:
    base = col[:-4]
    mapping = encoders.get(base, {})
    actual = df[[base, col]].drop_duplicates().set_index(base)[col].to_dict()
    mismatch = sum(1 for k, v in mapping.items() if actual.get(k) != v)
    check(f"{col}: encoders_pmh.json <-> 실제 값",
          mismatch == 0, f"불일치 {mismatch}건 (전체 카테고리 {len(mapping)}개 중)")

# ---------------------------------------------------------------
# 4. 타겟 비율 (전처리 전후 동일해야 함)
# ---------------------------------------------------------------
h("4. 타겟(is_closed_next) 비율 (전처리 전후 비교)")

rate_orig = df_orig['is_closed_next'].mean()
rate_proc = df['is_closed_next'].mean()
check("폐업 비율이 전처리 전후로 동일",
      abs(rate_orig - rate_proc) < 1e-9, f"원본 {rate_orig:.6f} vs 전처리 {rate_proc:.6f}")

# ---------------------------------------------------------------
# 5. nearest_same_industry_distance_m 결측 대체값 개수
# ---------------------------------------------------------------
h("5. nearest_same_industry_distance_m 결측 대체값(9999.0) 개수")

n_sentinel = int((df['nearest_same_industry_distance_m'] == DIST_FILL_VALUE).sum())
n_missing_orig = int(df_orig['nearest_same_industry_distance_m'].isna().sum())
check("대체값 개수가 원본 결측 건수와 일치",
      n_sentinel == n_missing_orig, f"원본 결측 {n_missing_orig:,}건 vs 대체값 {n_sentinel:,}건")

# ---------------------------------------------------------------
# 6. gu_name 복구 후 UNKNOWN 잔존 여부
# ---------------------------------------------------------------
h("6. gu_name 복구 결과")

n_unknown = int((df['gu_name'] == 'UNKNOWN').sum())
check("UNKNOWN으로 남은 gu_name 없음", n_unknown == 0, f"{n_unknown:,}건")

# ---------------------------------------------------------------
# 7. 수치형 컬럼 이상값 (inf, 음수)
# ---------------------------------------------------------------
h("7. 수치형 컬럼 이상값 (inf / 음수)")

for col in NUMERIC_COLS:
    n_inf = int(np.isinf(df[col]).sum())
    n_neg = int((df[col] < 0).sum())
    check(f"{col}: inf/음수 없음", n_inf == 0 and n_neg == 0, f"inf {n_inf}건, 음수 {n_neg}건")

# ---------------------------------------------------------------
# 결과
# ---------------------------------------------------------------
h("결과")
p(f"- 전체 판정: {'PASS' if all_ok else 'FAIL (위에서 FAIL 항목 확인)'}")

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print()
print(f"전체 판정: {'PASS' if all_ok else 'FAIL'}")
print(f"리포트: {REPORT_PATH}")
