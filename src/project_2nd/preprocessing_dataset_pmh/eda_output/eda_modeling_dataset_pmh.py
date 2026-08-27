"""
preprocessing_dataset_pmh/eda_output/eda_modeling_dataset_pmh.py

전처리를 시작하기 전, data/features/modeling_dataset.csv를 살펴보는 EDA 스크립트.
읽기 전용으로만 동작하며 data/raw, data/features의 어떤 파일도 수정하지 않는다.

다루는 내용:
  1. 기초 구조 (shape, dtype, 메모리, 결측치, 중복행)
  2. 타겟(is_closed_next) 분포 및 fold별 균형
  3. 수치형 변수 분포/분위수
  4. 범주형 변수 분포/카디널리티 (floor_category 포함)
  5. 타겟과의 이변량 관계 (수치형 그룹평균, 범주형별 폐업률, 시점별 추이)
  6. 수치형 변수 상관관계
  7. 결측치 패턴 (population_is_proxied 플래그, 최근접 동일업종 거리)

입력: data/features/modeling_dataset.csv (읽기 전용)
출력: src/project_2nd/preprocessing_dataset_pmh/eda_output/ 아래 png 그래프 + eda_report_pmh.md
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SRC = 'data/features/modeling_dataset.csv'
OUTPUT_DIR = 'src/project_2nd/preprocessing_dataset_pmh/eda_output'
REPORT_PATH = f'{OUTPUT_DIR}/eda_report_pmh.md'

TARGET = 'is_closed_next'

ID_COLS = ['snapshot_date', 'store_id', 'fold']
CATEGORICAL_COLS = ['industry_dae_code', 'industry_group', 'industry_jung_code',
                     'industry_jung_name', 'industry_code', 'industry_name',
                     'gu_name', 'dong_code', 'floor_category']
BINARY_COLS = ['previously_transitioned', 'tourist_zone_candidate', 'transitioned_next',
               'population_is_proxied']
NUMERIC_COLS = ['lng', 'lat', 'same_industry_count_300m', 'total_count_300m',
                'nearest_same_industry_distance_m', 'dong_industry_count',
                'coord_cluster_size', 'store_age_months', 'keyword_growth_score',
                'korean_pop', 'foreign_long_pop', 'foreign_short_pop', 'total_pop_avg',
                'foreign_short_ratio', 'industry_historical_rate', 'dong_historical_rate',
                'dong_industry_historical_rate']

os.makedirs(OUTPUT_DIR, exist_ok=True)
report = []


def h(title, level=2):
    report.append(f"\n{'#' * level} {title}\n")


def p(text):
    report.append(text)


def tbl(obj):
    """DataFrame/Series를 고정폭 텍스트 표로 렌더링(tabulate 의존성 없이)."""
    return f"```\n{obj.to_string()}\n```"


def save_fig(fig, name):
    path = f'{OUTPUT_DIR}/{name}.png'
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    p(f"![{name}]({name}.png)")


print("loading modeling_dataset.csv ...")
df = pd.read_csv(SRC, dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                              'industry_jung_code': str, 'industry_dae_code': str,
                              'snapshot_date': str})
print(f"loaded: {len(df):,} rows x {df.shape[1]} cols")

h("modeling_dataset.csv EDA 리포트 (pmh)", level=1)
p(f"- 행 수: {len(df):,}\n- 열 수: {df.shape[1]}\n- 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ---------------------------------------------------------------
# 1. 기초 구조
# ---------------------------------------------------------------
h("1. 기초 구조")

dtypes_tbl = df.dtypes.astype(str).rename('dtype').to_frame()
p(tbl(dtypes_tbl))

na_ratio = (df.isna().sum() / len(df) * 100).round(2)
na_tbl = na_ratio[na_ratio > 0].sort_values(ascending=False).rename('결측치 비율(%)').to_frame()
h("결측치가 있는 컬럼", level=3)
p(tbl(na_tbl) if len(na_tbl) else "결측치 없음")

dup_key = df.duplicated(subset=['store_id', 'snapshot_date']).sum()
dup_full = df.duplicated().sum()
p(f"\n- store_id+snapshot_date 중복: {dup_key:,}건\n- 완전 중복 행: {dup_full:,}건")

# ---------------------------------------------------------------
# 2. 타겟 분포
# ---------------------------------------------------------------
h("2. 타겟(is_closed_next) 분포")

target_ratio = (df[TARGET].value_counts(normalize=True).sort_index() * 100).rename('비율(%)').to_frame()
p(tbl(target_ratio))

fold_tbl = df.groupby('fold')[TARGET].agg(['count', 'mean']).rename(
    columns={'count': '표본수', 'mean': '폐업비율'})
h("fold별 표본수 / 폐업비율", level=3)
p(tbl(fold_tbl))

cross = pd.crosstab(df[TARGET], df['transitioned_next'])
h("is_closed_next x transitioned_next 교차표", level=3)
p(tbl(cross))

fig, ax = plt.subplots(figsize=(4, 4))
df[TARGET].value_counts().sort_index().plot(kind='bar', ax=ax, color=['#4C72B0', '#C44E52'])
ax.set_title('is_closed_next 분포')
ax.set_xlabel('is_closed_next')
save_fig(fig, 'target_distribution_pmh')

# ---------------------------------------------------------------
# 3. 수치형 변수 분포
# ---------------------------------------------------------------
h("3. 수치형 변수 분포")

desc = df[NUMERIC_COLS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
p(tbl(desc))

n_rows = -(-len(NUMERIC_COLS) // 4)
fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
for ax, col in zip(axes.flatten(), NUMERIC_COLS):
    vals = df[col].dropna()
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    ax.hist(vals.clip(lo, hi), bins=50, color='#4C72B0')
    ax.set_title(col, fontsize=9)
for ax in axes.flatten()[len(NUMERIC_COLS):]:
    ax.axis('off')
fig.tight_layout()
save_fig(fig, 'numeric_histograms_pmh')

# ---------------------------------------------------------------
# 4. 범주형 변수 분포
# ---------------------------------------------------------------
h("4. 범주형 변수 분포")

for col in ['industry_group', 'gu_name', 'floor_category']:
    vc = df[col].value_counts().rename('건수').to_frame()
    h(f"{col} (카디널리티={df[col].nunique()})", level=3)
    p(tbl(vc))

for col in ['industry_jung_name', 'industry_code', 'dong_code']:
    p(f"\n- {col} 카디널리티: {df[col].nunique()}")

fig, ax = plt.subplots(figsize=(8, 5))
df['industry_group'].value_counts().plot(kind='barh', ax=ax, color='#4C72B0')
ax.set_title('industry_group 빈도')
ax.invert_yaxis()
save_fig(fig, 'industry_group_counts_pmh')

fig, ax = plt.subplots(figsize=(6, 4))
df['floor_category'].value_counts().plot(kind='bar', ax=ax, color='#55A868')
ax.set_title('floor_category 빈도')
save_fig(fig, 'floor_category_counts_pmh')

# ---------------------------------------------------------------
# 5. 타겟과의 이변량 관계
# ---------------------------------------------------------------
h("5. 타겟과의 이변량 관계")

grp_mean = df.groupby(TARGET)[NUMERIC_COLS].mean().T.rename(columns={0: '정상(0)', 1: '폐업(1)'})
h("수치형 변수: 타겟별 평균", level=3)
p(tbl(grp_mean))

rate_by_group = df.groupby('industry_group')[TARGET].mean().sort_values(ascending=False).rename('폐업률').to_frame()
h("업종그룹(industry_group)별 폐업률", level=3)
p(tbl(rate_by_group))

rate_by_gu = df.groupby('gu_name')[TARGET].mean().sort_values(ascending=False).rename('폐업률').to_frame()
h("구(gu_name)별 폐업률", level=3)
p(tbl(rate_by_gu))

rate_by_floor = df.groupby('floor_category')[TARGET].mean().sort_values(ascending=False).rename('폐업률').to_frame()
h("층(floor_category)별 폐업률", level=3)
p(tbl(rate_by_floor))

rate_by_snap = df.groupby('snapshot_date')[TARGET].mean().sort_index()
h("시점(snapshot_date)별 폐업률", level=3)
p(tbl(rate_by_snap.rename('폐업률').to_frame()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
rate_by_group['폐업률'].plot(kind='barh', ax=axes[0], color='#C44E52')
axes[0].set_title('업종그룹별 폐업률')
axes[0].invert_yaxis()
rate_by_snap.plot(kind='bar', ax=axes[1], color='#55A868')
axes[1].set_title('시점별 폐업률')
fig.tight_layout()
save_fig(fig, 'closure_rate_by_group_and_time_pmh')

# ---------------------------------------------------------------
# 6. 수치형 변수 상관관계
# ---------------------------------------------------------------
h("6. 수치형 변수 상관관계")

corr = df[NUMERIC_COLS + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('수치형 변수 상관행렬')
fig.tight_layout()
save_fig(fig, 'correlation_heatmap_pmh')

target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).rename('corr').to_frame()
h("타겟과의 상관계수 (절대값 순)", level=3)
p(tbl(target_corr))

# ---------------------------------------------------------------
# 7. 결측치 패턴 / population_is_proxied 플래그
# ---------------------------------------------------------------
h("7. 결측치 패턴 / population_is_proxied 플래그")

overall_na = int(df.isna().sum().sum())
p(f"- 전체 결측치: {overall_na:,}건 (이전 버전에서 있던 gu_name/생활인구 결측이 파이프라인에서 대체값으로 채워짐)")

if 'population_is_proxied' in df.columns:
    proxied = df['population_is_proxied'].astype(bool)
    proxied_dongs = df.loc[proxied, 'dong_code'].unique()
    p(f"\n- population_is_proxied=True 행: {proxied.sum():,}건 ({proxied.mean() * 100:.2f}%)")
    p(f"- 대상 dong_code 수: {len(proxied_dongs)} -> {sorted(proxied_dongs.tolist())}")
    rate_by_proxy = df.groupby(proxied)[TARGET].mean().rename('폐업률').to_frame()
    rate_by_proxy.index = rate_by_proxy.index.map({False: 'proxied=False', True: 'proxied=True'})
    h("population_is_proxied별 폐업률", level=3)
    p(tbl(rate_by_proxy))

dist_missing = df['nearest_same_industry_distance_m'].isna()
p(f"\n- nearest_same_industry_distance_m 결측 행: {dist_missing.sum():,}건 ({dist_missing.mean() * 100:.2f}%)")
if dist_missing.sum() > 0:
    p("  (동일업종 매장이 해당 스냅샷에서 자기 자신뿐인 경우와 일치하는지 same_industry_count_300m<=1 비교)")
    match_check = (df.loc[dist_missing, 'same_industry_count_300m'] <= 1).mean()
    p(f"  -> 결측 행 중 same_industry_count_300m<=1 비율: {match_check * 100:.2f}%")

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"EDA 완료. 리포트: {REPORT_PATH}")
print(f"그래프: {OUTPUT_DIR}/*.png")
