"""modeling_dataset.csv EDA — 결측치/타깃분포/상관관계/좌표클러스터/fold-safety 검증을
한 번에 실행. 원본 파일(팀원 공유분)은 읽기만 하고 수정하지 않는다.
결과는 eda_report_pjw.md에 정리된 내용의 재현용 스크립트.
"""

import pandas as pd

SRC = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"

pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 200)

df = pd.read_csv(SRC)

print(f"전체 행수: {len(df)}, 컬럼수: {len(df.columns)}")
print(f"문서(modeling_설명.md) 기재 행수: 2,591,877 — 실제와 {2591877 - len(df)}행 차이, 버전 확인 필요\n")

print("=== 중복: (store_id, snapshot_date) ===")
print(f"중복 행수: {df.duplicated(subset=['store_id', 'snapshot_date']).sum()}\n")

print("=== 결측치 ===")
na = df.isna().sum()
print(na[na > 0])
print()

print("=== 타깃 분포 ===")
print(df["is_closed_next"].value_counts(normalize=True))
print()

print("=== fold별 타깃 비율 ===")
print(df.groupby("fold")["is_closed_next"].agg(["count", "mean"]))
print()

numeric_cols = [
    "lng", "lat", "same_industry_count_300m", "total_count_300m",
    "nearest_same_industry_distance_m", "dong_industry_count", "store_age_months",
    "keyword_growth_score", "korean_pop", "foreign_long_pop", "foreign_short_pop",
    "total_pop_avg", "foreign_short_ratio", "industry_historical_rate",
    "dong_historical_rate", "dong_industry_historical_rate",
]
print("=== 숫자형 피처 vs 타깃 상관계수 ===")
corr = df[numeric_cols + ["is_closed_next"]].corr()["is_closed_next"].drop("is_closed_next")
print(corr.sort_values(key=abs, ascending=False))
print()

print("=== 좌표 클러스터 (동일 lng/lat 공유 유니크 store_id 수) ===")
cluster = df.groupby(["lng", "lat"])["store_id"].transform("nunique")
for th in [2, 5, 20, 100]:
    print(f"  cluster_size>={th}: {(cluster >= th).sum()}행 ({(cluster >= th).mean():.2%})")
print()

print("=== fold-safe 검증: industry_historical_rate (표본 많은 업종 1개) ===")
top_ic = df["industry_code"].value_counts().index[0]
for fold in sorted(df["fold"].unique()):
    stored = df.loc[(df["industry_code"] == top_ic) & (df["fold"] == fold), "industry_historical_rate"].iloc[0]
    recomputed = df.loc[(df["industry_code"] == top_ic) & (df["fold"] != fold), "is_closed_next"].mean()
    print(f"  fold={fold}: 저장값={stored:.5f} fold제외재계산={recomputed:.5f} 일치={abs(stored - recomputed) < 1e-6}")
print()

print("=== fold-safe 검증: dong_historical_rate (표본 많은 동 1개) — 불일치 재현 ===")
top_dong = df["dong_code"].value_counts().index[0]
for fold in sorted(df["fold"].unique()):
    stored = df.loc[(df["dong_code"] == top_dong) & (df["fold"] == fold), "dong_historical_rate"].iloc[0]
    recomputed = df.loc[(df["dong_code"] == top_dong) & (df["fold"] != fold), "is_closed_next"].mean()
    print(f"  fold={fold}: 저장값={stored:.5f} fold제외재계산={recomputed:.5f} 일치={abs(stored - recomputed) < 1e-6}")
