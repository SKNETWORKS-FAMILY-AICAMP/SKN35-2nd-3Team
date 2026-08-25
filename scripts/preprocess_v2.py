"""modeling_dataset.csv를 정제해서 modeling_dataset_refined.csv로 저장.

원본 파일은 건드리지 않는다 (팀원 파이프라인 코드/산출물 미변경).

v2: industry_code류 제거를 시도했다가 5-fold 검증에서 ROC-AUC가 전 fold
일관되게 하락(-0.002)해서 되돌림 — industry_historical_rate와의 상관관계는
업종별 "평균" 기준이라 높았지만(0.9999), 트리 모델이 industry_code를 다른
피처와 조합해 만드는 세밀한 분기 정보까지는 대체하지 못했던 것으로 보임.

적용 내역:
1. total_pop_avg 제거 (korean_pop+foreign_long_pop+foreign_short_pop의 단순 합, 순수 중복)
2. is_mass_reclass_window 플래그 추가 (snapshot_date==202406, 소진공 대규모 재정비 구간 추정)
3. 생활인구 결측(1.6%)을 같은 gu_name 평균/최빈값으로 대체
4. population_imputed 플래그 추가
5. (v3) 파생 피처 3종 추가 — 전부 기존 컬럼 조합만 사용, 새 원본 데이터 불필요
   - industry_specialization_300m: same_industry_count_300m / total_count_300m (업종 특화도).
     total_count_300m==0이면 정의 불가라 NaN
   - competition_per_capita_300m: same_industry_count_300m / (korean_pop+foreign_long_pop+foreign_short_pop).
     인구 0이면 NaN
   - dong_industry_count_growth: (dong_code, industry_code) 그룹 내에서 snapshot_date 순으로
     dong_industry_count의 전기 대비 증감률. 그룹의 첫 스냅샷은 이전 값이 없어 NaN
"""

from pathlib import Path

import pandas as pd

SRC = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "modeling_dataset_refined.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["total_pop_avg"]
POP_COLS = ["korean_pop", "foreign_long_pop", "foreign_short_pop", "foreign_short_ratio"]

df = pd.read_csv(SRC)

df["is_mass_reclass_window"] = (df["snapshot_date"] == 202406).astype(int)

pop_missing = df["korean_pop"].isna()
df["population_imputed"] = pop_missing.astype(int)

gu_means = df.groupby("gu_name")[POP_COLS].transform("mean")
for col in POP_COLS:
    df[col] = df[col].fillna(gu_means[col])

gu_mode = df.groupby("gu_name")["tourist_zone_candidate"].transform(lambda s: s.mode().iloc[0] if s.notna().any() else 0)
df["tourist_zone_candidate"] = df["tourist_zone_candidate"].fillna(gu_mode)

# --- v3: 파생 피처 3종 ---
df["industry_specialization_300m"] = (
    df["same_industry_count_300m"] / df["total_count_300m"].replace(0, pd.NA)
)

local_pop = df["korean_pop"] + df["foreign_long_pop"] + df["foreign_short_pop"]
df["competition_per_capita_300m"] = (
    df["same_industry_count_300m"] / local_pop.replace(0, pd.NA)
)

df = df.sort_values(["dong_code", "industry_code", "snapshot_date"])
grp = df.groupby(["dong_code", "industry_code"])["dong_industry_count"]
prev = grp.shift(1)
df["dong_industry_count_growth"] = (df["dong_industry_count"] - prev) / prev.replace(0, pd.NA)
df = df.sort_index()

df = df.drop(columns=DROP_COLS)

df.to_csv(OUT, index=False)

print(f"저장: {OUT}")
print(f"행수: {len(df)}, 컬럼수: {len(df.columns)}")
print(f"population_imputed=1: {df['population_imputed'].sum()}")
print(f"is_mass_reclass_window=1: {df['is_mass_reclass_window'].sum()}")
for c in ["industry_specialization_300m", "competition_per_capita_300m", "dong_industry_count_growth"]:
    print(f"{c} 결측: {df[c].isna().sum()} ({df[c].isna().mean():.2%})")
print(f"기존 컬럼 중 남은 결측치:\n{df.drop(columns=['industry_specialization_300m', 'competition_per_capita_300m', 'dong_industry_count_growth']).isna().sum().pipe(lambda s: s[s > 0])}")
