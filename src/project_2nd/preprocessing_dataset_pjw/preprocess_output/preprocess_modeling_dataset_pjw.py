"""modeling_dataset.csv를 정제해서 modeling_dataset_refined_pjw.csv로 저장.

원본 파일은 건드리지 않는다 (팀원 파이프라인 코드/산출물 미변경).

v2: industry_code류 제거를 시도했다가 5-fold 검증에서 ROC-AUC가 전 fold
일관되게 하락(-0.002)해서 되돌림 — industry_historical_rate와의 상관관계는
업종별 "평균" 기준이라 높았지만(0.9999), 트리 모델이 industry_code를 다른
피처와 조합해 만드는 세밀한 분기 정보까지는 대체하지 못했던 것으로 보임.

적용 내역:
0. transitioned_next 제거 — is_closed_next(타깃)와 같은 "다음 스냅샷" 시점 정보라
   타깃 누수 의심(둘은 상호배타적으로 나옴, 즉 transitioned_next=1이면 is_closed_next=0이
   자동 확정됨). preprocess_report_pjw.md 참고
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
6. coord_cluster_size 컬럼 추가 — 서울 범위 밖 좌표는 없었지만, 정확히 같은 (lng, lat)을
   공유하는 서로 다른 store_id가 최대 883개까지 나옴(전체 행의 83.5%가 2개 이상과 좌표
   공유, 5.9%는 100개 이상과 공유). 대형 상가건물 때문일 수도 있고 지오코딩이 행정동
   중심좌표로 fallback된 것일 수도 있어 원인 확정은 못 하지만, 이런 좌표에서는
   same_industry_count_300m/nearest_same_industry_distance_m 같은 공간 피처의 신뢰도가
   떨어질 수 있음. 좌표 자체는 못 고치니 "이 좌표를 공유하는 유니크 store_id 수"를
   그대로 남겨서 후속 분석/모델링에서 참고하도록 함
7. dong_code 제거 — dong_historical_rate와 그룹평균 상관 0.87로 완전 중복은 아니었지만,
   5-fold ablation에서 있음/없음 성능이 사실상 동일(ROC-AUC 0.748361 vs 0.748395,
   F1은 오히려 without이 근소 우세)해서 제거. dong_historical_rate/
   dong_industry_historical_rate/gu_name/coord_cluster_size가 지역 정보를 충분히 대체함
"""

from pathlib import Path

import pandas as pd

SRC = r"C:\Users\playdata2\Desktop\플젝 공유\files-20260825T001524Z-1-001\files\modeling_dataset.csv"
REPO_ROOT = Path(__file__).resolve().parents[4]
OUT = REPO_ROOT / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["total_pop_avg", "transitioned_next", "dong_code"]
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

df["coord_cluster_size"] = df.groupby(["lng", "lat"])["store_id"].transform("nunique")

df = df.drop(columns=DROP_COLS)

df.to_csv(OUT, index=False)

print(f"저장: {OUT}")
print(f"행수: {len(df)}, 컬럼수: {len(df.columns)}")
print(f"population_imputed=1: {df['population_imputed'].sum()}")
print(f"is_mass_reclass_window=1: {df['is_mass_reclass_window'].sum()}")
for c in ["industry_specialization_300m", "competition_per_capita_300m", "dong_industry_count_growth"]:
    print(f"{c} 결측: {df[c].isna().sum()} ({df[c].isna().mean():.2%})")
print(f"coord_cluster_size >= 20인 행: {(df['coord_cluster_size'] >= 20).sum()} ({(df['coord_cluster_size'] >= 20).mean():.2%})")
print(f"기존 컬럼 중 남은 결측치:\n{df.drop(columns=['industry_specialization_300m', 'competition_per_capita_300m', 'dong_industry_count_growth']).isna().sum().pipe(lambda s: s[s > 0])}")
