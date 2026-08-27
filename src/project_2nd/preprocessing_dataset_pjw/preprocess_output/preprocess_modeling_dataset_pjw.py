"""modeling_dataset.csv를 정제해서 modeling_dataset_refined_pjw.csv로 저장.

원본 파일은 건드리지 않는다 (팀원 파이프라인 코드/산출물 미변경).

v5 (2026-08-26 재실행): 팀원분(taemin1997) "결측 해소" 커밋으로 gu_name/생활인구
결측이 파이프라인 상류(build_modeling_dataset.py, build_population_features.py)에서
직접 해결됨 — gu_name은 우리와 동일한 dong_code 앞 5자리 매핑 방식으로 100% 채워지고,
생활인구는 BallTree 최근접 이웃으로 근사 대체(`population_is_proxied` 플래그 컬럼 추가).
이에 따라:
- 아래 2.5번(gu_name 복구) 코드는 이제 항상 결측 0건이라 사실상 no-op이지만, 안전망으로
  그대로 둠(향후 상류가 다시 바뀌어도 방어됨)
- 3번(생활인구 결측 유지) 관련 실험(exp_a)은 더 이상 유효하지 않음 — 이제 결측 자체가
  거의 없어짐(대신 근사치로 채워짐). `population_is_proxied`는 새 컬럼이라 그대로 통과시킴
- modeling_dataset.csv 재실행 시 Windows 콘솔(cp949)에서 팀원 코드의 em dash(—) 출력이
  깨지는 인코딩 이슈 있음 — `PYTHONIOENCODING=utf-8`로 우회(팀원 코드 자체는 수정 안 함)
재검증 결과: ROC-AUC 원본 0.747793, 정제본 0.748440 (+0.00065) — v4 대비 오히려 개선폭이
소폭 커짐(censor/keyword 플래그 효과가 더 깨끗한 데이터 위에서 더 잘 드러난 것으로 추정)

v4 (2026-08-26): 팀원분이 파이프라인에 floor_category(층정보, ROC-AUC 0.721→0.728로
검증된 유일한 효과 있는 피처)와 coord_cluster_size(DBSCAN 기반, 저희가 만들었던
exact-match 버전보다 정교함 — 스냅샷 간 좌표 미세 오차까지 하나의 건물/복합상가로
묶어줌)를 공식 반영해서 새 modeling_dataset.csv(33컬럼)를 내려받음.
저희 쪽 coord_cluster_size는 이제 완전히 대체됐으므로 제거하고, 팀원분 버전을 그대로 사용.
`run_pipeline.sh`로 로컬에서 직접 재생성한 파일 기준(SRC 경로가 로컬 data/features/로 변경됨).

v3까지의 이력 (구 modeling_dataset.csv, 31컬럼 기준):
- industry_code류 제거를 시도했다가 5-fold 검증에서 ROC-AUC가 전 fold 일관되게
  하락(-0.002)해서 되돌림 — industry_historical_rate와의 상관관계는 업종별 "평균"
  기준이라 높았지만(0.9999), 트리 모델이 industry_code를 다른 피처와 조합해 만드는
  세밀한 분기 정보까지는 대체하지 못했던 것으로 보임.
- dong_code 제거 — dong_historical_rate와 그룹평균 상관 0.87로 완전 중복은 아니었지만,
  5-fold ablation에서 있음/없음 성능이 사실상 동일해서 제거. (v4에서는 스코프/컬럼이
  바뀌었으니 재검증 필요 — compare_dongcode_ablation_pjw.py로 다시 확인할 것)
- coord_cluster_size 자체 구현 시 시점 누수(전체 스냅샷 합산) 발견 후 snapshot_date
  기준으로 수정한 이력 있음 — v4에서는 팀원분 버전(스냅샷별 DBSCAN)을 쓰므로 해당 없음.

v4 적용 내역:
0. transitioned_next 제거 — is_closed_next(타깃)와 같은 "다음 스냅샷" 시점 정보라
   타깃 누수 의심(둘은 상호배타적으로 나옴, 즉 transitioned_next=1이면 is_closed_next=0이
   자동 확정됨). preprocess_report_pjw.md 참고
1. total_pop_avg 제거 (korean_pop+foreign_long_pop+foreign_short_pop의 단순 합, 순수 중복)
2. is_mass_reclass_window 플래그 추가 (snapshot_date==202406, 소진공 대규모 재정비 구간 추정)
2.5. gu_name 결측 30,181행(12개 dong_code, population_features.csv와 매칭 실패 — 팀원분
   문서의 "9개 동 매칭 실패"와 같은 종류 이슈, 새 파이프라인에서 12개로 늘어남) 복구.
   dong_code 앞 5자리(구 코드)가 같은 다른 행의 gu_name으로 매핑해보니 12개 전부 모호함
   없이 gu_name 하나로 정확히 복원됨(강북구 6/강동구 2/동대문구 2/구로구 1/강남구 1).
   범주형 gu_name은 결측으로 두면 안 되니 여기서 채움
3. 생활인구(korean_pop 등) 결측은 채우지 않고 그대로 둠 — gu 평균 대체 vs 결측 유지를
   5-fold로 직접 비교(exp_a)해보니 통계적으로 동률(오히려 결측 유지 쪽이 근소 우세,
   ROC-AUC 0.747610 vs 0.747853, 표준편차 0.0007~0.0009 범위 안). LightGBM이 결측을
   자체적으로 잘 처리해서 굳이 대체할 필요가 없다고 판단, `population_imputed` 플래그도
   같이 제거(결측 자체가 이미 그 정보를 담고 있어 플래그가 중복)
4. is_left_censored_age 플래그 추가 — store_age_months는 첫 스냅샷(202312)을 기준으로
   계산되는데, 그 시점에 이미 존재하던 매장(전체의 78.85%)은 실제 개업일을 몰라 나이가
   과소추정됨(팀원분 한계점 문서에 명시된 이슈). data/features/stores.csv의
   first_seen_snapshot=='202312' 여부를 플래그로 노출. exp_b로 5-fold 검증 — 전부
   일관되게 ROC-AUC 개선(단독 +0.0003)
4.5. is_trend_keyword_match 플래그 추가 — 기존 keyword_growth_score는 growth_rate<=0인
   키워드가 매칭돼도 값이 0으로 남아 "매칭 안 됨"과 구분이 안 됨(실제로 4,839행이 이 케이스).
   data/features/store_snapshots.csv(store_name)와 trend_keywords.csv(keyword)를 조인해
   매칭 여부 자체를 이진 플래그로 분리. exp_b로 5-fold 검증 — 전부 일관되게 ROC-AUC 개선
   (단독 +0.00015, censor_flag와 함께 넣으면 +0.0004)
5. 파생 피처 3종 추가 — 전부 기존 컬럼 조합만 사용, 새 원본 데이터 불필요
   - industry_specialization_300m: same_industry_count_300m / total_count_300m (업종 특화도).
     total_count_300m==0이면 정의 불가라 NaN
   - competition_per_capita_300m: same_industry_count_300m / (korean_pop+foreign_long_pop+foreign_short_pop).
     인구 0이면 NaN
   - dong_industry_count_growth: (dong_code, industry_code) 그룹 내에서 snapshot_date 순으로
     dong_industry_count의 전기 대비 증감률. 그룹의 첫 스냅샷은 이전 값이 없어 NaN
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
FEATURES_DIR = REPO_ROOT / "data" / "features"
SRC = FEATURES_DIR / "modeling_dataset.csv"
OUT = REPO_ROOT / "data" / "processed" / "modeling_dataset_refined_pjw.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["total_pop_avg", "transitioned_next", "dong_code"]

df = pd.read_csv(SRC)

df["is_mass_reclass_window"] = (df["snapshot_date"] == 202406).astype(int)

# gu_name 결측 복구: dong_code 앞 5자리(구 코드)가 같은 다른 행의 gu_name으로 매핑
prefix5 = df["dong_code"].astype(str).str[:5]
prefix_to_gu = (
    df.loc[df["gu_name"].notna()]
    .assign(prefix5=lambda d: d["dong_code"].astype(str).str[:5])
    .drop_duplicates("prefix5")
    .set_index("prefix5")["gu_name"]
)
df["gu_name"] = df["gu_name"].fillna(prefix5.map(prefix_to_gu))

# 생활인구 결측(korean_pop 등)은 의도적으로 그대로 둔다 — exp_a_pop_impute.py 검증 결과
# gu 평균 대체와 통계적으로 동률이라 LightGBM의 기본 결측 처리에 맡김

# store_age_months 좌측절단(left-censoring) 플래그: 첫 스냅샷(202312)에 이미 존재하던
# 매장은 실제 개업일을 몰라 나이가 과소추정됨(팀원분 한계점 문서에 명시) — exp_b_age_kw.py로
# 검증(5-fold 전부 일관되게 ROC-AUC 개선, +censor_flag 단독 +0.0003)
stores = pd.read_csv(FEATURES_DIR / "stores.csv", dtype={"store_id": str, "first_seen_snapshot": str})
first_seen_map = dict(zip(stores["store_id"], stores["first_seen_snapshot"]))
df["is_left_censored_age"] = (df["store_id"].map(first_seen_map) == "202312").astype(int)

# 트렌드 키워드 이진 매칭 플래그: 기존 keyword_growth_score는 growth_rate<=0인 키워드가
# 매칭돼도 0으로 남아 "매칭 안 됨"과 구분이 안 됨 — exp_b_age_kw.py로 검증
# (5-fold 전부 일관되게 ROC-AUC 개선, +kw_flag 단독 +0.00015, censor_flag와 함께 넣으면 +0.0004)
_snapshots = pd.read_csv(
    FEATURES_DIR / "store_snapshots.csv",
    dtype={"store_id": str, "snapshot_date": str},
    usecols=["store_id", "snapshot_date", "store_name"],
)
_trend_kw = pd.read_csv(FEATURES_DIR / "trend_keywords.csv")
_keywords = _trend_kw["keyword"].dropna().unique().tolist()

_join_key = df[["store_id", "snapshot_date"]].copy()
_join_key["snapshot_date"] = _join_key["snapshot_date"].astype(str)
_merged = _join_key.merge(_snapshots, on=["store_id", "snapshot_date"], how="left")
_store_name = _merged["store_name"].fillna("")

_is_match = pd.Series(False, index=df.index)
for _kw in _keywords:
    _is_match |= _store_name.str.contains(_kw, na=False, regex=False)
df["is_trend_keyword_match"] = _is_match.astype(int)

# --- 파생 피처 3종 ---
df["industry_specialization_300m"] = (
    df["same_industry_count_300m"] / df["total_count_300m"].replace(0, float("nan"))
)

local_pop = df["korean_pop"] + df["foreign_long_pop"] + df["foreign_short_pop"]
df["competition_per_capita_300m"] = (
    df["same_industry_count_300m"] / local_pop.replace(0, float("nan"))
)

df = df.sort_values(["dong_code", "industry_code", "snapshot_date"])
grp = df.groupby(["dong_code", "industry_code"])["dong_industry_count"]
prev = grp.shift(1)
df["dong_industry_count_growth"] = (df["dong_industry_count"] - prev) / prev.replace(0, float("nan"))
df = df.sort_index()

df = df.drop(columns=DROP_COLS)

df.to_csv(OUT, index=False)

print(f"저장: {OUT}")
print(f"행수: {len(df)}, 컬럼수: {len(df.columns)}")
print(f"is_mass_reclass_window=1: {df['is_mass_reclass_window'].sum()}")
print(f"is_left_censored_age=1: {df['is_left_censored_age'].sum()}")
print(f"is_trend_keyword_match=1: {df['is_trend_keyword_match'].sum()}")
print(f"gu_name 남은 결측: {df['gu_name'].isna().sum()}")
for c in ["industry_specialization_300m", "competition_per_capita_300m", "dong_industry_count_growth"]:
    print(f"{c} 결측: {df[c].isna().sum()} ({df[c].isna().mean():.2%})")
print(f"전체 결측치:\n{df.isna().sum().pipe(lambda s: s[s > 0])}")
