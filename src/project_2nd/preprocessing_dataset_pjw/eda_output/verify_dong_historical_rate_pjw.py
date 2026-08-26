"""dong_historical_rate 저장값 vs 재계산값 불일치(~30%) 원인 검증.

가설: build_modeling_dataset.py에서 dong_historical_rate는 스코프 필터링
(과학·기술/부동산/시설관리·임대 제외) 적용 "이전"에 계산되는데, 이 3개 업종군은
폐업률이 훨씬 낮아서(팀원 주석: 전체 평균의 절반 수준) 저장값이 낮게 나온다.
우리가 가진 modeling_dataset.csv는 이미 그 행들이 빠진 상태라 재계산하면 높게 나온다.

결론: 파이프라인 코드를 스코프 필터 적용 전까지 그대로 재현해서 계산하면 저장값과
소수점까지 일치 -> 데이터 누수/버그 아니라 계산 시점 차이(fold-safe하게 정상 동작).
"""

import hashlib

import pandas as pd

FEATURES_DIR = r"C:\sk-encoa\SKN35-2nd-3team\data\features"
TOP_DONG = "11680640"
EXCLUDED_GROUPS = ["과학·기술", "부동산", "시설관리·임대"]


def fold_of(store_id, k=5):
    h = hashlib.md5(store_id.encode()).hexdigest()
    return int(h, 16) % k


snapshots = pd.read_csv(
    f"{FEATURES_DIR}/store_snapshots.csv",
    dtype={"store_id": str, "dong_code": str, "industry_code": str, "snapshot_date": str},
)
snapshots = snapshots[snapshots["label_available"] == True].reset_index(drop=True)
industries = pd.read_csv(f"{FEATURES_DIR}/industries.csv", dtype=str)

df = snapshots.merge(industries[["industry_code", "custom_group"]], on="industry_code", how="left")
df = df.rename(columns={"custom_group": "industry_group"})
df["fold"] = df["store_id"].apply(fold_of)
df["is_closed_next"] = df["is_closed_next"].astype(str).map({"True": 1, "False": 0}).fillna(df["is_closed_next"]).astype(int)

excluded_mask = (df["dong_code"] == TOP_DONG) & (df["industry_group"].isin(EXCLUDED_GROUPS))
included_mask = (df["dong_code"] == TOP_DONG) & (~df["industry_group"].isin(EXCLUDED_GROUPS))
print(f"이 동에서 제외 대상 업종군 행 수: {excluded_mask.sum()}, 폐업률: {df.loc[excluded_mask, 'is_closed_next'].mean():.4f}")
print(f"이 동에서 최종 포함 업종군 폐업률: {df.loc[included_mask, 'is_closed_next'].mean():.4f}")

stored = pd.read_csv(
    f"{FEATURES_DIR}/modeling_dataset.csv", dtype={"dong_code": str}
)[["dong_code", "fold", "dong_historical_rate"]]

for k in sorted(df["fold"].unique()):
    train_mask = df["fold"] != k
    prefilter_rate = df.loc[train_mask].groupby("dong_code")["is_closed_next"].mean().get(TOP_DONG)
    stored_rate = stored.loc[(stored["dong_code"] == TOP_DONG) & (stored["fold"] == k), "dong_historical_rate"].mean()
    print(f"fold {k}: 저장값={stored_rate:.4f}  스코프필터 전 재현값={prefilter_rate:.4f}")
