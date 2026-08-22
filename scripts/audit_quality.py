import pandas as pd

ROOT = r"C:\sk-encoa\SKN35-2nd-3team\data\processed"
OUT = r"C:\sk-encoa\SKN35-2nd-3team\scripts\audit_quality_result.txt"

# 대한민국 대략적 경위도 범위
LAT_MIN, LAT_MAX = 33.0, 39.0
LON_MIN, LON_MAX = 124.0, 132.0

with open(OUT, "w", encoding="utf-8") as f:
    df = pd.read_csv(f"{ROOT}\\all_industries_features.csv", encoding="utf-8-sig")
    f.write("=== all_industries_features.csv ===\n")
    f.write(f"전체 {len(df)}행\n\n")
    f.write("결측치:\n")
    f.write(df.isna().sum()[df.isna().sum() > 0].to_string())
    f.write("\n\n")

    bad_coord = ((df["위도"] < LAT_MIN) | (df["위도"] > LAT_MAX) |
                 (df["경도"] < LON_MIN) | (df["경도"] > LON_MAX)) & df["위도"].notna()
    f.write(f"한국 범위 밖 좌표: {bad_coord.sum()}건\n")
    if bad_coord.sum() > 0:
        f.write(df.loc[bad_coord, ["업종명", "사업장명", "경도", "위도"]].head(10).to_string())
        f.write("\n")
    f.write("\n")

    df["개업일자_dt"] = pd.to_datetime(df["개업일자"], errors="coerce")
    old = df["개업일자_dt"].dt.year < 1950
    f.write(f"개업연도 1950년 이전: {old.sum()}건\n")
    if old.sum() > 0:
        f.write(df.loc[old, "개업연도"].value_counts().sort_index().to_string())
        f.write("\n")
    f.write("\n")

    f.write(f"tenure_years 상위 10개:\n{df['tenure_years'].nlargest(10).to_string()}\n\n")
    f.write(f"tenure_years < 0 (있으면 로직 오류): {(df['tenure_years'] < 0).sum()}건\n\n")

    f.write(f"업종밀집도 상위 10개:\n{df['업종밀집도'].nlargest(10).to_string()}\n\n")
    f.write(f"최근접버스정류장_거리m 상위 10개:\n{df['최근접버스정류장_거리m'].nlargest(10).to_string()}\n\n")
    f.write(f"최근접지하철역_거리m 상위 10개:\n{df['최근접지하철역_거리m'].nlargest(10).to_string()}\n\n")

    f.write("="*60 + "\n\n")

    df2 = pd.read_csv(f"{ROOT}\\retail_seoul_features.csv", encoding="utf-8-sig")
    f.write("=== retail_seoul_features.csv ===\n")
    f.write(f"전체 {len(df2)}행\n\n")
    f.write("결측치:\n")
    na2 = df2.isna().sum()
    f.write(na2[na2 > 0].to_string() if na2.sum() > 0 else "없음")
    f.write("\n\n")

    bad_coord2 = ((df2["위도"] < LAT_MIN) | (df2["위도"] > LAT_MAX) |
                  (df2["경도"] < LON_MIN) | (df2["경도"] > LON_MAX)) & df2["위도"].notna()
    f.write(f"한국 범위 밖 좌표: {bad_coord2.sum()}건\n\n")

    f.write(f"업종밀집도 상위 10개:\n{df2['업종밀집도'].nlargest(10).to_string()}\n\n")
    f.write(f"상가업소번호 중복 개수: {df2['상가업소번호'].duplicated().sum()}건\n")

print("done")
