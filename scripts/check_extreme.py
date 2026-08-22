import pandas as pd

df = pd.read_csv(
    r"C:\sk-encoa\SKN35-2nd-3team\data\processed\all_industries_features.csv",
    encoding="utf-8-sig",
)
out = r"C:\sk-encoa\SKN35-2nd-3team\scripts\check_extreme_result.txt"

with open(out, "w", encoding="utf-8") as f:
    f.write("=== 최근접버스정류장_거리m 최댓값 행 ===\n")
    row = df.loc[df["최근접버스정류장_거리m"].idxmax()]
    f.write(row[["업종명", "사업장명", "소재지주소", "경도", "위도", "최근접버스정류장_거리m"]].to_string())
    f.write("\n\n")

    f.write("=== 최근접지하철역_거리m 최댓값 행 ===\n")
    row2 = df.loc[df["최근접지하철역_거리m"].idxmax()]
    f.write(row2[["업종명", "사업장명", "소재지주소", "경도", "위도", "최근접지하철역_거리m"]].to_string())
    f.write("\n\n")

    f.write("=== 업종밀집도 최댓값 행 ===\n")
    row3 = df.loc[df["업종밀집도"].idxmax()]
    f.write(row3[["업종명", "사업장명", "소재지주소", "업종밀집도"]].to_string())
    f.write("\n")

print("done")
