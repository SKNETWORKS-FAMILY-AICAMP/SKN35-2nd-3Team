import pandas as pd

path = r"C:\sk-encoa\SKN35-2nd-3team\data\raw\commercial_20260630\소상공인시장진흥공단_상가(상권)정보_서울_202606.csv"
out_path = r"C:\sk-encoa\SKN35-2nd-3team\scripts\inspect_commercial_result.txt"

df = pd.read_csv(path, encoding="utf-8", nrows=3)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"columns ({len(df.columns)}): {df.columns.tolist()}\n\n")
    f.write(df.head(3).to_string())
    f.write("\n")

print("done")
