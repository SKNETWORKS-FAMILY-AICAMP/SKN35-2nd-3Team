import pandas as pd
from pathlib import Path

RAW = Path(r"C:\sk-encoa\SKN35-2nd-3team\data\raw")
files = {
    "restaurant": "restaurant_nationwide.csv",
    "lodging": "lodging_nationwide.csv",
    "beauty": "beauty_nationwide.csv",
    "laundry": "laundry_nationwide.csv",
    "bathhouse": "bathhouse_nationwide.csv",
    "pc_cafe": "pc_cafe_nationwide.csv",
}

out_path = Path(r"C:\sk-encoa\SKN35-2nd-3team\scripts\check_columns_result.txt")
with open(out_path, "w", encoding="utf-8") as f:
    all_cols = {}
    for label, fname in files.items():
        path = RAW / fname
        df = pd.read_csv(path, encoding="cp949", nrows=2)
        all_cols[label] = list(df.columns)
        f.write(f"[{label}] {fname} ({len(df.columns)} cols)\n")
        f.write(str(list(df.columns)))
        f.write("\n\n")

    base = set(all_cols["restaurant"])
    f.write("=== restaurant 기준과의 차집합 ===\n")
    for label, cols in all_cols.items():
        diff_missing = base - set(cols)
        diff_extra = set(cols) - base
        f.write(f"{label}: 없음={diff_missing} / 추가={diff_extra}\n")

print("done")
