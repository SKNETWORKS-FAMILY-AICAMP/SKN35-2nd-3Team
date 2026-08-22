import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import load_and_clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_TXT = Path(__file__).resolve().parent / "run_all_result.txt"

INDUSTRIES = {
    "일반음식점": "restaurant_nationwide.csv",
    "숙박업": "lodging_nationwide.csv",
    "미용업": "beauty_nationwide.csv",
    "세탁업": "laundry_nationwide.csv",
    "목욕장업": "bathhouse_nationwide.csv",
    "PC방": "pc_cafe_nationwide.csv",
}

results = []
with open(RESULT_TXT, "w", encoding="utf-8") as f:
    all_dfs = []
    for 업종명, fname in INDUSTRIES.items():
        path = RAW_DIR / fname
        df = load_and_clean(str(path), 업종명)
        out = PROCESSED_DIR / f"{path.stem.replace('_nationwide', '')}_clean.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        all_dfs.append(df)

        f.write(f"[{업종명}] {fname}\n")
        f.write(f"  정리 후 행 수: {len(df)}\n")
        f.write(f"  y 분포: {df['y'].value_counts().to_dict()}\n")
        f.write(f"  저장: {out.name}\n\n")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_out = PROCESSED_DIR / "all_industries_clean.csv"
    combined.to_csv(combined_out, index=False, encoding="utf-8-sig")

    f.write("=== 통합 결과 ===\n")
    f.write(f"전체 행 수: {len(combined)}\n")
    f.write("업종별 행 수:\n")
    f.write(combined["업종명"].value_counts().to_string())
    f.write("\n\n")
    f.write("전체 y 분포:\n")
    f.write(combined["y"].value_counts().to_string())
    f.write("\n\n업종별 y=1 비율:\n")
    f.write(combined.groupby("업종명")["y"].mean().to_string())
    f.write("\n")

print("done")
