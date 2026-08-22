import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retail_snapshot import estimate_closure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_A = ROOT / "data" / "raw" / "commercial_20250630" / "소상공인시장진흥공단_상가(상권)정보_서울_202506.csv"
SNAPSHOT_B = ROOT / "data" / "raw" / "commercial_20260630" / "소상공인시장진흥공단_상가(상권)정보_서울_202606.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_TXT = Path(__file__).resolve().parent / "run_retail_pilot_result.txt"

CATEGORIES = {
    "편의점": dict(category_col="상권업종소분류명", category_values=["편의점"]),
    "폰케이스판매점": dict(category_col="상권업종소분류명", category_values=["핸드폰 소매업"]),
    "무인매점": dict(name_keywords=["무인"]),
    "인형뽑기": dict(name_keywords=["뽑기", "클로우"]),
}

all_dfs = []
with open(RESULT_TXT, "w", encoding="utf-8") as f:
    for 업종명, kwargs in CATEGORIES.items():
        df = estimate_closure(str(SNAPSHOT_A), str(SNAPSHOT_B), 업종명, **kwargs)
        out = PROCESSED_DIR / f"retail_{업종명}_seoul_clean.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        all_dfs.append(df)

        f.write(f"[{업종명}] 시점A 필터링 결과: {len(df)}건\n")
        f.write(f"  y 분포: {df['y'].value_counts().to_dict()}\n")
        f.write(f"  폐업 추정 비율: {df['y'].mean():.2%}\n")
        f.write("  샘플 3건:\n")
        f.write(df.sample(min(3, len(df)), random_state=42).to_string())
        f.write("\n\n")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_out = PROCESSED_DIR / "retail_seoul_clean.csv"
    combined.to_csv(combined_out, index=False, encoding="utf-8-sig")

    f.write("=== 통합 결과 (서울, 4개 소매업 카테고리) ===\n")
    f.write(f"전체 행 수: {len(combined)}\n")
    f.write(combined["업종명"].value_counts().to_string())
    f.write("\n")

print("done")
