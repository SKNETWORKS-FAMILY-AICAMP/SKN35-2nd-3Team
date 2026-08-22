import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import load_and_clean  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "seoul_general_restaurants.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT = PROCESSED_DIR / "general_restaurants_seoul_clean.csv"
RESULT_TXT = Path(__file__).resolve().parent / "pilot_result.txt"

df = load_and_clean(str(RAW), "일반음식점")
df.to_csv(OUT, index=False, encoding="utf-8-sig")

with open(RESULT_TXT, "w", encoding="utf-8") as f:
    f.write(f"정리 후 행 수: {len(df)}\n\n")
    f.write("y(타깃) 분포:\n")
    f.write(df["y"].value_counts(dropna=False).to_string())
    f.write("\n\n")
    f.write("영업상태 분포:\n")
    f.write(df["영업상태"].value_counts(dropna=False).to_string())
    f.write("\n\n")
    f.write("결측치:\n")
    f.write(df.isna().sum().to_string())
    f.write("\n\n")
    f.write("tenure_years 기술통계:\n")
    f.write(df["tenure_years"].describe().to_string())
    f.write("\n\n")
    f.write("개업연도 범위: " + str(df["개업연도"].min()) + " ~ " + str(df["개업연도"].max()) + "\n\n")
    f.write("샘플 5행:\n")
    f.write(df.sample(5, random_state=42).to_string())
    f.write("\n")

print("done")
