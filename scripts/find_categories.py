import pandas as pd

path = r"C:\sk-encoa\SKN35-2nd-3team\data\raw\commercial_20260630\소상공인시장진흥공단_상가(상권)정보_서울_202606.csv"
out_path = r"C:\sk-encoa\SKN35-2nd-3team\scripts\find_categories_result.txt"

usecols = ["상권업종대분류명", "상권업종중분류명", "상권업종소분류명", "표준산업분류명"]
df = pd.read_csv(path, encoding="utf-8", usecols=usecols)

keywords = ["편의점", "무인", "휴대폰", "액세서리", "케이스", "인형", "뽑기", "크레인"]

with open(out_path, "w", encoding="utf-8") as f:
    for col in usecols:
        f.write(f"=== {col} 고유값 (전체 {df[col].nunique()}개) 중 키워드 매칭 ===\n")
        uniq = df[col].dropna().unique()
        for kw in keywords:
            matched = sorted([v for v in uniq if kw in str(v)])
            if matched:
                f.write(f"  [{kw}] {matched}\n")
        f.write("\n")

    f.write("=== 소분류명 전체 목록 ===\n")
    f.write(str(sorted(df["상권업종소분류명"].dropna().unique().tolist())))
    f.write("\n")

print("done")
