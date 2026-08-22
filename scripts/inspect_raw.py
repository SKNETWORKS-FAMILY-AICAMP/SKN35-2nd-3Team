import pandas as pd

path = r"C:\sk-encoa\SKN35-2nd-3team\data\raw\seoul_general_restaurants.csv"
out_path = r"C:\sk-encoa\SKN35-2nd-3team\scripts\inspect_result.txt"

usecols = [
    "개방자치단체코드", "관리번호", "인허가일자", "영업상태명", "폐업일자",
    "사업장명", "업태구분명", "상세영업상태명", "상세영업상태코드", "영업상태코드",
    "도로명주소", "지번주소", "좌표정보(X)", "좌표정보(Y)",
]

df = pd.read_csv(path, encoding="cp949", usecols=usecols, dtype=str)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"전체 행 수: {len(df)}\n\n")

    f.write("영업상태명 value_counts:\n")
    f.write(df["영업상태명"].value_counts(dropna=False).to_string())
    f.write("\n\n")

    f.write("상세영업상태명 value_counts:\n")
    f.write(df["상세영업상태명"].value_counts(dropna=False).to_string())
    f.write("\n\n")

    f.write("인허가일자 결측 개수: " + str(df["인허가일자"].isna().sum()) + "\n")
    f.write("인허가일자 min/max: " + str(df["인허가일자"].min()) + " ~ " + str(df["인허가일자"].max()) + "\n\n")

    f.write("폐업일자 결측 개수: " + str(df["폐업일자"].isna().sum()) + "\n")
    f.write("폐업일자 min/max: " + str(df["폐업일자"].dropna().min()) + " ~ " + str(df["폐업일자"].dropna().max()) + "\n\n")

    f.write("좌표정보(X) 결측 개수: " + str(df["좌표정보(X)"].isna().sum()) + "\n")
    f.write("좌표정보(Y) 결측 개수: " + str(df["좌표정보(Y)"].isna().sum()) + "\n\n")

    f.write("전체 컬럼 결측치 개수:\n")
    f.write(df.isna().sum().to_string())
    f.write("\n")

print("done")
