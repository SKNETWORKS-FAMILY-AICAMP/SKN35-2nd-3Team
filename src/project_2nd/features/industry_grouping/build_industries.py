"""
features/industry_grouping/build_industries.py

업종 마스터 테이블을 만든다. 실제 데이터 검증 결과 6개 스냅샷 전부
동일한 업종분류 체계(소분류 247개)를 사용하므로, 한 스냅샷에서만 추출하면 된다.
대분류명(10개: 음식/소매/과학·기술/수리·개인/교육/부동산/시설관리·임대/예술·스포츠/보건의료/숙박)을
그대로 custom_group으로 채택한다.

입력: data/raw/seoul_202606.csv (아무 스냅샷이나 무방, 최신 것 사용)
출력: data/features/industries.csv
"""
import pandas as pd

SRC = 'data/raw/seoul_202606.csv'
DEST = 'data/features/industries.csv'

df = pd.read_csv(SRC, usecols=['상권업종대분류코드', '상권업종대분류명',
                                '상권업종중분류코드', '상권업종중분류명',
                                '상권업종소분류코드', '상권업종소분류명'], dtype=str)

industries = df.drop_duplicates(subset=['상권업종소분류코드']).sort_values('상권업종소분류코드')
industries = industries.rename(columns={
    '상권업종대분류코드': 'industry_dae_code',
    '상권업종대분류명': 'custom_group',
    '상권업종중분류코드': 'industry_jung_code',
    '상권업종중분류명': 'industry_jung_name',
    '상권업종소분류코드': 'industry_code',
    '상권업종소분류명': 'industry_name',
})

industries.to_csv(DEST, index=False, encoding='utf-8-sig')
print(f"industries.csv: {len(industries)}행, custom_group {industries['custom_group'].nunique()}개")
