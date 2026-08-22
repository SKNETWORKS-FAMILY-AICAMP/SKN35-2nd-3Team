"""인허가 대상이 아닌 소매업(편의점, 무인매점 등) 전처리 파이프라인.

정확한 폐업일자가 없는 업종이라, 두 시점의 소상공인시장진흥공단
"상가(상권)정보" 스냅샷을 비교해 시점 A에는 있었지만 시점 B에 없는
상가업소번호를 폐업으로 추정한다.

주의: 실제 폐업일자가 아닌 근사치. 상호명/주소 표기 변경 등으로 인한
오차 가능성을 포함한다 (PROJECT_BRIEF.md 3-1절 참고).
"""

from __future__ import annotations

import pandas as pd

USECOLS = [
    "상가업소번호", "상호명", "상권업종대분류명", "상권업종중분류명",
    "상권업종소분류명", "표준산업분류명", "지번주소", "도로명주소", "경도", "위도",
]


def _load_snapshot(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", usecols=USECOLS, dtype=str)


def _filter_category(
    df: pd.DataFrame,
    category_col: str | None,
    category_values: list[str] | None,
    name_keywords: list[str] | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if category_col and category_values:
        mask &= df[category_col].isin(category_values)
    if name_keywords:
        kw_mask = pd.Series(False, index=df.index)
        for kw in name_keywords:
            kw_mask |= df["상호명"].fillna("").str.contains(kw, regex=False)
        mask &= kw_mask
    return df[mask]


def estimate_closure(
    snapshot_a_path: str,
    snapshot_b_path: str,
    업종명: str,
    *,
    category_col: str | None = None,
    category_values: list[str] | None = None,
    name_keywords: list[str] | None = None,
) -> pd.DataFrame:
    """시점 A(과거) 대비 시점 B(최근) 스냅샷 비교로 폐업 추정 DataFrame 반환.

    반환 컬럼: 업종명, 사업장명, 소재지주소, 경도, 위도, 상가업소번호, y
    y=1: 시점 A에는 있었지만 시점 B에 없음(폐업 추정) / y=0: 시점 B에도 존재(생존 추정)
    """
    a = _load_snapshot(snapshot_a_path)
    b = _load_snapshot(snapshot_b_path)

    a_f = _filter_category(a, category_col, category_values, name_keywords)
    b_f = _filter_category(b, category_col, category_values, name_keywords)

    b_ids = set(b_f["상가업소번호"])

    out = a_f.copy()
    out["업종명"] = 업종명
    out["y"] = (~out["상가업소번호"].isin(b_ids)).astype(int)
    out = out.rename(columns={"상호명": "사업장명"})
    out["소재지주소"] = out["지번주소"].fillna(out["도로명주소"])
    out["경도"] = pd.to_numeric(out["경도"], errors="coerce")
    out["위도"] = pd.to_numeric(out["위도"], errors="coerce")

    cols = ["업종명", "사업장명", "소재지주소", "경도", "위도", "상가업소번호", "y"]
    return out[cols].reset_index(drop=True)
