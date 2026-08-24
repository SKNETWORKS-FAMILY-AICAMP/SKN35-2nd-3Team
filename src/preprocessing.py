"""공공데이터(인허가) 표준 스키마 -> 공통 스키마 전처리 파이프라인.

지원 원본 스키마: 행정안전부 인허가데이터개방시스템 "공공데이터 제공표준" CSV
(일반음식점, 숙박업 등 6대 업종 공통 컬럼셋을 사용).
"""

from __future__ import annotations

import pandas as pd
from pyproj import Transformer

# 원본(EPSG:5174, Bessel 중부원점 TM) -> WGS84 위경도
_TM2LATLON = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)

TARGET_CUTOFF = pd.Timestamp("2024-01-01")

# 대한민국 대략적 경위도 범위 — 벗어나면 지오코딩 오류로 간주
KOREA_LAT_RANGE = (33.0, 39.0)
KOREA_LON_RANGE = (124.0, 132.0)

# 이 연도 이전 개업일자는 실제 개업일이 아니라 "날짜 모름"을 나타내는
# placeholder일 가능성이 높음(예: 1900-05-31로 동일하게 찍힌 사례들) — tenure 계산에서 제외
PLACEHOLDER_YEAR_THRESHOLD = 1950

# 영업상태명이 이 값이면서 폐업일자가 없으면 사실상 폐업으로 보되, 정확한 날짜를
# 몰라 2024년 기준 판단이 불가능해 제외한다(PC방의 인허가 취소/말소/만료/정지/중지
# 등). '휴업'은 리모델링 등 일시 중단일 수 있어 제외 대상에서 뺀다.
CLOSURE_LIKE_STATUSES = {"폐업", "취소/말소/만료/정지/중지", "제외/삭제/전출"}

RAW_COLUMNS = {
    "개업일자": "인허가일자",
    "폐업일자": "폐업일자",
    "영업상태명": "영업상태명",
    "사업장명": "사업장명",
    "지번주소": "지번주소",
    "도로명주소": "도로명주소",
    "좌표X": "좌표정보(X)",
    "좌표Y": "좌표정보(Y)",
}


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


def _transform_coords(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    mask = x.notna() & y.notna()
    lon = pd.Series(float("nan"), index=x.index, dtype="float64")
    lat = pd.Series(float("nan"), index=x.index, dtype="float64")
    if mask.any():
        lons, lats = _TM2LATLON.transform(x[mask].to_numpy(), y[mask].to_numpy())
        lon.loc[mask] = lons
        lat.loc[mask] = lats
    return lon, lat


def load_and_clean(file_path: str, 업종명: str, *, encoding: str = "cp949") -> pd.DataFrame:
    """원본 인허가 CSV를 읽어 공통 표준 스키마로 정리한 DataFrame을 반환한다.

    반환 컬럼: 업종명, 사업장명, 개업일자, 폐업일자, 영업상태, 소재지주소,
    경도, 위도, tenure_years, 개업연도, y

    2024년 이전에 폐업한 행, 그리고 폐업(류) 상태인데 폐업일자를 몰라 판단이
    불가능한 행은 결과에서 제외된다(y=0은 "현재 영업 중"만을 의미).
    """
    usecols = list(RAW_COLUMNS.values())
    df = pd.read_csv(file_path, encoding=encoding, usecols=usecols, dtype=str)

    out = pd.DataFrame(index=df.index)
    out["업종명"] = 업종명
    out["사업장명"] = df[RAW_COLUMNS["사업장명"]]
    out["개업일자"] = _parse_date(df[RAW_COLUMNS["개업일자"]])
    out["폐업일자"] = _parse_date(df[RAW_COLUMNS["폐업일자"]])
    out["영업상태"] = df[RAW_COLUMNS["영업상태명"]]
    # 지번주소를 우선 사용(결측 적음), 없으면 도로명주소로 보완
    out["소재지주소"] = df[RAW_COLUMNS["지번주소"]].fillna(df[RAW_COLUMNS["도로명주소"]])

    x = pd.to_numeric(df[RAW_COLUMNS["좌표X"]], errors="coerce")
    y = pd.to_numeric(df[RAW_COLUMNS["좌표Y"]], errors="coerce")
    out["좌표X"] = x
    out["좌표Y"] = y
    out["경도"], out["위도"] = _transform_coords(x, y)

    # 대한민국 범위를 벗어난 좌표는 지오코딩 오류로 보고 결측 처리
    out_of_bounds = out["위도"].notna() & (
        ~out["위도"].between(*KOREA_LAT_RANGE) | ~out["경도"].between(*KOREA_LON_RANGE)
    )
    out.loc[out_of_bounds, ["좌표X", "좌표Y", "경도", "위도"]] = float("nan")

    # 영업상태가 폐업(류)인데 폐업일자가 없는 행은 타깃을 정할 수 없어 제외
    ambiguous = out["영업상태"].isin(CLOSURE_LIKE_STATUSES) & out["폐업일자"].isna()
    out = out.loc[~ambiguous].copy()

    # 논리 오류(개업일자 > 폐업일자) 행 제외
    logical_error = out["폐업일자"].notna() & (out["개업일자"] > out["폐업일자"])
    out = out.loc[~logical_error].copy()

    today = pd.Timestamp.today().normalize()
    end_date = out["폐업일자"].fillna(today)
    out["tenure_days"] = (end_date - out["개업일자"]).dt.days
    out["tenure_years"] = out["tenure_days"] / 365.25
    out["개업연도"] = out["개업일자"].dt.year

    # placeholder로 의심되는 개업일자는 원본 개업일자는 보존하되 tenure/개업연도만 결측 처리
    placeholder_date = out["개업연도"] < PLACEHOLDER_YEAR_THRESHOLD
    out.loc[placeholder_date, ["tenure_days", "tenure_years", "개업연도"]] = float("nan")

    out["y"] = (out["폐업일자"].notna() & (out["폐업일자"] >= TARGET_CUTOFF)).astype(int)

    # 2024년 이전에 이미 폐업한 행은 제외한다. 업종밀집도/대중교통 접근성 같은
    # 피처는 "오늘" 기준으로 계산되는데, 오래전에 폐업한 가게에 현재 시점 피처를
    # 붙이면 실제 폐업 당시 환경과 무관한 값을 학습시키게 된다. y=0은
    # "현재 영업 중"만을 의미하도록 유지한다.
    stale_closure = out["폐업일자"].notna() & (out["폐업일자"] < TARGET_CUTOFF)
    out = out.loc[~stale_closure].copy()

    # 좌표X/Y(원본 TM좌표, 경도/위도로 변환 후엔 안 씀)와 tenure_days(tenure_years와
    # 100% 중복, 스케일만 다름)는 최종 결과에서 제외한다.
    out = out.drop(columns=["좌표X", "좌표Y", "tenure_days"])

    return out.reset_index(drop=True)
