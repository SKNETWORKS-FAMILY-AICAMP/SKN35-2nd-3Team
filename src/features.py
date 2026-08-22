"""공간 피처(업종 밀집도, 대중교통 접근성) 생성.

3백만 행 규모에서 pandas 순수 반복/브로드캐스트 거리계산은 비현실적이라
BallTree(haversine)로 공간 인덱싱해서 계산한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000.0


def _to_radians(lat: pd.Series, lon: pd.Series) -> np.ndarray:
    return np.radians(np.column_stack([lat.to_numpy(), lon.to_numpy()]))


def add_industry_density(
    df: pd.DataFrame,
    *,
    lat_col: str = "위도",
    lon_col: str = "경도",
    group_col: str = "업종명",
    radius_m: float = 300.0,
    out_col: str = "업종밀집도",
    active_mask: pd.Series,
) -> pd.DataFrame:
    """같은 업종명 그룹 내에서 반경(radius_m) 안에 있는 '현재 영업 중인' 동일업종 점포 수를 센다.

    active_mask는 필수: "현재 영업 중"을 나타내는 마스크를 넘겨서 경쟁 점포 풀을
    한정해야 한다 (예: all_industries는 영업상태=='영업/정상', 소매업 스냅샷은 y==0).
    이걸 생략하고 df 전체(폐업 이력 포함)를 풀로 쓰면, 인허가 데이터는 수십 년치
    누적 개/폐업 이력이라 밀집도가 크게 부풀려진다.
    쿼리 대상(카운트를 매기는 행)은 폐업 이력 포함 전체 df — "그 위치/그 시점에 주변 경쟁이
    얼마나 있었는가"를 모든 행에 부여하기 위함이다.
    """
    df = df.copy()
    df[out_col] = np.nan
    radius_rad = radius_m / EARTH_RADIUS_M

    active_df = df.loc[active_mask]

    for 업종, idx in df.groupby(group_col).groups.items():
        query_sub = df.loc[idx]
        pool_sub = active_df.loc[active_df[group_col] == 업종]

        query_valid = query_sub[lat_col].notna() & query_sub[lon_col].notna()
        pool_valid = pool_sub[lat_col].notna() & pool_sub[lon_col].notna()
        if query_valid.sum() == 0 or pool_valid.sum() == 0:
            continue

        pool_coords = _to_radians(pool_sub.loc[pool_valid, lat_col], pool_sub.loc[pool_valid, lon_col])
        tree = BallTree(pool_coords, metric="haversine")

        query_coords = _to_radians(query_sub.loc[query_valid, lat_col], query_sub.loc[query_valid, lon_col])
        counts = tree.query_radius(query_coords, r=radius_rad, count_only=True)

        # 쿼리한 점 자신이 활성 풀에도 속해 있으면 카운트에 자기 자신이 포함되므로 1을 빼준다
        self_in_pool = query_sub.loc[query_valid].index.isin(pool_sub.loc[pool_valid].index)
        counts = counts - self_in_pool.astype(int)

        df.loc[query_sub.loc[query_valid].index, out_col] = counts

    return df


def add_transit_accessibility(
    df: pd.DataFrame,
    bus_stops: pd.DataFrame,
    subway_stations: pd.DataFrame,
    *,
    lat_col: str = "위도",
    lon_col: str = "경도",
    bus_lat_col: str = "위도",
    bus_lon_col: str = "경도",
    subway_lat_col: str = "역위도",
    subway_lon_col: str = "역경도",
) -> pd.DataFrame:
    """각 지점에서 가장 가까운 버스정류장/지하철역까지 거리(m)를 계산한다."""
    df = df.copy()
    valid = df[lat_col].notna() & df[lon_col].notna()
    query_coords = _to_radians(df.loc[valid, lat_col], df.loc[valid, lon_col])

    bus_valid = bus_stops[bus_lat_col].notna() & bus_stops[bus_lon_col].notna()
    bus_tree = BallTree(
        _to_radians(bus_stops.loc[bus_valid, bus_lat_col], bus_stops.loc[bus_valid, bus_lon_col]),
        metric="haversine",
    )
    bus_dist, _ = bus_tree.query(query_coords, k=1)
    df.loc[valid, "최근접버스정류장_거리m"] = bus_dist[:, 0] * EARTH_RADIUS_M

    subway_valid = subway_stations[subway_lat_col].notna() & subway_stations[subway_lon_col].notna()
    subway_tree = BallTree(
        _to_radians(
            subway_stations.loc[subway_valid, subway_lat_col],
            subway_stations.loc[subway_valid, subway_lon_col],
        ),
        metric="haversine",
    )
    subway_dist, _ = subway_tree.query(query_coords, k=1)
    df.loc[valid, "최근접지하철역_거리m"] = subway_dist[:, 0] * EARTH_RADIUS_M

    return df
