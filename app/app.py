"""
진입점 (Streamlit)

역할: 첫 화면에서 로그인 상태(GUEST/NEW_MEMBER/OWNER/ADMIN)를 판별해서
      그에 맞는 메인 화면(지도+우측 패널)을 보여준다. (seoul-biz-ui-logic.md 1, 2번,
      seoul-biz-main-ui-mockup.md의 2컬럼 레이아웃 그대로)

실행:
    streamlit run app/app.py

주의 — 아직 안 붙은 부분(모델 연동 전이라 실제 값 대신 안내 문구로 대체):
  - GUEST/NEW_MEMBER 지도의 히트맵은 원래 predictions(모델 폐업확률) 가중이어야
    하는데, 아직 학습된 모델이 앱에 연동되지 않아서 store_snapshots.is_closed_next
    실측 평균으로 임시 대체했다 (_dong_survival_proxy 함수 주석 참고).
  - 지역 상세 클릭 시 업종별 생존점수 랭킹(ui-logic.md 4번)은 모델 배치 추론이
    필요해서 아직 준비 중 안내만 뜬다.
  - OWNER 화면의 "내 업종 생존점수"는 predictions 테이블에 이미 캐시된 값이
    있을 때만 보여주고, 없으면 모델 연동 전까지는 안내 문구만 뜬다.
  - "지금 뜨는 동네"(ui-logic.md 5번)는 growth_slope(시계열 증가율)를 뺀
    new_store_ratio/survival_ratio 두 지표만으로 계산한 v1이다. 문서에서 제안한
    dong_hot_index 캐시 테이블을 실제로 만들지는 아직 팀 논의가 필요하다.
"""

import json
import math
import sys
from pathlib import Path
from urllib.request import urlopen

# Streamlit은 이 파일(app/app.py)이 있는 폴더(app/)를 sys.path 맨 앞에 넣고 실행한다.
# 그런데 이 폴더 안에 이 파일 자신이 "app.py"라는 이름으로 있다 보니, 파이썬이
# "app"이라는 이름을 찾을 때 진짜 app/ 패키지(프로젝트 루트 밑)보다 이 app.py
# 파일 자체를 먼저 "app" 모듈로 착각해버린다 ("app"+".py"로 매치) — 그래서
# `from app.shared import ...`가 "ModuleNotFoundError: ... 'app' is not a
# package"로 깨진다. 최초 1회는 우연히 성공하기도 하는데, streamlit이 재실행할
# 때마다 이 폴더를 sys.path 맨 앞에 다시 꽂아 넣어서 결국 항상 이 에러로 귀결된다.
# 그래서 프로젝트 루트를 넣고 "app.shared"로 부르는 대신, app/ 폴더 자체를
# sys.path에 넣고 "shared"를 최상위 이름으로 바로 가져온다 — 이름 충돌 자체를
# 피하는 방식이라 재실행해도 안전하다.
_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

import folium
import numpy as np
import streamlit as st
from scipy.spatial import ConvexHull, Voronoi
from sqlalchemy import text
from streamlit_folium import st_folium
import base64

from shared import auth
from shared import components as ui
from shared.db import get_engine
# 프로덕션 모델 조회는 실행 PC의 로컬 .env가 연결한 TiDB를 기준으로 한다.
from shared.query_predictions import get_prediction_for_store
from shared.write_dong_view import increment_dong_view
from shared.write_user_view import increment_user_view

st.set_page_config(
    page_title="서울 상권 폐업예측",
    page_icon=":material/location_city:",
    layout="wide",
)

_MY_PAGE_EXISTS = (Path(__file__).resolve().parent / "pages" / "mypage.py").exists()
_BRAND_LOGO_PATH = str(Path(__file__).resolve().parent / "assets" / "brand_logo.png")
_BRAND_LOGO_EXISTS = Path(_BRAND_LOGO_PATH).exists()


def _wilson_lower_bound(p: float, n: int, z: float = 1.96) -> float:
    """Wilson score interval 하한(95%) — 표본이 적을수록 신뢰구간이 넓어져서
    하한이 크게 깎인다. survival_rate 원값으로만 정렬하면 표본 1~2건짜리가
    우연히 100%를 찍고 1위를 차지하는 문제(예: "중고 상품 소매업 100점,
    표본 1건 기준")를 막기 위해 도입."""
    if n <= 0:
        return 0.0
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)

def _rank_and_total_desc(value: float, all_values: list[float]) -> tuple[int, int] | tuple[None, None]:
    """값이 클수록 좋은 지표에서 몇 등인지(1등이 최고)와 전체 개수를 반환.
    "상위 86%"보다 "423개 동 중 364위"가 기저 폐업률 10.6%로 점수가 다 몰려있는
    상황(2026-08-28 지적: 평균 대비 점수차가 작아 극적으로 안 느껴짐)에서 순위
    차이를 더 직관적으로 전달한다."""
    if not all_values:
        return None, None
    total = len(all_values)
    rank = sum(1 for v in all_values if v > value) + 1
    return rank, total

# ---------------------------------------------------------------
# 데이터 조회 (읽기 전용, 전부 st.cache_data로 캐싱 — DB 부하 방지)
# ---------------------------------------------------------------
def _snapshot_minus_months(snapshot: str, months: int) -> str:
    """'YYYYMM' 문자열에서 months개월 전 'YYYYMM' 계산."""
    y, m = int(snapshot[:4]), int(snapshot[4:6])
    total = y * 12 + (m - 1) - months
    y2, m2 = divmod(total, 12)
    return f"{y2:04d}{m2 + 1:02d}"


@st.cache_data(ttl=3600)
def _dong_name_map() -> dict:
    engine = get_engine()
    if engine is None:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT dong_code, dong_name, gu_name FROM administrative_dongs")
        ).mappings().all()
    return {r["dong_code"]: f"{r['gu_name']} {r['dong_name']}" for r in rows}


@st.cache_data(ttl=3600)
def _industry_name_map() -> dict:
    engine = get_engine()
    if engine is None:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT industry_code, industry_name FROM industries")
        ).mappings().all()
    return {r["industry_code"]: r["industry_name"] for r in rows}

@st.cache_data(ttl=3600)
def _industry_options() -> list[tuple[str, str]]:
    """예비창업자 패널의 업종 선택 드롭다운용. 모델링 제외 업종군(3-2 문서:
    과학·기술/부동산/시설관리·임대)은 후보에서 제외."""
    engine = get_engine()
    if engine is None:
        return []
    sql = text(
        "SELECT industry_code, industry_name, custom_group FROM industries ORDER BY industry_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [
        (r["industry_code"], r["industry_name"])
        for r in rows
        if not ui.is_excluded_industry(r["custom_group"])
    ]


@st.cache_data(ttl=3600)
def _top_dongs_for_industry(industry_code: str, top_n: int = 3) -> list[dict]:
    """특정 업종 기준 동별 실측 생존율 랭킹 — 모델 연동 전 임시 대체값
    (_dong_survival_proxy와 동일한 패턴: store_snapshots.is_closed_next 실측 평균).
    survival_rate 원값으로 정렬하면 표본 5~8건짜리가 우연히 100%를 찍고 나란히
    1~3위를 차지하는 문제(2026-08-28 지적, 예: "기타 서양식 음식점" 8곳/5곳/7곳
    전부 100점) — 업종전환 추천과 동일하게 Wilson 하한으로 정렬해서 표본이
    적을수록 점수를 보수적으로 깎는다. 최소 표본도 5 -> 10으로 올림."""
    engine = get_engine()
    if engine is None:
        return []
    sql = text(
        """
        SELECT dong_code, COUNT(*) AS n,
               1 - AVG(CASE WHEN is_closed_next THEN 1 ELSE 0 END) AS survival_rate
        FROM store_snapshots
        WHERE industry_code = :industry_code
        GROUP BY dong_code
        HAVING n >= 10
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"industry_code": industry_code}).mappings().all()
    names = _dong_name_map()

    scored = []
    for r in rows:
        n = float(r["n"])
        survival_rate = float(r["survival_rate"])
        wilson_score = _wilson_lower_bound(survival_rate, int(n))
        scored.append(
            {
                "dong_code": r["dong_code"],
                "dong_name": names.get(r["dong_code"], r["dong_code"]),
                "n": n,
                "survival_rate": survival_rate,
                "wilson_score": wilson_score,
                "low_confidence": n < 30,
            }
        )
    scored.sort(key=lambda x: x["wilson_score"], reverse=True)
    return scored[:top_n]

@st.cache_data(ttl=3600)
def _transition_counts_from(industry_code: str) -> dict:
    """industry_transitions(실제 전환 이력)에서 현재 업종(from) 기준 목적지별
    전환 건수. industry_survival_stats(생존율)와 별개 테이블이라, "생존율은
    높은데 실제로 아무도 안 간 업종"과 "실제로 많이들 갔고 생존율도 높은 업종"을
    구분해서 보여줄 수 있게 함(2026-08-28 추가)."""
    engine = get_engine()
    if engine is None:
        return {}
    sql = text(
        """
        SELECT to_industry_code, COUNT(*) AS n
        FROM industry_transitions
        WHERE from_industry_code = :industry_code
        GROUP BY to_industry_code
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"industry_code": industry_code}).mappings().all()
    return {r["to_industry_code"]: int(r["n"]) for r in rows}


@st.cache_data(ttl=3600)
def _industry_switch_recommendations(industry_code: str, top_n: int = 3) -> list[dict]:
    ...


@st.cache_data(ttl=3600)
def _industry_switch_recommendations(industry_code: str, top_n: int = 3) -> list[dict]:
    """업종전환 추천(2026-08-28 추가, 2026-08-28 Wilson 하한 정렬로 수정):
    industry_survival_stats에서 현재 업종(from) 기준 전환 후보(to)를 뽑는다.
    survival_rate 원값으로 정렬하면 표본 1~2건짜리가 우연히 100%를 찍고 1위를
    차지하는 문제가 있어서(예: "중고 상품 소매업 100점, 표본 1건 기준"), Wilson
    score 하한으로 정렬한다 — 표본이 적을수록 점수가 보수적으로 깎인다. 3-2 문서
    규칙대로 모델링 제외 업종군(과학·기술/부동산/시설관리·임대)은 전환 "목적지"로
    추천하면 안 되므로 industries.custom_group으로 걸러낸다(ui.is_excluded_industry 재사용)."""
    engine = get_engine()
    if engine is None:
        return []
    sql = text(
        """
        SELECT s.to_industry_code, i.industry_name, i.custom_group,
               s.survival_rate, s.sample_size
        FROM industry_survival_stats s
        JOIN industries i ON i.industry_code = s.to_industry_code
        WHERE s.from_industry_code = :industry_code
          AND s.to_industry_code != :industry_code
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"industry_code": industry_code}).mappings().all()

    transition_counts = _transition_counts_from(industry_code)

    recs = []
    for r in rows:
        if ui.is_excluded_industry(r["custom_group"]):
            continue
        d = dict(r)
        d["wilson_score"] = _wilson_lower_bound(float(r["survival_rate"]), int(r["sample_size"]))
        d["low_confidence"] = r["sample_size"] < 30
        d["transition_count"] = transition_counts.get(r["to_industry_code"], 0)
        recs.append(d)

    recs.sort(key=lambda x: x["wilson_score"], reverse=True)
    return recs[:top_n]


@st.cache_data(ttl=3600)
def _dong_survival_proxy() -> list[dict]:
    """동별 대표 좌표 + 실측 생존율(임시 대체값, 모듈 docstring 참고).
    store_snapshots 하나만 GROUP BY하는 단일 집계 쿼리라 수백만 행이어도
    DB 쪽에서 처리되고, 결과는 동 개수(수백 건)만 파이썬으로 돌아온다."""
    engine = get_engine()
    if engine is None:
        return []
    sql = text(
        """
        SELECT dong_code, AVG(lat) AS lat, AVG(lng) AS lng,
               1 - AVG(CASE WHEN is_closed_next THEN 1 ELSE 0 END) AS survival_rate,
               COUNT(*) AS n
        FROM store_snapshots
        GROUP BY dong_code
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [
        {
            "dong_code": r["dong_code"],
            "lat": float(r["lat"]),
            "lng": float(r["lng"]),
            "survival_rate": float(r["survival_rate"]),
            "n": r["n"],
        }
        for r in rows
        if r["lat"] is not None and r["lng"] is not None
    ]


@st.cache_data(ttl=3600)
def _hot_keyword_ranking(top_n: int = 3) -> list[dict]:
    """관리자 대시보드(pages/admin_dashboard.py의 "지금 뜨는 사업" 탭)에서 이미 쓰던
    trend_keywords 기반 랭킹을 GUEST 메인화면에서도 재사용. "지금 뜨는 동네" 카드를
    3위까지만 보여주기로 하면서 그 밑이 비니, "밑에는 좀 다른걸 채우는건 어떤지 —
    인기 순위 뭐 이런것도 해놨으니까"(2026-08-27, 사용자 요청)에 따라 채워 넣음.
    2026-08-28 수정: 최초 스냅샷 매장수(first_store_count)도 같이 조회해서
    "19개 → 54개"처럼 증가 흐름을 보여줄 수 있게 함(비율만으론 설득력이 부족하다는 피드백)."""
    engine = get_engine()
    if engine is None:
        return []
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(snapshot_date) FROM trend_keywords")).scalar()
        if latest is None:
            return []
        rows = conn.execute(
            text(
                """
                SELECT t.keyword, t.store_count, t.growth_rate,
                       (SELECT t2.store_count FROM trend_keywords t2
                        WHERE t2.keyword = t.keyword
                        ORDER BY t2.snapshot_date ASC LIMIT 1) AS first_store_count
                FROM trend_keywords t
                WHERE t.snapshot_date = :latest
                ORDER BY t.growth_rate DESC
                LIMIT :n
                """
            ),
            {"latest": latest, "n": top_n},
        ).mappings().all()
    return [dict(r) for r in rows]

@st.cache_data(ttl=3600)
def _citywide_survival_avg() -> float | None:
    """_dong_survival_proxy를 재사용해서 서울 전체 가중평균 생존율을 계산.
    개별 동 카드에 "이 동네가 평균보다 높은지" 비교 기준선을 주기 위함."""
    points = _dong_survival_proxy()
    if not points:
        return None
    total_n = sum(p["n"] for p in points)
    if total_n == 0:
        return None
    weighted = sum(p["survival_rate"] * p["n"] for p in points)
    return weighted / total_n * 100


@st.cache_data(ttl=3600)
def _hot_dong_ranking(top_n: int = 5) -> list[dict]:
    engine = get_engine()
    if engine is None:
        return []
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(last_seen_snapshot) FROM stores")).scalar()
    if latest is None:
        return []
    cutoff = _snapshot_minus_months(latest, 3)
    sql = text(
        """
        SELECT dong_code, COUNT(*) AS total_stores,
               SUM(CASE WHEN first_seen_snapshot >= :cutoff THEN 1 ELSE 0 END) AS new_stores,
               SUM(CASE WHEN is_closed THEN 1 ELSE 0 END) AS closed_stores
        FROM stores
        GROUP BY dong_code
        HAVING total_stores >= 20
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"cutoff": cutoff}).mappings().all()

    # "신규매장은 많은데 그만큼 빨리 망하는" 동이 단순평균 때문에 1위로 뽑히는
    # 문제(2026-08-28 지적, 예: 응암2동 신규매장 9%로 1위인데 생존율은 서울
    # 평균보다 23%p 낮음) — 서울 평균 생존율보다 낮은 동은 "뜨는 동네" 후보에서
    # 아예 제외한다. "뜨는 동네"는 성장과 안정이 같이 있어야 의미가 있다는 판단.
    citywide_avg_ratio = (_citywide_survival_avg() or 0) / 100

    names = _dong_name_map()
    scored = []
    for r in rows:
        total_stores = float(r["total_stores"])
        new_stores = float(r["new_stores"])
        closed_stores = float(r["closed_stores"])
        new_ratio = new_stores / total_stores
        survival_ratio = 1 - (closed_stores / total_stores)
        if survival_ratio < citywide_avg_ratio:
            continue
        # growth_slope는 아직 빠져있음(모듈 docstring 참고) — 두 지표 단순 평균.
        hot_index = (new_ratio + survival_ratio) / 2
        scored.append(
            {
                "dong_code": r["dong_code"],
                "dong_name": names.get(r["dong_code"], r["dong_code"]),
                "new_ratio": new_ratio,
                "survival_ratio": survival_ratio,
                "total_stores": total_stores,
                "new_stores": new_stores,
                "hot_index": hot_index,
            }
        )
    scored.sort(key=lambda s: s["hot_index"], reverse=True)
    return scored[:top_n]


@st.cache_data(ttl=600)
def _owner_latest_snapshot(store_id: str) -> dict | None:
    engine = get_engine()
    if engine is None:
        return None
    sql = text(
        """
        SELECT snapshot_date, lat, lng, dong_code, industry_code, store_name
        FROM store_snapshots WHERE store_id = :store_id
        ORDER BY snapshot_date DESC LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"store_id": store_id}).mappings().first()
    return dict(row) if row else None


@st.cache_data(ttl=600)
def _owner_competition_density(store_id: str, snapshot_date: str) -> dict | None:
    """spatial_density_features(반경 300m 내 동일업종/전체 매장 수, 20m 내 클러스터
    크기, 가장 가까운 동일업종 매장까지 거리)를 owner 패널에서 활용."""
    engine = get_engine()
    if engine is None:
        return None
    sql = text(
        """
        SELECT same_industry_count_300m, total_count_300m, coord_cluster_size,
               nearest_same_industry_distance_m
        FROM spatial_density_features
        WHERE store_id = :store_id AND snapshot_date = :snapshot_date
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"store_id": store_id, "snapshot_date": snapshot_date}).mappings().first()
    return dict(row) if row else None

@st.cache_data(ttl=3600)
def _population_feature(dong_code: str) -> dict | None:
    engine = get_engine()
    if engine is None:
        return None
    sql = text(
        """
        SELECT korean_pop, foreign_long_pop, foreign_short_pop, total_pop_avg,
               foreign_short_ratio, tourist_zone_candidate
        FROM population_features WHERE dong_code = :dong_code
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"dong_code": dong_code}).mappings().first()
    return dict(row) if row else None

@st.cache_data(ttl=3600)
def _dong_compare_stats(dong_code: str) -> dict | None:
    """동네 비교 카드용 — 생존율/유동인구/경쟁밀도 요약 한 번에 조회.
    _dong_survival_proxy, _population_feature와 같은 원본 데이터를 재사용하되
    비교 카드 하나에 필요한 값만 모아서 반환."""
    match = next((p for p in _dong_survival_proxy() if p["dong_code"] == dong_code), None)
    pop = _population_feature(dong_code)
    return {
        "survival_score": round(match["survival_rate"] * 100) if match else None,
        "total_pop_avg": float(pop["total_pop_avg"]) if pop else None,
    }

@st.cache_data(ttl=3600)
def _dong_top_industries(dong_code: str, top_n: int = 3) -> list[dict]:
    """지역상세 패널용 — 해당 동에 실제로 많은 업종 구성(최신 스냅샷 기준 매장수
    순위). 모델 배치추론이 필요한 "업종별 생존점수 랭킹"(ui-logic.md 4번)과는
    다르게, 이건 이미 있는 store_snapshots만으로 바로 보여줄 수 있는 실측 데이터."""
    engine = get_engine()
    if engine is None:
        return []
    sql = text(
        """
        SELECT s.industry_code, i.industry_name, COUNT(*) AS n
        FROM store_snapshots s
        JOIN industries i ON i.industry_code = s.industry_code
        WHERE s.dong_code = :dong_code
          AND s.snapshot_date = (SELECT MAX(snapshot_date) FROM store_snapshots)
        GROUP BY s.industry_code, i.industry_name
        ORDER BY n DESC
        LIMIT :top_n
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"dong_code": dong_code, "top_n": top_n}).mappings().all()
    return [{"industry_name": r["industry_name"], "n": float(r["n"])} for r in rows]

def _latest_owner_prediction(store_id: str) -> dict | None:
    """현재 TiDB의 프로덕션 모델 기준으로 기존 점주의 최신 예측값을 조회한다."""
    return get_prediction_for_store(store_id)


def _nearest_dong(lat: float, lng: float) -> str | None:
    """A-1: 최근접 중심점으로 dong_code 판별 (schema-ui-mapping.md) — GeoJSON 없이
    동별 평균 좌표(_dong_survival_proxy)를 중심점 삼아 유클리드 거리로 근사."""
    points = _dong_survival_proxy()
    if not points:
        return None
    best, best_d = None, float("inf")
    for p in points:
        d = (p["lat"] - lat) ** 2 + (p["lng"] - lng) ** 2
        if d < best_d:
            best, best_d = p["dong_code"], d
    return best


# ---------------------------------------------------------------
# 지도 — 동 경계 색칠 (Voronoi 기반)
#
# 히트맵(흐릿하게 번지는 색)은 사용자 피드백으로 뺐다("히트맵으로 구분하지 말고
# 경계선 같은게 낫지 않나", 2026-08-27). 문제는 실제 행정동 폴리곤(GeoJSON)이
# 없다는 것 — schema-ui-mapping.md에서 이미 "GeoJSON 없이 동별 평균 좌표를
# 중심점 삼아 최근접 거리로 동을 판별한다"(A-1)로 정해뒀다. 그 판별 기준을 그대로
# 그림으로 그리면 동 경계가 된다: 어떤 지점이 동 A의 중심점에 가장 가까우면 그
# 지점은 동 A 영역이라는 게 A-1의 정의이고, "각 중심점에 가장 가까운 영역들로
# 지도를 나누는 것"이 정확히 보로노이(Voronoi) 다이어그램의 정의라서 둘이 완전히
# 같은 기준이다. 그래서 진짜 행정동 경계는 아니고 근사치이지만(실제 동 경계는
# 인구/도로 등을 따라 삐뚤빼뚤하지, 중심점 사이의 수직이등분선이 아님), 적어도
# "지도 어디를 클릭하면 어느 동으로 잡히는지"와는 100% 일치하는 경계선이다.
# ---------------------------------------------------------------
def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """scipy Voronoi는 가장자리 셀을 무한 영역으로 남겨두는데, 지도에 그리려면
    유한한 다각형이어야 한다. 무한 방향 능선을 화면 밖 먼 지점까지로 잘라서 닫힌
    다각형으로 만들어주는 표준 트릭(각 사이트 주변 능선을 각도순으로 정렬해서
    부채꼴을 완성)."""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # 이미 닫힌 능선
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _clip_polygon(subject: list[tuple[float, float]], bbox: tuple[float, float, float, float]):
    """서덜랜드-호지먼(Sutherland-Hodgman) 다각형 클리핑 — 무한 방향으로 뻗어나간
    가장자리 셀을 지도 표시 범위(bbox) 안으로 잘라낸다. 외부 라이브러리(shapely)
    없이 표준 알고리즘만으로 구현."""
    xmin, ymin, xmax, ymax = bbox

    def inside(p, edge):
        x, y = p
        if edge == "left":
            return x >= xmin
        if edge == "right":
            return x <= xmax
        if edge == "bottom":
            return y >= ymin
        return y <= ymax

    def intersect(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        if edge in ("left", "right"):
            xedge = xmin if edge == "left" else xmax
            t = (xedge - x1) / (x2 - x1) if x2 != x1 else 0
            return (xedge, y1 + t * (y2 - y1))
        yedge = ymin if edge == "bottom" else ymax
        t = (yedge - y1) / (y2 - y1) if y2 != y1 else 0
        return (x1 + t * (x2 - x1), yedge)

    output = subject
    for edge in ("left", "right", "bottom", "top"):
        if not output:
            break
        input_list = output
        output = []
        n = len(input_list)
        for i in range(n):
            cur, prev = input_list[i], input_list[i - 1]
            cur_in, prev_in = inside(cur, edge), inside(prev, edge)
            if cur_in:
                if not prev_in:
                    output.append(intersect(prev, cur, edge))
                output.append(cur)
            elif prev_in:
                output.append(intersect(prev, cur, edge))
    return output


def _clip_polygon_to_convex(
    subject_latlngs: list[tuple[float, float]],
    clip_xy: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """위경도 폴리곤을 반시계 방향의 볼록 경계 안으로 자른다."""
    subject = [(lng, lat) for lat, lng in subject_latlngs]

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersection(start, end, clip_start, clip_end):
        direction = (end[0] - start[0], end[1] - start[1])
        clip_direction = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
        denominator = (
            direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
        )
        if abs(denominator) < 1e-12:
            return end
        offset = (clip_start[0] - start[0], clip_start[1] - start[1])
        ratio = (
            offset[0] * clip_direction[1] - offset[1] * clip_direction[0]
        ) / denominator
        return (start[0] + ratio * direction[0], start[1] + ratio * direction[1])

    output = subject
    for clip_start, clip_end in zip(clip_xy, clip_xy[1:] + clip_xy[:1]):
        if not output:
            break
        input_list = output
        output = []
        for index, current in enumerate(input_list):
            previous = input_list[index - 1]
            current_inside = cross(clip_start, clip_end, current) >= -1e-10
            previous_inside = cross(clip_start, clip_end, previous) >= -1e-10
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current, clip_start, clip_end))

    return [(lat, lng) for lng, lat in output]


def _point_in_geojson_geometry(lng: float, lat: float, geometry: dict) -> bool:
    """외부 GIS 의존성 없이 점이 GeoJSON Polygon/MultiPolygon 안인지 판정한다."""
    def in_ring(ring: list[list[float]]) -> bool:
        inside = False
        previous = len(ring) - 1
        for current, (current_lng, current_lat, *_) in enumerate(ring):
            previous_lng, previous_lat, *_ = ring[previous]
            crosses_latitude = (current_lat > lat) != (previous_lat > lat)
            if crosses_latitude:
                crossing_lng = (
                    (previous_lng - current_lng)
                    * (lat - current_lat)
                    / (previous_lat - current_lat)
                    + current_lng
                )
                if lng < crossing_lng:
                    inside = not inside
            previous = current
        return inside

    def in_polygon(rings: list[list[list[float]]]) -> bool:
        return bool(rings) and in_ring(rings[0]) and not any(
            in_ring(hole) for hole in rings[1:]
        )

    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        return in_polygon(coordinates)
    if geometry.get("type") == "MultiPolygon":
        return any(in_polygon(polygon) for polygon in coordinates)
    return False


def _dong_boundary_cells(points: list[dict]) -> list[tuple[list[tuple[float, float]], dict]]:
    """동 중심점들로 보로노이 다각형을 만들어 [(위경도 좌표 리스트, 원본 point), ...]
    로 반환. 점이 4개 미만이면(보로노이가 성립 안 함) 빈 리스트."""
    if len(points) < 4:
        return []

    xy = np.array([[p["lng"], p["lat"]] for p in points])
    vor = Voronoi(xy)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    lat_min = min(p["lat"] for p in points) - 0.02
    lat_max = max(p["lat"] for p in points) + 0.02
    lng_min = min(p["lng"] for p in points) - 0.02
    lng_max = max(p["lng"] for p in points) + 0.02
    bbox = (lng_min, lat_min, lng_max, lat_max)

    cells = []
    for i, region in enumerate(regions):
        poly_xy = [tuple(vertices[v]) for v in region]
        poly_xy = _clip_polygon(poly_xy, bbox)
        if len(poly_xy) < 3:
            continue
        latlngs = [(y, x) for x, y in poly_xy]  # folium은 (lat, lng) 순서
        cells.append((latlngs, points[i]))
    return cells


# 지도 div의 고정 폭. fit_bounds가 "서울 밖 지역이 빠져나온다"는 피드백을 준 진짜
# 원인은, 지도 div의 가로세로 비율이 서울의 실제 모양(가로가 세로보다 살짝 더 긴
# 형태)과 안 맞았던 것 — width=None이면 파이썬 쪽에서 실제 렌더 폭을 알 수 없어서
# 세로(height)를 서울 모양에 맞게 계산할 방법이 없었다. 그래서 폭은 고정값으로
# 두고, 세로는 아래 _build_map에서 서울 동 중심점들의 실제 위경도 범위(bbox)
# 종횡비에 맞춰 매번 계산한다(2026-08-27, "저거 빠져나오는거 어떻게 잘 못 맞추나,
# 배치를 다르게 해도 괜찮아" 피드백 반영).
_MAP_WIDTH = 820
_SEOUL_DISTRICT_GEOJSON_URL = (
    "https://raw.githubusercontent.com/southkorea/seoul-maps/master/"
    "kostat/2013/json/seoul_municipalities_geo_simple.json"
)
_SEOUL_DONG_GEOJSON_URL = (
    "https://raw.githubusercontent.com/southkorea/seoul-maps/master/"
    "kostat/2013/json/seoul_submunicipalities_geo_simple.json"
)


@st.cache_data(ttl=86400, show_spinner=False)
def _seoul_district_geojson() -> dict | None:
    """서울 25개 구 단순 경계를 하루 동안 캐시하고, 오프라인이면 None을 반환."""
    try:
        with urlopen(_SEOUL_DISTRICT_GEOJSON_URL, timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _seoul_dong_geojson() -> dict | None:
    """서울 행정동 단순 경계를 하루 동안 캐시하고, 오프라인이면 None을 반환."""
    try:
        with urlopen(_SEOUL_DONG_GEOJSON_URL, timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def _build_map(mode: str, owner_snapshot: dict | None, clicked: dict | None):
    # 첨부 시안의 면 지도 표현을 현재 데이터에 적용한다. 최초에는 서울 자치구,
    # 지역 선택 후에는 해당 구의 실제 행정동 경계를 표시하고 네트워크가 없을 때만
    # 기존 동 중심점 Voronoi 셀을 근사 지도 폴백으로 사용한다.
    def styled_map(location: list[float], zoom_start: int) -> folium.Map:
        map_obj = folium.Map(
            location=location,
            zoom_start=zoom_start,
            tiles=None,
            prefer_canvas=False,
            control_scale=False,
        )
        folium.TileLayer(
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="&copy; OpenStreetMap contributors",
            name="밝은 지도",
            overlay=False,
            control=False,
            opacity=0.18,
        ).add_to(map_obj)
        map_obj.get_root().html.add_child(
            folium.Element(
                """
                <style>
                .leaflet-container {
                    background: #EAF3FF !important;
                    font-family: Pretendard, "Noto Sans KR", sans-serif;
                }
                .leaflet-interactive:focus { outline: none; }
                .leaflet-control-zoom {
                    border: 1px solid #D9E4F2 !important;
                    border-radius: 12px !important;
                    box-shadow: 0 4px 14px rgba(23, 78, 145, 0.12) !important;
                    overflow: hidden;
                }
                .leaflet-control-zoom a {
                    color: #174E91 !important;
                    border-color: #E6EDF6 !important;
                }
                .leaflet-tooltip {
                    background: #303238;
                    border: 0;
                    border-radius: 999px;
                    box-shadow: 0 4px 14px rgba(18, 34, 53, 0.2);
                    color: #FFFFFF;
                    font-weight: 700;
                    padding: 7px 11px;
                }
                .leaflet-tooltip::before { display: none; }
                .leaflet-tooltip.district-label {
                    background: transparent;
                    box-shadow: none;
                    color: #27364A;
                    font-size: 0.72rem;
                    font-weight: 800;
                    padding: 0;
                    text-shadow: 0 1px 2px rgba(255, 255, 255, 0.95);
                    white-space: nowrap;
                }
                .leaflet-tooltip.dong-label {
                    background: rgba(255, 255, 255, 0.78);
                    border: 0;
                    border-radius: 6px;
                    box-shadow: none;
                    color: #34465D;
                    font-size: 0.64rem;
                    font-weight: 700;
                    padding: 2px 4px;
                    white-space: nowrap;
                }
                .leaflet-tooltip.selected-district-label {
                    background: rgba(35, 38, 43, 0.94);
                    border-radius: 9px;
                    box-shadow: 0 5px 16px rgba(20, 30, 45, 0.24);
                    color: #FFFFFF;
                    font-size: 0.82rem;
                    font-weight: 800;
                    padding: 7px 10px;
                    white-space: nowrap;
                }
                .leaflet-control-attribution {
                    background: rgba(255, 255, 255, 0.72) !important;
                    color: #748094;
                }
                </style>
                """
            )
        )
        return map_obj

    if mode == "OWNER" and owner_snapshot:
        lat, lng = float(owner_snapshot["lat"]), float(owner_snapshot["lng"])
        m = styled_map([lat, lng], zoom_start=16)
        folium.CircleMarker(
            [lat, lng],
            radius=29,
            color="#F36A2E",
            weight=6,
            opacity=1,
            fill=True,
            fill_color="#174E91",
            fill_opacity=0.96,
            tooltip=owner_snapshot.get("store_name") or "내 가게",
        ).add_to(m)
        folium.Marker(
            [lat, lng],
            tooltip=owner_snapshot.get("store_name") or "내 가게",
            icon=folium.Icon(color="orange", icon="home"),
        ).add_to(m)
        return m, 620  # 한 지점 확대라 종횡비 문제가 없어서 기존 고정 높이 유지

    m = styled_map([37.5665, 126.9780], zoom_start=11)

    points = _dong_survival_proxy()
    names = _dong_name_map()
    selected_dong = _nearest_dong(clicked["lat"], clicked["lng"]) if clicked else None
    selected_name = names.get(selected_dong, "") if selected_dong else ""
    selected_district = selected_name.split()[0] if selected_name else None

    def district_name(point: dict) -> str:
        full_name = names.get(point["dong_code"], point["dong_code"])
        first_part = full_name.split()[0]
        return first_part if first_part.endswith("구") else "기타"

    # 구별 생존율은 기존 동별 생존율을 표본 수로 가중 평균한다. 화면의 진한 파란색은
    # 상대적으로 높은 폐업위험을 뜻하며, 데이터 계산식 자체는 바꾸지 않는다.
    district_stats: dict[str, dict[str, float]] = {}
    for point in points:
        district = district_name(point)
        stats = district_stats.setdefault(
            district,
            {"weighted_survival": 0.0, "samples": 0.0, "lat": 0.0, "lng": 0.0, "count": 0.0},
        )
        samples = max(float(point["n"]), 1.0)
        stats["weighted_survival"] += point["survival_rate"] * samples
        stats["samples"] += samples
        stats["lat"] += point["lat"]
        stats["lng"] += point["lng"]
        stats["count"] += 1

    district_risk = {
        district: 1 - (stats["weighted_survival"] / stats["samples"])
        for district, stats in district_stats.items()
        if district != "기타" and stats["samples"]
    }
    risk_values = list(district_risk.values())
    risk_breaks = (
        np.quantile(risk_values, [0.2, 0.4, 0.6, 0.8]).tolist()
        if risk_values
        else []
    )
    district_palette = ("#E0EDFF", "#C9DFFF", "#AACCF8", "#83B1EC", "#568FD8")
    risk_labels = ("매우 낮음", "낮음", "보통", "높음", "매우 높음")

    geojson_bounds = None
    district_geojson = _seoul_district_geojson()
    dong_geojson = _seoul_dong_geojson() if selected_district else None
    selected_dong_name = (
        selected_name.split(maxsplit=1)[1]
        if selected_name and len(selected_name.split(maxsplit=1)) == 2
        else None
    )
    district_code_by_name = {
        feature.get("properties", {}).get("name"): str(
            feature.get("properties", {}).get("code", "")
        )
        for feature in (district_geojson or {}).get("features", [])
    }
    selected_district_code = district_code_by_name.get(selected_district)
    showing_dong_map = bool(
        selected_district and selected_district_code and dong_geojson
    )
    city_hull_xy: list[tuple[float, float]] = []
    if showing_dong_map:
        all_dong_risks = [1 - point["survival_rate"] for point in points]
        dong_risk_breaks = (
            np.quantile(all_dong_risks, [0.2, 0.4, 0.6, 0.8]).tolist()
            if all_dong_risks
            else []
        )
        point_by_dong_name = {}
        for point in points:
            full_name = names.get(point["dong_code"], point["dong_code"])
            name_parts = full_name.split(maxsplit=1)
            if len(name_parts) == 2 and name_parts[0] == selected_district:
                point_by_dong_name[name_parts[1]] = point

        district_dong_features = [
            feature
            for feature in dong_geojson.get("features", [])
            if str(feature.get("properties", {}).get("code", "")).startswith(
                selected_district_code
            )
        ]
        normalized_feature_names = {
            str(feature.get("properties", {}).get("name", ""))
            .replace("·", ".")
            .replace(" ", ""): feature.get("properties", {}).get("name")
            for feature in district_dong_features
        }
        feature_points: dict[str, list[tuple[str, dict]]] = {
            feature.get("properties", {}).get("name"): []
            for feature in district_dong_features
        }
        visible_dong_names = set()
        for dong_name, point in point_by_dong_name.items():
            normalized_name = dong_name.replace("·", ".").replace(" ", "")
            feature_name = normalized_feature_names.get(normalized_name)
            if not feature_name:
                feature_name = next(
                    (
                        feature.get("properties", {}).get("name")
                        for feature in district_dong_features
                        if _point_in_geojson_geometry(
                            point["lng"], point["lat"], feature.get("geometry", {})
                        )
                    ),
                    None,
                )
            if feature_name:
                feature_points[feature_name].append((dong_name, point))
                visible_dong_names.add(dong_name)

        display_features = []
        for feature in district_dong_features:
            properties = feature.get("properties", {})
            dong_name = properties.get("name")
            matched_points = feature_points.get(dong_name, [])
            matched_samples = sum(
                max(float(point["n"]), 1.0) for _, point in matched_points
            )
            risk = district_risk.get(selected_district, 0.0)
            if matched_samples:
                weighted_survival = sum(
                    point["survival_rate"] * max(float(point["n"]), 1.0)
                    for _, point in matched_points
                )
                risk = 1 - (weighted_survival / matched_samples)
            level = sum(risk > threshold for threshold in dong_risk_breaks)
            display_properties = dict(properties)
            display_properties["fill_color"] = district_palette[level]
            display_properties["risk_text"] = (
                f"폐업위험 {risk_labels[level]} · {risk * 100:.0f}%"
            )
            display_properties["selected"] = any(
                matched_name == selected_dong_name
                for matched_name, _ in matched_points
            )
            display_features.append(
                {
                    "type": "Feature",
                    "properties": display_properties,
                    "geometry": feature.get("geometry"),
                }
            )

        dong_layer = folium.GeoJson(
            {"type": "FeatureCollection", "features": display_features},
            name=f"{selected_district} 행정동",
            style_function=lambda feature: {
                "color": "#F36A2E"
                if feature["properties"]["selected"]
                else "#FFFFFF",
                "weight": 3.8 if feature["properties"]["selected"] else 1.8,
                "opacity": 1,
                "fillColor": feature["properties"]["fill_color"],
                "fillOpacity": 0.8,
            },
            highlight_function=lambda feature: {
                "weight": 3.2,
                "fillOpacity": 0.92,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "risk_text"],
                aliases=["", ""],
                labels=False,
                sticky=True,
            ),
            smooth_factor=1.2,
            zoom_on_click=False,
        ).add_to(m)
        geojson_bounds = dong_layer.get_bounds()
    elif district_geojson:
        display_features = []
        for feature in district_geojson.get("features", []):
            district = feature.get("properties", {}).get("name")
            if district not in district_risk:
                continue
            risk = district_risk[district]
            level = sum(risk > threshold for threshold in risk_breaks)
            properties = dict(feature.get("properties", {}))
            properties["fill_color"] = district_palette[level]
            properties["risk_text"] = (
                f"폐업위험 {risk_labels[level]} · {risk * 100:.0f}%"
            )
            properties["selected"] = district == selected_district
            display_features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": feature.get("geometry"),
                }
            )

        district_layer = folium.GeoJson(
            {"type": "FeatureCollection", "features": display_features},
            name="서울 자치구",
            style_function=lambda feature: {
                "color": "#F36A2E"
                if feature["properties"]["selected"]
                else "#FFFFFF",
                "weight": 3.5 if feature["properties"]["selected"] else 2.2,
                "opacity": 1,
                "fillColor": feature["properties"]["fill_color"],
                "fillOpacity": 0.78,
            },
            highlight_function=lambda feature: {
                "weight": 3.5,
                "fillOpacity": 0.9,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "risk_text"],
                aliases=["", ""],
                labels=False,
                sticky=True,
            ),
            smooth_factor=1.5,
            zoom_on_click=False,
        ).add_to(m)
        geojson_bounds = district_layer.get_bounds()
    else:
        # 네트워크가 없으면 현재 DB의 동 중심점으로 만든 근사 구 지도를 사용한다.
        raw_cells = _dong_boundary_cells(points)
        if len(points) >= 4:
            point_xy = np.array([[point["lng"], point["lat"]] for point in points])
            hull_vertices = point_xy[ConvexHull(point_xy).vertices]
            hull_center = hull_vertices.mean(axis=0)
            expanded_hull = hull_center + ((hull_vertices - hull_center) * 1.07)
            city_hull_xy = [tuple(vertex) for vertex in expanded_hull]

        cells = []
        for latlngs, point in raw_cells:
            clipped = (
                _clip_polygon_to_convex(latlngs, city_hull_xy)
                if city_hull_xy
                else latlngs
            )
            if len(clipped) >= 3:
                cells.append((clipped, point))
        edge_districts: dict[
            tuple[tuple[float, float], tuple[float, float]], list[str]
        ] = {}

        for latlngs, point in cells:
            district = district_name(point)
            if district not in district_risk:
                continue
            risk = district_risk[district]
            level = sum(risk > threshold for threshold in risk_breaks)
            folium.Polygon(
                locations=latlngs,
                color=district_palette[level],
                weight=0,
                opacity=0,
                fill=True,
                fill_color=district_palette[level],
                fill_opacity=0.76,
                smooth_factor=1.5,
                tooltip=folium.Tooltip(
                    f"{district} · 폐업위험 {risk_labels[level]} · {risk * 100:.0f}%"
                ),
            ).add_to(m)

            for start, end in zip(latlngs, latlngs[1:] + latlngs[:1]):
                start_key = (round(start[0], 7), round(start[1], 7))
                end_key = (round(end[0], 7), round(end[1], 7))
                edge_key = tuple(sorted((start_key, end_key)))
                edge_districts.setdefault(edge_key, []).append(district)

        boundary_segments = []
        selected_segments = []
        for edge, owners in edge_districts.items():
            owner_set = set(owners)
            is_boundary = len(owners) == 1 or len(owner_set) > 1
            if not is_boundary:
                continue
            segment = [list(edge[0]), list(edge[1])]
            boundary_segments.append(segment)
            if selected_district and selected_district in owner_set:
                selected_segments.append(segment)

        if boundary_segments:
            folium.PolyLine(
                locations=boundary_segments,
                color="#FFFFFF",
                weight=2.4,
                opacity=0.96,
                line_cap="round",
                line_join="round",
                interactive=False,
            ).add_to(m)
        if selected_segments:
            folium.PolyLine(
                locations=selected_segments,
                color="#F36A2E",
                weight=3.3,
                opacity=1,
                line_cap="round",
                line_join="round",
                interactive=False,
            ).add_to(m)

    # 별도 HTML 레이어를 추가하지 않고 Folium 영구 툴팁으로 단계에 맞는 지역명을
    # 표시한다. 동 단계에서는 DB에 좌표가 있는 행정동만 라벨을 올려 겹침을 줄인다.
    if showing_dong_map:
        for dong_name, point in point_by_dong_name.items():
            if dong_name not in visible_dong_names:
                continue
            is_selected = dong_name == selected_dong_name
            folium.CircleMarker(
                [point["lat"], point["lng"]],
                radius=1,
                color="transparent",
                weight=0,
                opacity=0,
                fill=False,
                interactive=False,
                tooltip=folium.Tooltip(
                    f"{dong_name} · 선택됨" if is_selected else dong_name,
                    permanent=True,
                    sticky=False,
                    direction="top" if is_selected else "center",
                    class_name=(
                        "selected-district-label" if is_selected else "dong-label"
                    ),
                    offset=(0, -7) if is_selected else (0, 0),
                ),
            ).add_to(m)
    else:
        for district, stats in district_stats.items():
            if district == "기타" or not stats["count"]:
                continue
            folium.CircleMarker(
                [stats["lat"] / stats["count"], stats["lng"] / stats["count"]],
                radius=1,
                color="transparent",
                weight=0,
                opacity=0,
                fill=False,
                interactive=False,
                tooltip=folium.Tooltip(
                    district,
                    permanent=True,
                    sticky=False,
                    direction="center",
                    class_name="district-label",
                    offset=(0, 0),
                ),
            ).add_to(m)

    # 초기에는 서울 전체 자치구, 선택 후에는 해당 구의 행정동 경계에 맞춰 확대한다.
    # 클릭 좌표 판정과 조회 카운팅은 기존 흐름을 그대로 유지한다.
    map_height = 620  # points가 없을 때(DB 미연결 등) 대비 기본값
    if points:
        if geojson_bounds:
            (lat_min, lng_min), (lat_max, lng_max) = geojson_bounds
            lats = [lat_min, lat_max]
            lngs = [lng_min, lng_max]
        elif city_hull_xy:
            lats = [lat for lng, lat in city_hull_xy]
            lngs = [lng for lng, lat in city_hull_xy]
        else:
            lats = [p["lat"] for p in points]
            lngs = [p["lng"] for p in points]
        lat_min, lat_max = min(lats), max(lats)
        lng_min, lng_max = min(lngs), max(lngs)
        m.fit_bounds([[lat_min, lng_min], [lat_max, lng_max]])

        # 그래도 여전히 서울 밖 지역이 꽤 보인다는 재피드백(2026-08-27: "상당히
        # 아쉽네 저거 빠져나오는거, 어떻게 잘 못 맞추나, 배치를 다르게 해도
        # 괜찮아") — 원인은 지도 div 자체의 가로세로 비율이 서울의 실제 모양과
        # 안 맞아서, Leaflet이 fit_bounds를 만족시키려고 한쪽 축을 필요 이상
        # zoom-out 시켰기 때문. 지도 높이를 서울 bbox의 실제 종횡비에 맞춰
        # 계산해서(_MAP_WIDTH는 고정) 두 축이 거의 동시에 딱 맞게 fit되도록 함.
        # 경도 1도의 실제 거리는 위도에 따라 cos(위도)만큼 줄어드므로, 그만큼
        # 보정해야 정확한 종횡비가 나온다.
        lat_span = max(lat_max - lat_min, 1e-6)
        lng_span = lng_max - lng_min
        center_lat = sum(lats) / len(lats)
        effective_lng_span = lng_span * math.cos(math.radians(center_lat))
        aspect = effective_lng_span / lat_span  # 가로/세로
        map_height = int(_MAP_WIDTH / max(aspect, 0.5))
        map_height = max(480, min(map_height, 900))  # 극단적으로 길쭉해지는 것 방지

    return m, map_height


# ---------------------------------------------------------------
# 우측 패널
# ---------------------------------------------------------------
_RANK_BADGES = ("1", "2", "3")


def _rank_card(badge: str, title: str, pill_text: str, caption: str,
               pill_bg: str = "#e8f5e9", pill_color: str = "#2e7d32") -> None:
    """순위 카드 하나(뱃지 + 제목 + 색상 알약 + 캡션) — 동 랭킹/업종 랭킹 둘 다
    이 카드 스타일을 그대로 재사용한다(2026-08-27, "지금 뜨는 동네"/"지금 뜨는
    업종" 두 섹션 공용)."""
    with st.container(border=True):
        st.markdown(
            f"""<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                 <div style="display:flex;align-items:center;gap:10px;min-width:0;">
                   <span style="display:inline-flex;align-items:center;justify-content:center;
                                width:28px;height:28px;border-radius:8px;background:#179B3B;
                                color:#fff;font-weight:800;font-size:0.88rem;line-height:1;">{badge}</span>
                   <span style="font-weight:700;font-size:1.05rem;">{title}</span>
                 </div>
                 <div style="flex-shrink:0;background:{pill_bg};color:{pill_color};
                             font-weight:700;padding:4px 14px;border-radius:999px;
                             font-size:0.9rem;white-space:nowrap;">
                   {pill_text}
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )
        st.caption(caption)

def _render_dong_search_box():
    """동 이름으로 바로 검색해서 지역상세 패널로 이동하는 검색창(2026-08-28 추가,
    같은 날 헤더 줄로 위치 이동). 상단 헤더의 브랜드/로그인 사이에 끼워 넣을
    거라 서브헤더 텍스트 없이 selectbox 하나만 컴팩트하게 둔다. 선택하는 순간
    바로 지역상세로 넘어가서(별도 "보기" 버튼 없음) 한 번의 조작으로 끝난다."""
    names = _dong_name_map()
    points = {p["dong_code"]: p for p in _dong_survival_proxy()}
    if not names:
        return

    options = sorted(names.items(), key=lambda x: x[1])
    codes = [c for c, _ in options]
    labels = [n for _, n in options]

    idx = st.selectbox(
        "동 이름으로 검색",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
        index=None,
        placeholder="동 이름으로 검색",
        key="dong_search_select",
        label_visibility="collapsed",
        width=280,
    )
    if idx is not None:
        code = codes[idx]
        point = points.get(code)
        if point:
            new_click = {"lat": point["lat"], "lng": point["lng"]}
            if st.session_state.get("region_click") != new_click:
                st.session_state["region_click"] = new_click
                st.rerun()


def _reset_region_selection() -> None:
    """전체 지도 복귀 시 검색 선택도 함께 비워 즉시 재진입하는 것을 막는다."""
    st.session_state["region_click"] = None
    st.session_state["dong_search_select"] = None


def _render_hot_dong_panel():
    """"카드가 밋밋해서 볼품없다"(1차) → 순위 뱃지+알약(2차) → 4/5위 스타일 안맞음
    → 3위까지만+"지금 뜨는 업종" 추가(3차)를 거쳐, 비율만 있고 실제 모수가 없어서
    설득력이 떨어진다는 피드백(2026-08-28)에 따라 구체적 숫자(총 매장수, 서울
    평균 대비)를 캡션에 추가."""
    st.subheader("지금 뜨는 동네")
    st.caption("최근 3개월 신규 매장과 생존율을 함께 반영한 순위예요.")
    citywide_avg = _citywide_survival_avg()
    ranking = _hot_dong_ranking(top_n=3)
    if not ranking:
        st.caption("아직 집계할 데이터가 부족해요.")
    else:
        for i, r in enumerate(ranking):
            caption = (
                f"총 {r['total_stores']:.0f}곳 중 최근 3개월 신규 {r['new_stores']:.0f}곳 "
                f"· 생존율 {r['survival_ratio'] * 100:.0f}%"
            )
            if citywide_avg is not None:
                diff = r["survival_ratio"] * 100 - citywide_avg
                caption += f" (서울 평균 대비 {'+' if diff >= 0 else ''}{diff:.0f}%p)"
            _rank_card(
                badge=_RANK_BADGES[i],
                title=r["dong_name"],
                pill_text=f"▲ {r['new_ratio'] * 100:.0f}%",
                caption=caption,
            )

    st.subheader("지금 뜨는 업종")
    keywords = _hot_keyword_ranking(top_n=3)
    if not keywords:
        st.caption("아직 집계할 데이터가 부족해요.")
    else:
        for i, k in enumerate(keywords):
            growth = k["growth_rate"]
            if growth is None:
                pill_text, pill_bg, pill_color = "—", "#eeeeee", "#616161"
            elif growth >= 0:
                pill_text, pill_bg, pill_color = f"{growth * 100:+.0f}%", "#e8f5e9", "#2e7d32"
            else:
                pill_text, pill_bg, pill_color = f"{growth * 100:+.0f}%", "#ffebee", "#c62828"

            first_count = k.get("first_store_count")
            if first_count and first_count != k["store_count"]:
                caption = f"매장수 {first_count}개 → {k['store_count']}개"
            else:
                caption = f"매장수 {k['store_count']}개"

            _rank_card(
                badge=_RANK_BADGES[i],
                title=k["keyword"],
                pill_text=pill_text,
                caption=caption,
                pill_bg=pill_bg,
                pill_color=pill_color,
            )


def _render_new_member_panel():
    """예비창업자(NEW_MEMBER) 전용 패널 — 로그인 안 한 GUEST와 화면이 완전히
    똑같다는 피드백(2026-08-28) 반영. 아직 가게가 없어 OWNER처럼 "내 가게 기준"
    분석은 못 주지만, "관심 업종을 고르면 그 업종이 잘 되는 동네를 보여주는"
    예비창업자 맞춤 탐색 기능을 추가해서 GUEST와 차별화한다."""
    st.subheader("관심 업종으로 동네 찾기")
    options = _industry_options()
    if not options:
        st.caption("업종 데이터를 불러올 수 없어요.")
    else:
        labels = [name for _, name in options]
        codes = [code for code, _ in options]
        idx = st.selectbox(
            "관심 업종을 선택해보세요", range(len(labels)), format_func=lambda i: labels[i]
        )
        selected_code = codes[idx]
        top_dongs = _top_dongs_for_industry(selected_code, top_n=3)
        if not top_dongs:
            st.caption("이 업종은 아직 데이터가 부족해요.")
        else:
            for i, d in enumerate(top_dongs):
                caption = f"{labels[idx]} 매장 {d['n']:.0f}곳 기준"
                if d["low_confidence"]:
                    caption = f"⚠️ {caption} — 표본 적어 참고만"
                _rank_card(
                    badge=_RANK_BADGES[i],
                    title=d["dong_name"],
                    pill_text=f"{d['wilson_score'] * 100:.0f}점",
                    caption=caption,
                )

    st.divider()
    _render_hot_dong_panel()


def _render_owner_panel(user: dict, snapshot: dict | None):
    if snapshot is None:
        st.warning("내 가게 정보를 찾을 수 없어요.")
        return

    st.subheader("우리 가게 현황")
    pop = _population_feature(snapshot["dong_code"])
    if pop:
        with st.container(border=True):
            st.markdown("**우리 동네 유동인구**")
            st.caption(
                f"내국인 {pop['korean_pop']:.0f}명 · 장기체류 외국인 {pop['foreign_long_pop']:.0f}명 "
                f"· 단기체류 외국인 {pop['foreign_short_pop']:.0f}명"
            )
            if pop["tourist_zone_candidate"]:
                st.caption("관광 특수 지역으로 분류돼요(단기체류 외국인 비중 상위권).")

    # 반경 300m 내 경쟁 밀도(2026-08-28 추가) — spatial_density_features는 이미
    # 스키마에 있던 피처인데 owner 패널에서 안 쓰고 있었음.
    density = _owner_competition_density(user["store_id"], snapshot["snapshot_date"])
    if density:
        with st.container(border=True):
            st.markdown("**주변 경쟁 강도 (반경 300m)**")
            ratio = (
                density["same_industry_count_300m"] / density["total_count_300m"] * 100
                if density["total_count_300m"] else 0
            )
            st.caption(
                f"동일업종 {density['same_industry_count_300m']}곳 / 전체 {density['total_count_300m']}곳 "
                f"(동일업종 비중 {ratio:.0f}%)"
            )
            # nearest_same_industry_distance_m은 동일업종이 자기뿐이면 NULL
            # (schema.sql 주석 참고).
            nearest_dist = density.get("nearest_same_industry_distance_m")
            if nearest_dist is not None:
                st.caption(f"가장 가까운 동일업종 매장까지 {float(nearest_dist):.0f}m")
            else:
                st.caption("반경 내 동일업종 경쟁점이 없어요.")
            if density["coord_cluster_size"] and density["coord_cluster_size"] >= 3:
                st.caption(f"복합상가·건물 추정 — 반경 20m 내 매장 {density['coord_cluster_size']}곳")

    pred = _latest_owner_prediction(user["store_id"])
    if pred:
        score = ui.proba_to_survival_score(float(pred["score"]))
        # 동 평균/전체 분포 대비 비교(2026-08-28 추가) — GUEST 지도용으로 이미
        # 캐싱된 _dong_survival_proxy를 재사용, DB를 더 안 때림.
        dong_scores = [p["survival_rate"] * 100 for p in _dong_survival_proxy()]
        dong_avg = next(
            (p["survival_rate"] * 100 for p in _dong_survival_proxy() if p["dong_code"] == snapshot["dong_code"]),
            None,
        )
        shap_lines = None
        raw_shap = pred.get("shap_top_features")
        if raw_shap:
            # TODO(ui-logic.md 3-1): feature명 -> 자연어 문장 매핑 테이블 아직 없음 —
            # 지금은 원본 feature명을 그대로 보여준다.
            feats = json.loads(raw_shap) if isinstance(raw_shap, str) else raw_shap
            shap_lines = [
                f"{f['feature']} ({'유리하게 작용' if f['shap_value'] > 0 else '불리하게 작용'})"
                for f in feats
            ]
        ui.score_card("내 업종 생존점수", score, shap_lines=shap_lines)

        # 기저 폐업률이 10.6%로 낮아 절대 점수가 구조적으로 80~95점대에 몰리는
        # 문제(2026-08-28 지적) — ui-logic.md 3-2에서 정해둔 "절대 점수보다 상대
        # 순위 병기" 방향을 반영해 서울 전체 동 분포 대비 퍼센타일을 같이 보여줌.
        rank, total = _rank_and_total_desc(score, dong_scores)
        if rank is not None:
            st.caption(f"서울 {total}개 동 중 {rank}위")
    else:
        # 모델이 아직 앱에 연동되지 않아 predictions가 비어있는 동안은 안내 문구 없이
        # 공란으로 둔다(2026-08-28 사용자 요청: "shap이나 모델 데이터로 못하는건
        # 공란으로 해줘 어차피 들어가니까") — 모델 연동되면 pred가 채워지면서 위
        # score_card 분기로 자동 전환됨.
        pass

    st.subheader("업종전환 추천")
    names = _industry_name_map()
    current_name = names.get(snapshot["industry_code"], snapshot["industry_code"])
    st.caption(f"현재 업종: {current_name}")
    recs = _industry_switch_recommendations(snapshot["industry_code"], top_n=3)
    if recs:
        for r in recs:
            switch_score = round(r["wilson_score"] * 100)
            caveat = f"표본 {r['sample_size']}건 기준"
            if r["low_confidence"]:
                caveat = f"⚠️ 표본 {r['sample_size']}건 — 신뢰도 낮음, 참고만 하세요"
            if r["transition_count"] > 0:
                caveat += f" · 실제 전환 사례 {r['transition_count']}건"
            ui.score_card(
                title=r["industry_name"],
                survival_score=switch_score,
                extra_caveat=caveat,
            )
        ui.short_term_switch_caveat()
    else:
        st.info("이 업종에 대한 전환 통계가 아직 없어요.")


def _render_region_detail_panel(clicked: dict):
    dong_code = _nearest_dong(clicked["lat"], clicked["lng"])
    if dong_code is None:
        st.caption("해당 위치의 동을 찾지 못했어요.")
        return

    names = _dong_name_map()
    st.caption("선택한 지역")
    st.subheader(names.get(dong_code, dong_code))

    # 지도 호버 툴팁에 있던 "우수 · 생존율 xx%" 등급 표시를 여기로 옮김(사용자
    # 요청, 2026-08-27 — 위 _build_map 주석 참고). _nearest_dong과 같은 A-1
    # 최근접 중심점 기준 데이터(_dong_survival_proxy)에서 이 동의 값을 찾는다.
    match = next((p for p in _dong_survival_proxy() if p["dong_code"] == dong_code), None)
    score = None
    rank = None
    total = None
    if match:
        score = round(match["survival_rate"] * 100)
        ui.grade_badge(score)
        ui.confidence_notice()

        all_dong_scores = [p["survival_rate"] * 100 for p in _dong_survival_proxy()]
        rank, total = _rank_and_total_desc(score, all_dong_scores)

    # 유동인구 — 평균 한 줄이 아니라 내국인/장기·단기체류 외국인 구성을 나눠서
    # 보여주고(2026-08-28, "너무 아쉽네 살짝 더 잘 보여질 수 있을거 같은데" 피드백),
    # 관광특구 후보 플래그도 같이 노출.
    pop = _population_feature(dong_code)

    if match:
        average_population = float(pop["total_pop_avg"]) if pop else None
        population_value = (
            f"{average_population / 10000:.1f}만"
            if average_population is not None and average_population >= 10000
            else f"{average_population:,.0f}명"
            if average_population is not None
            else "-"
        )
        with st.container(horizontal=True, gap="xsmall"):
            st.metric("상권 점수", f"{score}점", border=True)
            st.metric(
                "서울 순위",
                f"{rank}위" if rank is not None else "-",
                help=f"서울 {total}개 동 기준" if total is not None else None,
                border=True,
            )
            st.metric("일평균 유동인구", population_value, border=True)

    if pop:
        with st.container(border=True):
            st.markdown("**유동인구**")
            st.caption(
                f"내국인 {pop['korean_pop']:.0f}명 · 장기체류 외국인 {pop['foreign_long_pop']:.0f}명 "
                f"· 단기체류 외국인 {pop['foreign_short_pop']:.0f}명 (평균 {pop['total_pop_avg']:.0f}명)"
            )
            if pop["tourist_zone_candidate"]:
                st.caption("관광 특수 지역으로 분류돼요(단기체류 외국인 비중 상위권).")

    # 이 동네 실제 업종 구성 — store_snapshots 실측 데이터라 모델 없이 바로
    # 보여줄 수 있음(2026-08-28 추가). 업종별 생존점수 랭킹(ui-logic.md 4번, 모델
    # 배치추론 필요)과는 별개로, "여기 뭐가 많은지"를 보여주는 참고 정보.
    top_industries = _dong_top_industries(dong_code, top_n=3)
    if top_industries:
        with st.container(border=True):
            st.markdown("**이 동네 주요 업종**")
            for ind in top_industries:
                st.caption(f"{ind['industry_name']} · {ind['n']:.0f}곳")

    # 업종별 생존점수 랭킹(ui-logic.md 4번)은 전체 업종 배치 추론(모델)이 필요해서
    # 아직 못 채움 — 안내 문구 없이 공란으로 둔다(2026-08-28 사용자 요청: "shap이나
    # 모델 데이터로 못하는건 공란으로 해줘 어차피 들어가니까").

        # 동네 비교(2026-08-28 추가) — 세션에 "비교 목록"을 담아두고 최대 3곳까지
    # 나란히 표를 보여준다. 새 데이터/모델 없이 이미 조회 중인 값만 재사용.
    st.divider()
    if "compare_dongs" not in st.session_state:
        st.session_state["compare_dongs"] = []

    compare_list = st.session_state["compare_dongs"]
    already_added = dong_code in compare_list
    col_add, col_clear = st.columns([2, 1])
    with col_add:
        if not already_added and len(compare_list) < 3:
            if st.button(
                "비교 목록에 담기",
                key="add_compare",
                icon=":material/add:",
                width="stretch",
            ):
                compare_list.append(dong_code)
                st.rerun()
        elif already_added:
            st.caption("이미 비교 목록에 있어요.")
        else:
            st.caption("비교는 최대 3곳까지 가능해요.")
    with col_clear:
        if compare_list and st.button(
            "비우기",
            key="clear_compare",
            icon=":material/delete:",
            width="stretch",
        ):
            st.session_state["compare_dongs"] = []
            st.rerun()

    if compare_list:
        st.markdown("**동네 비교**")
        names = _dong_name_map()
        cols = st.columns(len(compare_list))
        for col, code in zip(cols, compare_list):
            stats = _dong_compare_stats(code)
            with col:
                st.markdown(f"**{names.get(code, code)}**")
                if stats and stats["survival_score"] is not None:
                    st.metric("생존점수", f"{stats['survival_score']}점")
                if stats and stats["total_pop_avg"] is not None:
                    st.caption(f"유동인구 {stats['total_pop_avg']:.0f}명")


# ---------------------------------------------------------------
# 레이아웃(사이드바 제거 + 상단 헤더) — 사용자가 첨부한 목업 4장처럼
# "사이드바 빼고 저 사진 마냥 배치"해달라는 요청(2026-08-28) 반영.
#
# 주의: Streamlit 기본 헤더 툴바([data-testid="stHeader"], Deploy 메뉴가 있는
# 그 흰 바)는 position:absolute + z-index:999990짜리 오버레이라서, 우리가
# 만든 상단 헤더 행이 페이지 맨 위(0~60px)에 있으면 그 뒤에 가려서 안 보인다
# (버튼은 DOM엔 있는데 화면엔 안 뜨는 상태 — 실제로 스크린샷 찍어서 좌표까지
# 확인함). 그래서 기본 헤더를 아예 display:none으로 죽이고 우리 헤더로
# 대체한다.
# ---------------------------------------------------------------
def _inject_layout_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stSidebarCollapseButton"] { display: none; }
        [data-testid="stHeader"] { display: none; }
        .stApp { background: #F7F8FA; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1480px;
            margin: 0 auto;
            padding: 1.25rem 1.5rem 2.5rem;
        }

        .st-key-top_header {
            background: #FFFFFF;
            border: 1px solid #E3E7EC;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(29, 56, 89, 0.06);
            padding: 0.9rem 1.15rem;
            margin-bottom: 1.25rem;
        }
        .st-key-brand_block [data-testid="stPageLink-NavLink"] {
            border: 0;
            background: transparent;
            box-shadow: none;
            color: #15171A;
            font-size: 1.35rem;
            font-weight: 800;
            padding: 0;
            min-height: auto;
        }
        .st-key-brand_block [data-testid="stPageLink-NavLink"] span:first-child {
            color: #2376D8;
        }
        .st-key-brand_block [data-testid="stCaptionContainer"] {
            color: #7A828C;
            margin-top: -0.15rem;
        }
        .st-key-header_actions [data-testid="stPageLink-NavLink"] {
            border-radius: 999px;
            border-color: #2376D8;
            background: #2376D8;
            color: #FFFFFF;
            font-weight: 700;
            padding-inline: 1rem;
            min-height: 2.65rem;
        }
        .st-key-header_actions button {
            min-height: 2.65rem;
            font-weight: 700;
        }
        .st-key-header_actions [data-baseweb="select"] > div {
            border-radius: 999px;
            background: #FFFFFF;
            min-height: 2.65rem;
        }

        .st-key-map_card,
        .st-key-insight_panel {
            background: #FFFFFF;
            border-color: #E0E5EB;
            border-radius: 18px;
            box-shadow: 0 6px 22px rgba(29, 56, 89, 0.06);
            padding: 1.2rem 1.25rem 1.25rem;
        }
        .st-key-map_card iframe {
            border-radius: 14px;
            border: 1px solid #E0E5EB;
        }
        .st-key-map_card [data-testid="stCaptionContainer"] {
            color: #7A828C;
        }
        .st-key-insight_panel [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #E3E7EC;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(29, 56, 89, 0.04);
        }
        .st-key-insight_panel h3 {
            letter-spacing: -0.02em;
        }

        .st-key-login_cta {
            background: #174E91;
            border-color: #174E91;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-top: 0.5rem;
        }
        .st-key-login_cta h4,
        .st-key-login_cta [data-testid="stCaptionContainer"] {
            color: #FFFFFF;
        }
        .st-key-login_cta [data-testid="stCaptionContainer"] {
            opacity: 0.78;
        }
        .st-key-login_cta [data-testid="stPageLink-NavLink"] {
            width: fit-content;
            border: 0;
            border-radius: 999px;
            background: #FFFFFF;
            color: #174E91;
            font-weight: 800;
            padding-inline: 1rem;
        }
        .st-key-onboarding_cta {
            background: #EAF3FF;
            border-color: #CFE2FB;
            border-radius: 16px;
            padding: 0.85rem 1rem;
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-inline: 0.8rem;
            }
            .st-key-top_header {
                padding: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )





@st.cache_data
def _brand_logo_base64() -> str | None:
    """로고(아이콘+텍스트 통짜 이미지)를 클릭 가능한 <a> 태그 안에 넣으려면
    st.image가 아니라 <img> 태그로 직접 그려야 해서(2026-08-28, "누르면 메인으로
    이동하게" 요청) base64로 인코딩. 코드로 아이콘+텍스트를 따로 그리던 방식은
    "텍스트 섞지 말고" 요청으로 통짜 로고 이미지로 교체."""
    if not _BRAND_LOGO_EXISTS:
        return None
    with open(_BRAND_LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _render_top_header(mode: str, user: dict | None, owner_snapshot: dict | None, show_search: bool):
    """시안처럼 브랜드는 왼쪽, 검색과 계정 액션은 오른쪽에 한 줄로 배치한다."""
    with st.container(key="top_header"):
        col_brand, col_actions = st.columns([5, 5], gap="large", vertical_alignment="center")

        with col_brand:
            with st.container(key="brand_block", gap=None):
                logo_b64 = _brand_logo_base64()
                if logo_b64:
                    st.markdown(
                        f"""
                        <a href="/" target="_self" style="display:inline-block;">
                            <img src="data:image/png;base64,{logo_b64}" style="height:42px; width:auto;">
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.page_link(
                        "app.py",
                        label="서울 상권 폐업예측",
                        icon=":material/location_city:",
                        width="content",
                    )
                st.caption("데이터로 먼저 보는 우리 동네 상권")

        with col_actions:
            with st.container(
                key="header_actions",
                horizontal=True,
                wrap=True,
                horizontal_alignment="right",
                vertical_alignment="center",
                gap="xsmall",
            ):
                if show_search:
                    _render_dong_search_box()

                if mode == "GUEST":
                    st.page_link(
                        "pages/login.py",
                        label="로그인",
                        icon=":material/login:",
                        width="content",
                    )
                else:
                    label = {
                        "owner": "기존점주",
                        "founder": "예비창업자",
                        "admin": "관리자",
                    }.get(user["user_type"], user["user_type"])
                    display_name = user["login_id"]
                    if (
                        user["user_type"] == "owner"
                        and owner_snapshot
                        and owner_snapshot.get("store_name")
                    ):
                        display_name = owner_snapshot["store_name"]
                    st.caption(f"{label} · {display_name}")

                    if mode == "ADMIN":
                        st.page_link(
                            "pages/admin_dashboard.py",
                            label="관리자 대시보드",
                            icon=":material/dashboard:",
                            width="content",
                        )
                    elif mode in ("OWNER", "NEW_MEMBER") and _MY_PAGE_EXISTS:
                        st.page_link(
                            "pages/mypage.py",
                            label="마이페이지",
                            icon=":material/person:",
                            width="content",
                        )

                    if st.button(
                        "로그아웃",
                        key="top_logout",
                        icon=":material/logout:",
                        width="content",
                    ):
                        auth.logout()
                        st.rerun()


# ---------------------------------------------------------------
# 메인
# ---------------------------------------------------------------
def main():
    _inject_layout_css()
    mode = auth.get_screen_mode()
    user = auth.current_user()
    owner_snapshot = _owner_latest_snapshot(user["store_id"]) if mode == "OWNER" else None

    _render_top_header(mode, user, owner_snapshot, show_search=mode in ("GUEST", "NEW_MEMBER"))

    if mode == "ADMIN":
        st.info("관리자 계정으로 로그인하셨어요. 위에서 관리자 대시보드로 이동해주세요.")
        return

    if "region_click" not in st.session_state:
        st.session_state["region_click"] = None

    col_map, col_panel = st.columns([65, 35], gap="medium")

    with col_map:
        with st.container(border=True, key="map_card"):
            st.caption("서울시 상권 데이터")
            st.subheader("내 가게 위치" if mode == "OWNER" else "우리 동네 상권 지도")
            st.caption("지도에서 관심 지역을 선택하면 오른쪽에서 상세 지표를 확인할 수 있어요.")

            # 지도 계산과 클릭 처리 로직은 그대로 두고 카드 안으로만 옮긴다.
            fmap, map_height = _build_map(mode, owner_snapshot, st.session_state["region_click"])
            map_state = st_folium(fmap, height=map_height, width=_MAP_WIDTH, key="main_map")
        if mode != "OWNER":
            if st.session_state["region_click"]:
                st.caption("행정동 단위 보기 · 선택한 동은 주황색으로 표시돼요.")
            else:
                st.caption(
                    "자치구 단위 보기 · 옅을수록 폐업위험이 낮고 진할수록 높아요."
                )
            if mode != "OWNER" and map_state and map_state.get("last_clicked"):
                st.session_state["region_click"] = map_state["last_clicked"]
                clicked_dong = _nearest_dong(
                    map_state["last_clicked"]["lat"], map_state["last_clicked"]["lng"]
                )
                if clicked_dong:
                    increment_dong_view(clicked_dong, user["user_type"] if user else None)
                    increment_user_view(user["user_id"] if user else None, clicked_dong)
                st.rerun()

    with col_panel:
        with st.container(border=True, key="insight_panel"):
            clicked = st.session_state["region_click"]
            if mode == "OWNER":
                _render_owner_panel(user, owner_snapshot)
            elif clicked:
                st.button(
                    "전체 지도로 돌아가기",
                    key="back_to_map",
                    icon=":material/arrow_back:",
                    type="tertiary",
                    on_click=_reset_region_selection,
                )
                _render_region_detail_panel(clicked)
            elif mode == "NEW_MEMBER":
                ui.onboarding_banner()
                _render_new_member_panel()
            else:
                _render_hot_dong_panel()
                ui.login_cta_banner()


if __name__ == "__main__":
    main()
