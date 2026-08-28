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
from scipy.spatial import Voronoi
from sqlalchemy import text
from streamlit_folium import st_folium

from shared import auth
from shared import components as ui
from shared.db import get_engine

st.set_page_config(page_title="서울 상권 폐업예측", layout="wide")


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
    인기 순위 뭐 이런것도 해놨으니까"(2026-08-27, 사용자 요청)에 따라 채워 넣음."""
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
                SELECT keyword, store_count, growth_rate
                FROM trend_keywords
                WHERE snapshot_date = :latest
                ORDER BY growth_rate DESC
                LIMIT :n
                """
            ),
            {"latest": latest, "n": top_n},
        ).mappings().all()
    return [dict(r) for r in rows]


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

    names = _dong_name_map()
    scored = []
    for r in rows:
        new_ratio = r["new_stores"] / r["total_stores"]
        survival_ratio = 1 - (r["closed_stores"] / r["total_stores"])
        # growth_slope는 아직 빠져있음(모듈 docstring 참고) — 두 지표 단순 평균.
        hot_index = (new_ratio + survival_ratio) / 2
        scored.append(
            {
                "dong_code": r["dong_code"],
                "dong_name": names.get(r["dong_code"], r["dong_code"]),
                "new_ratio": new_ratio,
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
        SELECT lat, lng, dong_code, industry_code, store_name
        FROM store_snapshots WHERE store_id = :store_id
        ORDER BY snapshot_date DESC LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"store_id": store_id}).mappings().first()
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


def _latest_owner_prediction(store_id: str) -> dict | None:
    """predictions 캐시 조회만 한다 (ui-logic.md 3번). 캐시가 없을 때 그 자리에서
    모델을 호출해 새로 INSERT하는 부분은 모델이 아직 앱에 연동되지 않아 비워둠 —
    TODO: 모델 로딩 + app/shared/write_prediction.log_prediction 호출 추가."""
    engine = get_engine()
    if engine is None:
        return None
    sql = text(
        """
        SELECT score, shap_top_features FROM predictions
        WHERE query_type = 'existing_store' AND store_id = :store_id
        ORDER BY created_at DESC LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"store_id": store_id}).mappings().first()
    return dict(row) if row else None


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


def _build_map(mode: str, owner_snapshot: dict | None, clicked: dict | None):
    # 타일은 다시 일반 OpenStreetMap으로 — cartodbpositron(파스텔 배경) + 꽉 채운
    # 경계선 색칠을 같이 쓰니 실제 지도가 아니라 "벌집"처럼 보인다는 피드백(2026-08-27:
    # "벌집이야? 그냥 일반 지도로 하고 나누는선만 예쁘게 해줘"). 그래서 배경은 원래
    # 익숙한 실제 지도로 되돌리고, 그 위에 경계선만 살짝 얹는 방식으로 바꿈 — 안은
    # 거의 비워서(fill_opacity를 확 낮춤) 지도 자체(도로/건물/지명)가 그대로 비치고,
    # 선(테두리)만 위험도 색으로 또렷하게 보이게 함. 흔히 보는 "지도 위에 위험지역
    # 윤곽만 표시"하는 방식.
    #
    # 반환값은 (folium.Map, 지도 높이) 튜플 — 아래에서 계산하는 높이를
    # main()에서 st_folium(height=...)에 그대로 넘겨준다.
    if mode == "OWNER" and owner_snapshot:
        lat, lng = float(owner_snapshot["lat"]), float(owner_snapshot["lng"])
        m = folium.Map(location=[lat, lng], zoom_start=16, tiles="OpenStreetMap")
        folium.Marker(
            [lat, lng],
            tooltip=owner_snapshot.get("store_name") or "내 가게",
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(m)
        return m, 620  # 한 지점 확대라 종횡비 문제가 없어서 기존 고정 높이 유지

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="OpenStreetMap")

    # 클릭했을 때 실제 셀 모양이 아니라 네모난 점선 상자가 뜨는 문제(2026-08-27,
    # 사용자 지적: "저걸 누르면 저 격자 부분이 눌려야지 왜 저렇게 된거야") — 브라우저가
    # 클릭된 SVG 도형에 기본으로 씌우는 포커스 사각형(도형 모양을 무시하고 항상
    # 네모)이 원인. 그 기본 사각형은 꺼버리고, 선택된 셀 자체를 우리가 직접 진하게
    # 칠해서(아래 SELECTED_COLOR) 그게 "눌린 상태" 표시가 되도록 함.
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-interactive:focus{outline:none;}</style>"
    ))

    points = _dong_survival_proxy()
    # 호버 툴팁에 있던 "우수 · 생존율 xx%" 표시는 지역상세 패널로 옮기고, 지도
    # 위 툴팁은 구+동 이름만 보여주기로 함(사용자 요청, 2026-08-27: "마우스
    # 올려놨을때 우수 생존율 이거는 저 사진에다 하는게 좋을거 같고, 그냥 구랑
    # 동이름 노출되게끔"). 등급/생존율은 클릭 후 _render_region_detail_panel에서 표시.
    names = _dong_name_map()

    # A-1과 동일한 최근접 중심점 기준으로, 지금 선택돼 있는 동을 구해서 그 셀만
    # 다르게 칠한다. `clicked`는 st.session_state["region_click"]에서 오는데, 이
    # 값은 새로 클릭하기 전까지 리런이 몇 번 되든 그대로 남아있어서(사용자 요청:
    # "포커스가 안 풀리게") 선택 표시가 저절로 유지된다 — 브라우저 자체 포커스처럼
    # 지도가 다시 그려지면 없어지는 방식이 아니라, 매번 이 값 기준으로 다시 칠하는
    # 방식이라 리런에 안전함.
    selected_dong = _nearest_dong(clicked["lat"], clicked["lng"]) if clicked else None

    DEFAULT_COLOR = "#7fc7e8"   # 연한 하늘색
    SELECTED_COLOR = "#1976a8"  # 눌렀을 때 살짝 진한 하늘색
    selected_latlngs = None     # 선택된 셀 좌표 — 아래서 그 영역으로 확대할 때 씀
    for latlngs, p in _dong_boundary_cells(points):
        is_selected = selected_dong is not None and p["dong_code"] == selected_dong
        color = SELECTED_COLOR if is_selected else DEFAULT_COLOR
        folium.Polygon(
            locations=latlngs,
            color=color,
            weight=2.5 if is_selected else 1.2,
            # 평소(선택 안 됐을 때)엔 구분선이 거의 안 보인다는 피드백(2026-08-27:
            # "안 눌렀을 때 구분선 잘 안 보여서 살짝만 보이게") — 0.45 → 0.65로 조금
            # 더 뚜렷하게. 그래도 선택된 셀보단 확실히 옅게(0.9) 남겨서 "살짝만".
            opacity=0.9 if is_selected else 0.65,
            fill=True,
            fill_color=color,
            fill_opacity=0.28 if is_selected else 0.08,
            smooth_factor=2,
            line_cap="round",
            line_join="round",
            tooltip=names.get(p["dong_code"], p["dong_code"]),
        ).add_to(m)
        if is_selected:
            selected_latlngs = latlngs

    # 클릭하면 그 셀 영역으로 확대(사용자 요청: "이걸 누르면 저 사진처럼 확대되는
    # 그런건 어렵나"). 셀의 좌표 범위(bounding box)를 구해서 지도가 딱 그 영역만
    # 보이게 자동으로 이동/확대한다 — 별도 지도 클릭 이벤트 없이, "선택된 셀이
    # 바뀌면 그 셀에 맞춰 다시 그린다"는 지금 구조 그대로 자연스럽게 들어맞음.
    #
    # 선택된 셀이 없을 때(초기 화면)는 고정 zoom_start=11만 쓰면 서울시 범위를 훨씬
    # 넘어서 고양시/구리시/하남시/성남시/인천 등 서비스와 무관한 지역까지 넓게
    # 보여서 "이거 안 맞는거 너무 불편하다"는 피드백(2026-08-27, 스크린샷) — 서울
    # 상권 서비스인데 지도가 서울 범위에 안 맞게 너무 넓게 잡혀있었음. 그래서 동
    # 중심점(points, 전부 서울 소속) 전체의 좌표 범위로 fit_bounds해서, 처음 화면부터
    # 서울 영역에 딱 맞게 자동으로 줌/이동되도록 함.
    map_height = 620  # points가 없을 때(DB 미연결 등) 대비 기본값
    if selected_latlngs:
        lats = [pt[0] for pt in selected_latlngs]
        lngs = [pt[1] for pt in selected_latlngs]
        m.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    elif points:
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
_RANK_BADGES = ("🥇", "🥈", "🥉")


def _rank_card(badge: str, title: str, pill_text: str, caption: str,
               pill_bg: str = "#e8f5e9", pill_color: str = "#2e7d32") -> None:
    """순위 카드 하나(뱃지 + 제목 + 색상 알약 + 캡션) — 동 랭킹/업종 랭킹 둘 다
    이 카드 스타일을 그대로 재사용한다(2026-08-27, "지금 뜨는 동네"/"지금 뜨는
    업종" 두 섹션 공용)."""
    with st.container(border=True):
        st.markdown(
            f"""<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                 <div style="display:flex;align-items:center;gap:10px;min-width:0;">
                   <span style="font-size:1.4rem;line-height:1;">{badge}</span>
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


def _render_hot_dong_panel():
    """"카드가 밋밋해서 볼품없다"(1차) → 순위 뱃지(🥇🥈🥉)+초록 알약으로 꾸몄더니
    이번엔 "4위/5위는 뱃지 없이 텍스트뿐이라 안 맞는다"는 피드백(2026-08-27,
    스크린샷) — 뱃지 있는 카드(1~3위)와 없는 카드(4~5위)가 한 목록에 섞여서
    스타일이 안 맞아 보였음. 그래서 "지금 뜨는 동네"는 뱃지가 자연스러운 3위까지만
    보여주고, 그 아래 빈 공간은 사용자 제안대로("밑에는 좀 다른걸 채우는건 어떤지,
    인기 순위 뭐 이런것도 해놨으니까") 이미 admin 대시보드에 만들어둔 trend_keywords
    기반 "지금 뜨는 업종" 랭킹을 같은 카드 스타일로 재사용해서 채움(_hot_keyword_ranking)."""
    st.subheader("지금 뜨는 동네")
    ranking = _hot_dong_ranking(top_n=3)
    if not ranking:
        st.caption("아직 집계할 데이터가 부족해요.")
    else:
        for i, r in enumerate(ranking):
            _rank_card(
                badge=_RANK_BADGES[i],
                title=r["dong_name"],
                pill_text=f"▲ {r['new_ratio'] * 100:.0f}%",
                caption="최근 3개월 신규매장 비율",
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
            _rank_card(
                badge=_RANK_BADGES[i],
                title=k["keyword"],
                pill_text=pill_text,
                caption=f"매장수 {k['store_count']}개",
                pill_bg=pill_bg,
                pill_color=pill_color,
            )


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

    pred = _latest_owner_prediction(user["store_id"])
    if pred:
        score = ui.proba_to_survival_score(float(pred["score"]))
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
    else:
        st.info("아직 우리 가게 분석 결과가 없어요. (모델 연동 후 자동으로 계산돼요)")

    st.subheader("업종전환 추천")
    st.caption("준비 중이에요 — industry_survival_stats 기반 추천 로직은 다음 단계에서 붙일 예정이에요.")
    ui.short_term_switch_caveat()


def _render_region_detail_panel(clicked: dict):
    st.subheader("지역 상세")
    dong_code = _nearest_dong(clicked["lat"], clicked["lng"])
    if dong_code is None:
        st.caption("해당 위치의 동을 찾지 못했어요.")
        return

    names = _dong_name_map()
    st.markdown(f"**{names.get(dong_code, dong_code)}**")

    # 지도 호버 툴팁에 있던 "우수 · 생존율 xx%" 등급 표시를 여기로 옮김(사용자
    # 요청, 2026-08-27 — 위 _build_map 주석 참고). _nearest_dong과 같은 A-1
    # 최근접 중심점 기준 데이터(_dong_survival_proxy)에서 이 동의 값을 찾는다.
    match = next((p for p in _dong_survival_proxy() if p["dong_code"] == dong_code), None)
    if match:
        score = round(match["survival_rate"] * 100)
        ui.grade_badge(score)
        ui.confidence_notice()

    pop = _population_feature(dong_code)
    if pop:
        st.caption(f"유동인구 평균 {pop['total_pop_avg']:.0f}명")

    st.info(
        "업종별 생존점수 랭킹은 아직 준비 중이에요. "
        "(전체 업종 배치 추론이 필요 — ui-logic.md 4번 참고)"
    )


# ---------------------------------------------------------------
# 사이드바 로그인
# ---------------------------------------------------------------
def _render_sidebar_auth():
    """app.py는 메인 화면(지도+패널)만 보여준다 — 로그인/회원가입 입력 폼은
    pages/login.py로 분리했음(사용자 요청, 2026-08-27). 여기서는 로그인
    상태 표시 + 이동 링크만 담당한다."""
    with st.sidebar:
        if auth.is_logged_in():
            user = auth.current_user()
            label = {"owner": "기존점주", "founder": "예비창업자", "admin": "관리자"}.get(
                user["user_type"], user["user_type"]
            )
            st.success(f"{label}로 로그인됨 ({user['login_id']})")
            st.page_link("pages/my_page.py", label="마이페이지")
            if st.button("로그아웃"):
                auth.logout()
                st.rerun()
            if user["user_type"] == "admin":
                st.page_link("pages/admin_dashboard.py", label="관리자 대시보드로 이동")
        else:
            st.page_link("pages/login.py", label="로그인 / 회원가입")


# ---------------------------------------------------------------
# 메인
# ---------------------------------------------------------------
def main():
    _render_sidebar_auth()
    mode = auth.get_screen_mode()

    st.title("서울 상권 폐업예측")

    if mode == "ADMIN":
        st.info("관리자 계정으로 로그인하셨어요. 좌측에서 관리자 대시보드로 이동해주세요.")
        st.page_link("pages/admin_dashboard.py", label="관리자 대시보드 열기", icon="➡️")
        return

    user = auth.current_user()
    owner_snapshot = _owner_latest_snapshot(user["store_id"]) if mode == "OWNER" else None

    if "region_click" not in st.session_state:
        st.session_state["region_click"] = None

    col_map, col_panel = st.columns([6, 4])

    with col_map:
        # width를 None(컬럼 폭에 맞춤)으로 두면 _build_map이 실제 렌더 폭을 몰라서
        # 높이를 서울 모양에 맞게 계산할 수가 없었음 — 그래서 폭을 _MAP_WIDTH로
        # 고정하고, 높이는 _build_map이 서울 bbox 종횡비에 맞춰 계산해서 돌려주는
        # 값을 그대로 씀(2026-08-27, "저거 빠져나오는거" 재피드백 반영 — 위
        # _build_map 주석 참고).
        fmap, map_height = _build_map(mode, owner_snapshot, st.session_state["region_click"])
        map_state = st_folium(fmap, height=map_height, width=_MAP_WIDTH, key="main_map")
        if mode != "OWNER" and map_state and map_state.get("last_clicked"):
            st.session_state["region_click"] = map_state["last_clicked"]
            st.rerun()

    with col_panel:
        if mode == "GUEST":
            ui.login_cta_banner()
        elif mode == "NEW_MEMBER":
            ui.onboarding_banner()

        clicked = st.session_state["region_click"]
        if mode == "OWNER":
            _render_owner_panel(user, owner_snapshot)
        elif clicked:
            _render_region_detail_panel(clicked)
        else:
            _render_hot_dong_panel()


if __name__ == "__main__":
    main()