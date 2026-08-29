"""
마이페이지 (My Page) - Streamlit UI
ERD에 존재하는 테이블(users, stores, store_snapshots, predictions, administrative_dongs,
industries, user_view_history)로 실제 DB 연동해서 구성했습니다.
- 내 정보           ← users (실제 로그인 세션은 shared/auth.py 재사용)
- 내 매장 현황       ← stores + administrative_dongs (owner만 해당)
- 내 매장 스냅샷 추이 ← store_snapshots (owner만 해당)
- 내 분석 히스토리   ← predictions (user_id로 필터)
- 최근 관심있게 본 지역 ← user_view_history + population_features + stores 집계
  (지도 클릭 이력에 유동인구/매장수/폐업률을 붙임, owner/founder 공통)
- 자주 본 지역 TOP 3 ← user_view_history (누적 조회수 기준 정렬, owner/founder 공통)
- 지금 뜨는 업종 트렌드 ← trend_keywords (founder 전용, 창업 참고 정보)

DB 연결이 없거나(로컬 .env 미설정) 아직 쌓인 데이터가 없는 섹션은 화면 흐름 확인용
더미 데이터로 폴백합니다 (각 폴백 지점에 TODO 주석 참고).
"""

import importlib.util
import sys
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

# login.py/app.py와 동일한 이유(진입점 app/app.py의 파일명이 "app"이라 `app.shared...`
# 형태의 import가 깨짐)로, app/ 폴더 자체를 sys.path에 넣고 "shared"를 최상위 이름으로
# 바로 가져온다. shared.auth처럼 상대 import(from .db import ...)를 쓰는 모듈은 이 방식
# (진짜 패키지 import)이어야 정상 동작하므로, db.py 전용 importlib 우회와는 별도로 둔다.
_APP_DIR = str(Path(__file__).resolve().parents[1])
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

from shared import auth  # noqa: E402

# app.shared.db를 일반 `from app.shared.db import ...`로 가져오면 깨진다: 이 프로젝트의
# 진입점이 app/app.py라서 Streamlit이 sys.path에 app/ 디렉터리를 넣는데, 그 안에 있는
# app.py 파일과 이름이 겹쳐서 "app" 패키지 해석 자체가 실패한다 (repo 루트를 sys.path에
# 추가해도 app/app.py가 항상 먼저 매칭됨). 그래서 db.py를 파일 경로로 직접 로드해서 우회한다.
_DB_MODULE_PATH = Path(__file__).resolve().parents[1] / "shared" / "db.py"


@st.cache_resource
def _load_db_module():
    spec = importlib.util.spec_from_file_location("_mypage_db_pmh", _DB_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_store_name(store_id: str) -> str | None:
    """store_snapshots.store_name(최신 스냅샷)에서 실제 상호명을 조회. DB 연결이 없거나
    해당 store_id가 없으면 None을 반환하므로 호출부에서 더미 데이터로 폴백하면 된다."""
    if not store_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT store_name FROM store_snapshots "
        "WHERE store_id = :store_id ORDER BY snapshot_date DESC LIMIT 1"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"store_id": store_id}).first()
    return row[0] if row else None


def get_favorite_regions(user_id: str, limit: int = 3) -> list[dict] | None:
    """predictions(user_id=본인) → stores → administrative_dongs 집계로 실제 조회 지역
    TOP N을 가져온다. DB 연결이 없으면 None(호출부에서 폴백 판단), 연결은 됐지만
    조회 이력이 없으면 빈 리스트를 반환한다."""
    if not user_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT d.dong_name, COUNT(*) AS cnt "
        "FROM predictions p "
        "JOIN stores s ON p.store_id = s.store_id "
        "JOIN administrative_dongs d ON s.dong_code = d.dong_code "
        "WHERE p.user_id = :user_id "
        "GROUP BY d.dong_name "
        "ORDER BY cnt DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "limit": limit}).all()
    return [{"dong_name": r[0], "count": r[1]} for r in rows]


def get_my_recent_views(user_id: str, limit: int = 5) -> list[dict] | None:
    """지도에서 이 사용자가 클릭해서 본 동 목록을 최근 조회순으로 가져온다.
    (user_id, dong_code)마다 1행씩 누적되는 user_view_history 테이블 기준
    (app/shared/write_user_view.py의 increment_user_view()가 클릭마다 채워줌).
    동 이름은 administrative_dongs를 조인해서 바로 붙인다 — app.py의
    _dong_name_map()은 진입점 파일 이름 충돌 때문에 이 페이지에서 가져다 쓸 수
    없어서(모듈 상단 주석 참고), admin_dashboard.py의 인기 조회지역 섹션과 같은
    방식으로 이 파일 안에서 직접 조인한다.
    단순히 동 이름만 나열하면 판단 근거가 없어서, population_features(유동인구)와
    stores 집계(매장수/폐업률)를 같이 붙여 "이 동네가 어떤 곳인지" 바로 보이게 한다.
    DB 연결이 없으면 None(호출부에서 폴백 판단), 연결은 됐지만 조회 이력이
    없으면(또는 아직 테이블이 없으면) 빈 리스트를 반환한다.
    """
    if not user_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text
    from sqlalchemy.exc import DatabaseError

    sql = text(
        "SELECT d.dong_name, d.gu_name, v.view_count, v.last_viewed_at, "
        "p.total_pop_avg, "
        "COALESCE(sc.total_stores, 0) AS total_stores, "
        "COALESCE(sc.closed_stores, 0) AS closed_stores "
        "FROM user_view_history v "
        "JOIN administrative_dongs d ON d.dong_code = v.dong_code "
        "LEFT JOIN population_features p ON p.dong_code = v.dong_code "
        "LEFT JOIN ("
        "    SELECT dong_code, COUNT(*) AS total_stores, SUM(is_closed) AS closed_stores "
        "    FROM stores GROUP BY dong_code"
        ") sc ON sc.dong_code = v.dong_code "
        "WHERE v.user_id = :user_id "
        "ORDER BY v.last_viewed_at DESC "
        "LIMIT :limit"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"user_id": user_id, "limit": limit}).mappings().all()
    except DatabaseError:
        # user_view_history가 아직 스키마에 없는 경우(마이그레이션 전)도 에러 대신 빈 목록으로.
        return []

    # DB 서버(TiDB Cloud) 시스템 시간대가 UTC라서 last_viewed_at이 NOW()로 UTC 기준
    # 저장돼 있다 — 한국 시간(KST, UTC+9)으로 보정해서 돌려준다.
    result = []
    for row in rows:
        d = dict(row)
        if d.get("last_viewed_at"):
            d["last_viewed_at"] = d["last_viewed_at"] + timedelta(hours=9)
        d["total_pop_avg"] = float(d["total_pop_avg"]) if d["total_pop_avg"] is not None else None
        total_stores = d["total_stores"] or 0
        closed_stores = d["closed_stores"] or 0
        d["total_stores"] = total_stores
        d["closure_rate"] = (closed_stores / total_stores) if total_stores else None
        result.append(d)
    return result


def get_top_viewed_regions(user_id: str, limit: int = 3) -> list[dict] | None:
    """user_view_history를 view_count(누적 조회 횟수) 기준으로 정렬해 "자주 본 지역"
    TOP N을 가져온다. get_my_recent_views()와 같은 테이블이지만 정렬 기준만 다르다
    (최근순 대신 누적 조회수순) — 반복해서 찾아본 지역이라 진짜 관심사에 더 가깝다.
    DB 연결이 없으면 None, 연결은 됐지만 조회 이력이 없으면 빈 리스트."""
    if not user_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text
    from sqlalchemy.exc import DatabaseError

    sql = text(
        "SELECT d.dong_code, d.dong_name, d.gu_name, v.view_count "
        "FROM user_view_history v "
        "JOIN administrative_dongs d ON d.dong_code = v.dong_code "
        "WHERE v.user_id = :user_id "
        "ORDER BY v.view_count DESC "
        "LIMIT :limit"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"user_id": user_id, "limit": limit}).mappings().all()
    except DatabaseError:
        return []
    return [dict(row) for row in rows]


@st.cache_data(ttl=3600)
def get_dong_center_point(dong_code: str) -> dict | None:
    """동의 대표 좌표(store_snapshots 평균 lat/lng)를 구한다. app.py의
    _dong_survival_proxy()/_render_dong_search_box()가 쓰는 것과 완전히 동일한
    방식(동별 매장 좌표 평균)이라, 여기서 만든 {"lat":..., "lng":...}를
    st.session_state["region_click"]에 그대로 넣으면 app.py가 지도를 그 동으로
    확대해서 보여준다 — app.py는 파일명 충돌 때문에 이 페이지에서 직접 import할
    수 없어서(모듈 상단 주석 참고) 같은 쿼리를 독립적으로 둔다."""
    if not dong_code:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text("SELECT AVG(lat) AS lat, AVG(lng) AS lng FROM store_snapshots WHERE dong_code = :dong_code")
    with engine.connect() as conn:
        row = conn.execute(sql, {"dong_code": dong_code}).mappings().first()
    if not row or row["lat"] is None:
        return None
    return {"lat": float(row["lat"]), "lng": float(row["lng"])}


@st.cache_data(ttl=300)  # 5분 캐시: 유동인구/키워드 순위처럼 개인화 안 된 전체 집계라 부하를 줄인다
def get_trend_keywords_for_founder(limit: int = 5) -> list[dict]:
    """trend_keywords에서 최신 snapshot_date 기준 store_count 상위 N개.
    admin_dashboard.py의 get_trend_keywords()와 같은 로직이지만, 이 페이지는
    관리자 화면 모듈을 import하지 않는 구조라(각 페이지가 독립적으로 shared/db만
    가져다 씀) 여기서도 동일한 쿼리를 자체적으로 둔다. 예비창업자에게 "요즘 뜨는
    업종/키워드"를 창업 참고 정보로 보여주기 위함 — DB 연결이 없거나 아직 적재된
    스냅샷이 없으면 빈 리스트(호출부가 데모 데이터로 폴백)."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return []

    from sqlalchemy import text

    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(snapshot_date) FROM trend_keywords")).scalar()
        if latest is None:
            return []
        rows = conn.execute(
            text(
                "SELECT keyword, store_count, growth_rate FROM trend_keywords "
                "WHERE snapshot_date = :latest "
                "ORDER BY store_count DESC "
                "LIMIT :limit"
            ),
            {"latest": latest, "limit": limit},
        ).mappings().all()
    return [
        {
            "keyword": r["keyword"],
            "store_count": r["store_count"],
            "growth_rate": float(r["growth_rate"]) if r["growth_rate"] is not None else None,
        }
        for r in rows
    ]


def get_industry_name_map() -> dict:
    """industries.industry_code -> industry_name 매핑. 매장 현황/스냅샷/분석 히스토리
    세 군데에서 공통으로 쓰여서 조인 대신 한 번 조회해 재사용한다 (app.py의
    _industry_name_map()과 동일한 발상). DB 연결이 없으면 빈 dict."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return {}

    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT industry_code, industry_name FROM industries")).mappings().all()
    return {r["industry_code"]: r["industry_name"] for r in rows}


@st.cache_data(ttl=3600)
def get_industry_options() -> list[tuple[str, str]]:
    """가게 등록 폼의 업종 선택 드롭다운용 — industries 테이블 그대로."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return []

    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT industry_code, industry_name FROM industries ORDER BY industry_name")
        ).mappings().all()
    return [(r["industry_code"], r["industry_name"]) for r in rows]


@st.cache_data(ttl=3600)
def get_dong_options() -> list[tuple[str, str]]:
    """가게 등록 폼의 위치(동) 선택 드롭다운용 — administrative_dongs 그대로."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return []

    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT dong_code, dong_name, gu_name FROM administrative_dongs ORDER BY gu_name, dong_name")
        ).mappings().all()
    return [(r["dong_code"], f"{r['gu_name']} {r['dong_name']}") for r in rows]


_FLOOR_CATEGORIES = ["1층", "2층이상", "지하", "기타"]


def register_new_store(
    user_id: str, store_name: str, industry_code: str, dong_code: str, floor_category: str
) -> str | None:
    """예비창업자가 마이페이지에서 직접 매장을 등록한다. 스키마 변경 없이 기존
    stores/store_snapshots/users 테이블만 사용한다:
      1. stores에 새 매장 1건 추가 (store_id는 이 페이지에서 새로 발급)
      2. store_snapshots에 이번 달 스냅샷 1건 추가 (상호명/좌표/층 등 실측 정보)
      3. users.store_id를 새로 등록한 매장으로 연결

    user_type은 일부러 안 건드린다 — "owner"로 바꾸려면 shared/auth.py의 로그인
    규칙(로그인 아이디 = store_id, 고정 비밀번호 "1234")까지 맞춰야 해서 로그인
    자격 자체가 바뀌어버린다(범위 밖인 auth.py를 건드려야 함). 대신 이 페이지의
    매장 섹션 노출 조건을 "store_id 존재 여부"로만 판단하도록 바꿔서, user_type이
    그대로 founder여도 등록한 매장 정보가 바로 보이게 한다.

    좌표는 실제 지오코딩이 없어서, app.py가 지도 중심점으로 쓰는 것과 동일한 방식
    (해당 동 기존 매장들의 평균 lat/lng, get_dong_center_point())으로 근사한다.
    """
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    point = get_dong_center_point(dong_code)
    if point is None:
        return None

    from sqlalchemy import text

    store_id = f"SELF{uuid.uuid4().hex[:20].upper()}"
    snapshot_month = date.today().strftime("%Y%m")

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO stores (store_id, current_industry_code, dong_code, "
                "first_seen_snapshot, last_seen_snapshot, n_snapshots_observed, "
                "is_closed, had_temporary_gap) "
                "VALUES (:store_id, :industry_code, :dong_code, :snapshot, :snapshot, "
                "1, FALSE, FALSE)"
            ),
            {
                "store_id": store_id, "industry_code": industry_code,
                "dong_code": dong_code, "snapshot": snapshot_month,
            },
        )
        conn.execute(
            text(
                "INSERT INTO store_snapshots (store_id, snapshot_date, industry_code, "
                "dong_code, store_name, floor_category, lng, lat, is_closed_next, "
                "transitioned_next, label_available) "
                "VALUES (:store_id, :snapshot, :industry_code, :dong_code, :store_name, "
                ":floor_category, :lng, :lat, FALSE, FALSE, FALSE)"
            ),
            {
                "store_id": store_id, "snapshot": snapshot_month, "industry_code": industry_code,
                "dong_code": dong_code, "store_name": store_name,
                "floor_category": floor_category, "lng": point["lng"], "lat": point["lat"],
            },
        )
        conn.execute(
            text("UPDATE users SET store_id = :store_id WHERE user_id = :user_id"),
            {"store_id": store_id, "user_id": user_id},
        )

    return store_id


def get_store_status(store_id: str) -> dict | None:
    """stores + administrative_dongs 조인으로 매장 현황을 가져온다. DB 연결이 없거나
    store_id가 없거나 조회된 행이 없으면 None(호출부에서 폴백 판단)."""
    if not store_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT s.current_industry_code, d.dong_code, d.dong_name, d.gu_name, "
        "s.is_closed, s.first_seen_snapshot, s.last_seen_snapshot "
        "FROM stores s "
        "JOIN administrative_dongs d ON d.dong_code = s.dong_code "
        "WHERE s.store_id = :store_id"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"store_id": store_id}).mappings().first()
    if row is None:
        return None

    result = dict(row)
    names = get_industry_name_map()
    result["industry_name"] = names.get(result["current_industry_code"], result["current_industry_code"])
    return result


def get_store_snapshots(store_id: str, limit: int = 5) -> list[dict] | None:
    """store_snapshots를 최근 순으로 가져온다. transitioned_next는 스키마상 불리언(다음
    스냅샷에서 업종이 바뀌었는지 여부)이라, LEAD 윈도함수로 바로 다음 스냅샷의
    industry_code를 같이 가져와서 "무슨 업종으로 전환됐는지"까지 보여줄 수 있게 한다.
    LEAD는 WHERE로 걸러진 이 매장의 전체 스냅샷을 snapshot_date 오름차순으로 훑고 난
    뒤에 바깥쪽 ORDER BY DESC/LIMIT이 적용되므로, limit를 걸어도 다음 업종 매칭 자체는
    깨지지 않는다. DB 연결이 없으면 None, 연결은 됐지만 스냅샷이 없으면 빈 리스트."""
    if not store_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT snapshot_date, industry_code, is_closed_next, transitioned_next, "
        "LEAD(industry_code) OVER (ORDER BY snapshot_date) AS next_industry_code "
        "FROM store_snapshots "
        "WHERE store_id = :store_id "
        "ORDER BY snapshot_date DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"store_id": store_id, "limit": limit}).mappings().all()

    names = get_industry_name_map()
    result = []
    for r in rows:
        d = dict(r)
        d["industry_name"] = names.get(d["industry_code"], d["industry_code"])
        d["next_industry_name"] = names.get(d["next_industry_code"], d["next_industry_code"])
        result.append(d)
    return result


_QUERY_TYPE_LABELS = {"existing_store": "폐업 예측", "new_location": "신규 입지 분석"}


def get_prediction_history(user_id: str, limit: int = 10) -> list[dict] | None:
    """predictions을 user_id로 필터해서 최근 조회순(created_at DESC)으로 가져온다.
    store_name은 store_snapshots(최신 스냅샷)에서, industry_name은 industries에서
    매핑한다. DB 연결이 없으면 None, 연결은 됐지만 조회 이력이 없으면 빈 리스트."""
    if not user_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text
    import json

    sql = text(
        "SELECT p.query_type, p.store_id, p.industry_code, p.score, p.shap_top_features, "
        "(SELECT ss.store_name FROM store_snapshots ss WHERE ss.store_id = p.store_id "
        " ORDER BY ss.snapshot_date DESC LIMIT 1) AS store_name "
        "FROM predictions p "
        "WHERE p.user_id = :user_id "
        "ORDER BY p.created_at DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "limit": limit}).mappings().all()

    names = get_industry_name_map()
    result = []
    for r in rows:
        d = dict(r)
        d["query_type"] = _QUERY_TYPE_LABELS.get(d["query_type"], d["query_type"])
        d["industry_name"] = names.get(d["industry_code"], d["industry_code"])
        d["score"] = float(d["score"])
        raw_shap = d.get("shap_top_features")
        d["shap_top_features"] = json.loads(raw_shap) if isinstance(raw_shap, str) else raw_shap
        result.append(d)
    return result


def _format_snapshot_date(snapshot: str) -> str:
    """'YYYYMM' -> 'YYYY-MM' (더미 데이터도 실제 스키마 형식(VARCHAR(6))에 맞춰뒀다)."""
    if not snapshot or len(snapshot) != 6:
        return snapshot
    return f"{snapshot[:4]}-{snapshot[4:]}"


# ────────────────────────────────────────────────
# 페이지 기본 설정
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="마이페이지 | 상권분석",
    page_icon="🗂️",
    layout="wide",
)

# 상단 로고 (지도 핀 아이콘 + "서울 상권분석" 워드마크)를 base64 SVG로 인라인 삽입.
# TODO: 팀에서 최종 로고 이미지 파일을 확정하면 이 상수 대신 그 파일을 base64 인코딩해서 교체.
_LOGO_SVG_B64 = (
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzNjAgMTAwIiB3aWR0aD0iMzYwIiBo"
    "ZWlnaHQ9IjEwMCI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InBpbkdyYWQiIHgxPSIwJSIgeTE9IjAlIiB4Mj0i"
    "MCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzNENkZEOSIvPgogICAgICA8c3RvcCBv"
    "ZmZzZXQ9IjUwJSIgc3RvcC1jb2xvcj0iI0Q4NDU0QSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiM3"
    "QjhCM0IiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxwYXRoCiAgICBkPSJNNDYgNEMyNS42IDQgOSAyMC42"
    "IDkgNDFjMCAzMCAzNyA1NSAzNyA1NXMzNy0yNSAzNy01NUM4MyAyMC42IDY2LjQgNCA0NiA0eiIKICAgIGZpbGw9InVybCgjcGlu"
    "R3JhZCkiCiAgLz4KICA8dGV4dCB4PSIxMDQiIHk9IjUyIiBmb250LWZhbWlseT0iJ01hbGd1biBHb3RoaWMnLCdBcHBsZSBTRCBH"
    "b3RoaWMgTmVvJyxzYW5zLXNlcmlmIgogICAgICAgIGZvbnQtc2l6ZT0iMzQiIGZvbnQtd2VpZ2h0PSI4MDAiIGZpbGw9IiMxMTEx"
    "MTEiPuyEnOyauCDsg4HqtozrtoTshJ08L3RleHQ+CiAgPHRleHQgeD0iMTA2IiB5PSI3NiIgZm9udC1mYW1pbHk9IkFyaWFsLHNh"
    "bnMtc2VyaWYiCiAgICAgICAgZm9udC1zaXplPSIxMyIgZmlsbD0iIzhhOGE4YSI+U2VvdWwgQ29tbWVyY2lhbCBBbmFseXNpczwv"
    "dGV4dD4KPC9zdmc+Cg=="
)


# ────────────────────────────────────────────────
# 더미 데이터: 내 매장 (DB 연결이 없거나 이 store_id로 조회된 행이 없을 때의 폴백)
# 계정당 매장 1개 (users.store_id가 단일 FK). 상권이 여러 개인 점주는 계정을 따로 관리한다는
# 전제라서 매장 선택 UI는 두지 않는다. 실제 스키마 형식(VARCHAR(6) 'YYYYMM')에 맞춰뒀다.
# ────────────────────────────────────────────────
DUMMY_STORE_NAME = "카페 온기"

DUMMY_STORE_STATUS = {
    "current_industry_code": "CS200001",
    "industry_name": "카페",
    "dong_code": "1120510800",
    "dong_name": "역삼동",
    "gu_name": "강남구",
    "is_closed": False,
    "first_seen_snapshot": "202503",
    "last_seen_snapshot": "202608",
}

DUMMY_STORE_SNAPSHOTS = [
    {
        "snapshot_date": "202608",
        "industry_code": "CS200001",
        "industry_name": "카페",
        "is_closed_next": False,
        "transitioned_next": False,
        "next_industry_code": None,
        "next_industry_name": None,
    },
    {
        "snapshot_date": "202605",
        "industry_code": "CS200001",
        "industry_name": "카페",
        "is_closed_next": False,
        "transitioned_next": False,
        "next_industry_code": None,
        "next_industry_name": None,
    },
    {
        "snapshot_date": "202511",
        "industry_code": "CS100001",
        "industry_name": "음식점",
        "is_closed_next": False,
        "transitioned_next": True,
        "next_industry_code": "CS200001",
        "next_industry_name": "카페",
    },
]


# 프로필 이미지를 대신하는 아바타 5종. TODO: 실제 프로필 이미지 업로드 기능이 생기면 대체.
AVATAR_EMOJIS = ["🧑‍💼", "👩‍💻", "🧑‍🎨", "👨‍🍳", "🧕"]


def _pick_avatar(user_id: str) -> str:
    """user_id 기준으로 고정된 아바타를 골라, 리런할 때마다 안 바뀌고 사람마다 다르게 보이게 한다."""
    idx = sum(ord(c) for c in user_id) % len(AVATAR_EMOJIS)
    return AVATAR_EMOJIS[idx]


def _risk_level(score: float) -> tuple[str, str]:
    """0~1 사이 점수를 (위험도 라벨, st.badge 색상)으로 변환."""
    if score >= 0.6:
        return "높음", "red"
    if score >= 0.3:
        return "보통", "orange"
    return "낮음", "blue"


def _prediction_target_label(pred: dict) -> str:
    """예측 대상을 사람이 읽을 수 있는 이름으로. store_id 노출 대신 상호명/신규 입지를 우선한다."""
    if pred.get("store_name"):
        return pred["store_name"]
    if not pred.get("store_id"):
        return "신규 입지"
    return pred["store_id"]


# ────────────────────────────────────────────────
# 스타일
# ────────────────────────────────────────────────
def inject_custom_css():
    st.markdown(
        """
        <style>
        .st-key-header_profile {
            flex: 1;
            margin: 0 !important;
        }
        .st-key-header_card {
            padding: 1.5rem !important;
        }
        .st-key-header_card .st-key-pw_btn {
            margin: 0 !important;
        }
        .st-key-header_card [data-testid="stMarkdownContainer"] {
            margin: 0 !important;
        }
        .st-key-top_logo {
            margin-top: -3rem;
            margin-bottom: -0.5rem;
            width: fit-content;
        }
        .mp-logo-link {
            display: inline-block;
            line-height: 0;
        }
        .mp-logo-link img {
            height: 56px;
        }
        .st-key-page_title {
            margin-top: 1.5rem;
            margin-bottom: 2rem;
        }
        .st-key-page_title h1 {
            font-size: 1.5rem;
        }
        [data-testid="stHeading"] h3 {
            font-size: 1.25rem;
        }
        .mp-avatar {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6C63FF, #A78BFA);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        }
        .st-key-danger_zone {
            border-color: rgba(255, 75, 75, 0.35) !important;
            background-color: rgba(255, 75, 75, 0.05);
        }
        .st-key-withdraw_btn button {
            border-color: rgba(255, 75, 75, 0.5);
            color: #ff4b4b;
        }
        .st-key-withdraw_btn button:hover {
            border-color: #ff4b4b;
            color: #ff4b4b;
            background-color: rgba(255, 75, 75, 0.08);
        }
        .st-key-store_status div[data-testid="stMetric"] {
            background-color: rgba(127, 127, 127, 0.08);
            border-radius: 0.5rem;
            padding: 0.5rem 0.75rem;
        }
        .st-key-store_status div[data-testid="stMetricValue"] {
            font-size: 1.1rem;
        }
        .st-key-store_status div[data-testid="stMetricLabel"] {
            font-size: 0.75rem;
        }
        /* "자주 본 지역" 카드: header_profile과 같은 방식으로 정보 영역에 flex:1을 줘서
           옆에 놓인 "지도에서 보기" 버튼과 같은 줄에서 오른쪽으로 밀어낸다(2026-08-29). */
        [class*="st-key-top_viewed_info_"] {
            flex: 1;
            margin: 0 !important;
        }
        .mp-rank-badge {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6C63FF, #A78BFA);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .mp-mini-bar-track {
            width: 100%;
            height: 6px;
            border-radius: 999px;
            background: rgba(127, 127, 127, 0.15);
            overflow: hidden;
        }
        .mp-mini-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #6C63FF, #A78BFA);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────
# 세션 상태 초기화
# user_info는 shared/auth.py의 실제 로그인 세션(main()에서 st.session_state.user_info로
# 미리 채워둠)을 그대로 쓰고, 여기서는 그 아래 파생 데이터만 DB에서 채운다.
# ────────────────────────────────────────────────
def init_session_state():
    user = st.session_state.user_info
    store_id = user.get("store_id")

    if "display_name" not in st.session_state:
        # 헤더 인사말 등에 쓰는 이름. login_id 대신 실제 상호명(store_snapshots.store_name)을
        # DB에서 가져와 쓰고, DB 연결이 없거나 못 찾으면 더미 상호명 → login_id 순으로 폴백한다.
        st.session_state.display_name = (
            get_store_name(store_id)
            or (DUMMY_STORE_NAME if store_id else None)
            or user["login_id"]
        )

    if "predictions" not in st.session_state:
        # predictions 테이블 (FK user_id로 필터된 결과)
        # TODO(데모용 임시 폴백): DB 연결이 없거나 아직 조회 이력이 없으면 화면 확인용 더미로.
        db_predictions = get_prediction_history(user["user_id"])
        st.session_state.predictions = db_predictions or [
            {
                "query_type": "폐업 예측",
                "store_id": store_id,
                "store_name": DUMMY_STORE_NAME,
                "industry_code": "CS200001",
                "industry_name": "카페",
                "score": 0.23,
                "shap_top_features": [
                    {"feature": "매장 운영 개월수", "shap_value": -0.08},
                    {"feature": "반경 300m 동종업종 수", "shap_value": 0.05},
                ],
            },
            {
                "query_type": "신규 입지 분석",
                "store_id": None,
                "store_name": None,
                "industry_code": "CS100001",
                "industry_name": "분식점",
                "score": 0.67,
                "shap_top_features": [
                    {"feature": "유동인구", "shap_value": 0.15},
                    {"feature": "동일업종 밀집도", "shap_value": -0.10},
                ],
            },
        ]

    # favorite_regions/my_recent_views는 다른 페이지(메인 지도)에서의 클릭/분석으로
    # 계속 늘어나는 값이라, 다른 값들과 달리 세션당 1회 캐싱하면 안 된다 — 마이페이지를
    # 한 번 연 뒤 메인페이지로 돌아가 동을 더 클릭해도 session_state가 세션 내내 유지돼
    # 예전 값이 그대로 보이는 버그가 있었다(2026-08-28). 매 페이지 로드마다 새로 조회한다.

    # predictions(user_id=본인) → stores → administrative_dongs 집계 (관제실의 인기 조회
    # 지역과 같은 방식 + WHERE user_id=본인).
    # TODO(데모용 임시 폴백): DB 연결이 없거나 아직 조회 이력이 없으면 화면 확인용 더미로.
    db_regions = get_favorite_regions(user["user_id"])
    st.session_state.favorite_regions = db_regions or [
        {"dong_name": "역삼동", "count": 8},
        {"dong_name": "서교동", "count": 5},
        {"dong_name": "성수동", "count": 3},
    ]

    # user_view_history(지도 클릭 → 개인별 조회 이력) 기준 "최근 관심있게 본 지역"
    # (유동인구/매장수/폐업률까지 같이 붙여서 판단 근거를 보여준다).
    # TODO(데모용 임시 폴백): 지도 클릭 로직이 아직 increment_user_view()를 호출하지
    # 않으면 항상 빈 리스트가 오는데, 화면 확인을 위해 더미로 채워서 보여준다.
    db_views = get_my_recent_views(user["user_id"])
    st.session_state.my_recent_views = db_views or [
        {
            "dong_name": "역삼동", "gu_name": "강남구", "view_count": 4,
            "last_viewed_at": "2026-08-27 21:10",
            "total_pop_avg": 45000.0, "total_stores": 512, "closure_rate": 0.08,
        },
        {
            "dong_name": "합정동", "gu_name": "마포구", "view_count": 2,
            "last_viewed_at": "2026-08-26 14:32",
            "total_pop_avg": 32000.0, "total_stores": 398, "closure_rate": 0.11,
        },
    ]

    # 같은 user_view_history를 누적 조회수(view_count) 기준으로 다시 정렬한 "자주 본 지역".
    # TODO(데모용 임시 폴백): 위와 동일한 이유로 이력이 없으면 더미로.
    db_top_viewed = get_top_viewed_regions(user["user_id"])
    st.session_state.top_viewed_regions = db_top_viewed or [
        {"dong_code": "11680640", "dong_name": "역삼1동", "gu_name": "강남구", "view_count": 4},
        {"dong_code": "11440680", "dong_name": "합정동", "gu_name": "마포구", "view_count": 2},
    ]

    if user["user_type"] == "founder":
        # trend_keywords 기준 "지금 뜨는 업종 트렌드" — 예비창업자 창업 참고용.
        # TODO(데모용 임시 폴백): DB 연결이 없거나 아직 적재된 스냅샷이 없으면 더미로.
        st.session_state.founder_trend_keywords = get_trend_keywords_for_founder(limit=5) or [
            {"keyword": "무인카페", "store_count": 128, "growth_rate": 0.42},
            {"keyword": "반려동물동반", "store_count": 96, "growth_rate": 0.31},
            {"keyword": "회전초밥", "store_count": 87, "growth_rate": 0.12},
        ]

    if "store_status" not in st.session_state and store_id:
        # TODO(데모용 임시 폴백): DB 연결이 없거나 이 store_id로 조회된 행이 없으면 더미로.
        st.session_state.store_status = get_store_status(store_id) or DUMMY_STORE_STATUS

    if "store_snapshots" not in st.session_state and store_id:
        # TODO(데모용 임시 폴백): DB 연결이 없거나 스냅샷이 없으면 화면 확인용 더미로.
        st.session_state.store_snapshots = get_store_snapshots(store_id) or DUMMY_STORE_SNAPSHOTS


# ────────────────────────────────────────────────
# 0. 헤더 (프로필 요약 + 비밀번호 변경 진입점)
# ────────────────────────────────────────────────
def render_header():
    # border + 가로 배치 + 세로 중앙정렬을 컨테이너 하나에서 동시에 처리한다.
    # (border 컨테이너 안에 st.columns를 또 넣는 이중 중첩 구조는, 바깥 컨테이너와 그 안의
    # columns 행 높이가 서로 달라질 수 있어 정렬이 어긋나는 원인이 될 수 있다.)
    avatar = _pick_avatar(st.session_state.user_info["user_id"])

    with st.container(border=True, horizontal=True, vertical_alignment="center", key="header_card"):
        with st.container(key="header_profile"):
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.6rem;">
                    <div class="mp-avatar">{avatar}</div>
                    <div>
                        <div style="font-size:1.05rem; font-weight:600; line-height:1.3;">
                            {st.session_state.display_name}님, 환영합니다 👋
                        </div>
                        <div style="font-size:0.85rem; opacity:0.65; line-height:1.3;">
                            마이페이지에서 내 정보와 매장 현황을 확인할 수 있습니다.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("🔑 비밀번호 변경", key="pw_btn"):
            show_change_password_dialog()


# ────────────────────────────────────────────────
# 1. 내 정보 (users)
# ────────────────────────────────────────────────
def render_my_info():
    with st.container(border=True):
        st.subheader("👤 내 정보")

        user = st.session_state.user_info
        # owner는 login_id가 곧 store_id(auth.py 로그인 규칙)라서 굳이 따로 안 보여주고,
        # founder/admin은 store_id가 애초에 없다 — "아이디" 한 필드로 통일한다.
        _TYPE_LABELS = {"owner": "기존점주", "founder": "예비창업자", "admin": "관리자"}

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("아이디", value=user["login_id"], disabled=True)
        with col2:
            st.text_input(
                "사용자 유형",
                value=_TYPE_LABELS.get(user["user_type"], user["user_type"]),
                disabled=True,
            )
        if user.get("store_id"):
            st.caption(f"연결된 매장 코드(store_id): {user['store_id']}")


@st.dialog("🔑 비밀번호 변경")
def show_change_password_dialog():
    with st.form("change_password_form"):
        current_pw = st.text_input("현재 비밀번호", type="password")
        new_pw = st.text_input("새 비밀번호", type="password")
        new_pw_confirm = st.text_input("새 비밀번호 확인", type="password")
        submitted = st.form_submit_button("변경하기")

        if submitted:
            if new_pw != new_pw_confirm:
                st.error("새 비밀번호가 일치하지 않습니다.")
            elif not current_pw or not new_pw:
                st.error("모든 항목을 입력해주세요.")
            else:
                # TODO: password_hash 검증 및 갱신 로직 (DB 업데이트)
                st.success("비밀번호가 변경되었습니다.")


@st.dialog("⚠️ 회원 탈퇴")
def show_delete_account_dialog():
    st.error("탈퇴 시 계정 정보가 삭제되며 복구할 수 없습니다.")
    confirm = st.checkbox("안내 사항을 확인했습니다.")
    if st.button("회원 탈퇴", type="primary", disabled=not confirm):
        # TODO: 실제 탈퇴 처리 로직 (users 레코드 삭제/비활성화) — 지금은 DB는 그대로 두고
        # 화면상으로만 탈퇴된 것처럼 로그아웃 처리한다.
        st.success("탈퇴가 완료되었습니다.")
        time.sleep(1.2)
        auth.logout()
        st.switch_page("app.py")


# ────────────────────────────────────────────────
# 2. 내 매장 현황 (stores) - owner 전용
# ────────────────────────────────────────────────
def render_store_status():
    with st.container(border=True, key="store_status"):
        st.subheader(
            "🏪 내 매장 현황",
            help=(
                "stores 테이블: current_industry_code, dong_code, is_closed, first/last_seen_snapshot "
                "· 상호명은 store_snapshots.store_name(최신 스냅샷) 기준"
            ),
        )

        store = st.session_state.store_status

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("상호명", st.session_state.display_name)
        col2.metric("현재 업종", store["industry_name"])
        col3.metric("위치 (동)", f"{store['gu_name']} {store['dong_name']}")
        col4.metric("폐업 여부", "폐업" if store["is_closed"] else "영업 중")

        st.caption(
            f"📅 데이터 수집 기간: {_format_snapshot_date(store['first_seen_snapshot'])} "
            f"~ {_format_snapshot_date(store['last_seen_snapshot'])}"
        )


# ────────────────────────────────────────────────
# 3. 내 매장 스냅샷 추이 (store_snapshots) - owner 전용
# ────────────────────────────────────────────────
def render_store_snapshots():
    with st.container(border=True):
        st.subheader(
            "📈 내 매장 스냅샷 추이",
            help="store_snapshots 테이블: snapshot_date, industry_code, is_closed_next, transitioned_next",
        )

        snapshots = st.session_state.store_snapshots
        if not snapshots:
            st.info("아직 기록된 스냅샷이 없습니다.")
            return

        for snap in snapshots:
            with st.container(border=True):
                date_col, badge_col = st.columns([2, 5], vertical_alignment="center")
                with date_col:
                    st.markdown(f"**{_format_snapshot_date(snap['snapshot_date'])}**")
                with badge_col:
                    with st.container(horizontal=True):
                        st.badge(snap["industry_name"], icon="🏷️", color="blue")
                        if snap["is_closed_next"]:
                            st.badge("다음 폐업 예정", icon="⚠️", color="red")
                        else:
                            st.badge("정상 운영 유지", icon="✅", color="green")
                        if snap["transitioned_next"] and snap.get("next_industry_name"):
                            st.badge(f"→ {snap['next_industry_name']} 전환", icon="🔄", color="orange")


# ────────────────────────────────────────────────
# 4. 내 분석 히스토리 (predictions)
# ────────────────────────────────────────────────
def render_prediction_history():
    with st.container(border=True):
        st.subheader("📜 내 분석 히스토리")

        predictions = st.session_state.predictions
        if not predictions:
            st.info("아직 조회한 분석 기록이 없습니다.")
            return

        for i, pred in enumerate(predictions):
            with st.container(border=True):
                info_col, score_col, cta_col = st.columns(
                    [3, 2, 1], vertical_alignment="center"
                )
                with info_col:
                    st.markdown(f"**{pred['query_type']}**")
                    st.caption(f"{_prediction_target_label(pred)} · {pred.get('industry_name', pred['industry_code'])}")
                with score_col:
                    risk_label, risk_color = _risk_level(pred["score"])
                    st.badge(f"위험도 {risk_label}", color=risk_color)
                    st.progress(pred["score"], text=f"{pred['score']:.0%}")
                with cta_col:
                    with st.container(horizontal_alignment="right"):
                        if st.button("상세보기 →", key=f"pred_detail_{i}"):
                            show_prediction_detail_dialog(pred)


@st.dialog("📄 분석 상세")
def show_prediction_detail_dialog(pred: dict):
    st.markdown(f"**{pred['query_type']}**")
    st.caption(f"{_prediction_target_label(pred)} · {pred.get('industry_name', pred['industry_code'])}")

    risk_label, risk_color = _risk_level(pred["score"])
    st.badge(f"위험도 {risk_label}", color=risk_color)
    st.progress(pred["score"], text=f"{pred['score']:.0%}")

    shap_features = pred.get("shap_top_features")
    if shap_features:
        st.markdown("**주요 기여 요인 (SHAP)**")
        for feat in shap_features:
            st.write(f"- {feat['feature']}: {feat['shap_value']}")

    # TODO: 전용 상세 페이지(app/pages/prediction_detail.py 등)가 생기면
    # 이 다이얼로그 대신 st.switch_page로 이동시키는 편이 낫다.


# ────────────────────────────────────────────────
# 5. 내가 관심있는 지역 TOP 3 (owner 전용 - predictions 기반)
# ────────────────────────────────────────────────
def render_favorite_regions():
    with st.container(border=True):
        st.subheader("📍 내가 관심있는 지역 TOP 3")

        regions = st.session_state.favorite_regions
        if not regions:
            st.info("아직 조회 이력이 없습니다.")
        else:
            for rank, r in enumerate(regions, start=1):
                # Streamlit 컨테이너 키에 CSS를 거는 대신, 카드 테두리까지 통째로 직접 그린다.
                # (내부 DOM 구조를 추측해서 맞추는 방식이 신뢰할 수 없어서 자체 완결형으로 전환)
                st.markdown(
                    '<div style="'
                    'border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; '
                    'padding:1rem 1.25rem; margin-bottom:0.75rem; '
                    'display:flex; align-items:center; gap:1rem;">'
                    f'<div class="mp-rank-badge" style="flex-shrink:0;">{rank}</div>'
                    '<div style="flex:1; min-width:0;">'
                    '<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">'
                    f'{r["dong_name"]}</div>'
                    '<div style="font-size:0.8rem; opacity:0.65;">'
                    f'{r["count"]}회 조회</div>'
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )


# ────────────────────────────────────────────────
# 5-1. 최근 관심있게 본 지역 (owner/founder 공통 - user_view_history 기반, 메인 지도 클릭 집계)
# 단순 지역명 나열이 아니라 유동인구/매장수/폐업률을 같이 붙여서 판단 근거를 준다.
# ────────────────────────────────────────────────
def render_recent_views():
    with st.container(border=True):
        st.subheader("🕘 최근 관심있게 본 지역")

        views = st.session_state.my_recent_views
        if not views:
            st.info("아직 지도에서 조회한 지역이 없습니다.")
            return

        for v in views:
            location = f"{v.get('gu_name') or ''} {v.get('dong_name') or ''}".strip()

            detail_parts = []
            if v.get("total_pop_avg") is not None:
                detail_parts.append(f"유동인구 {v['total_pop_avg']:,.0f}명")
            if v.get("total_stores"):
                detail_parts.append(f"매장 {v['total_stores']:,}개")
            if v.get("closure_rate") is not None:
                detail_parts.append(f"폐업률 {v['closure_rate']:.0%}")
            detail_line = " · ".join(detail_parts)

            st.markdown(
                '<div style="'
                'border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; '
                'padding:1rem 1.25rem; margin-bottom:0.75rem;">'
                '<div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">'
                '<div style="font-weight:700; font-size:1.05rem;">'
                f"{location}</div>"
                '<div style="font-size:0.8rem; opacity:0.65; text-align:right;">'
                f"{v['view_count']}회 조회 · 최근 {v['last_viewed_at']}</div>"
                "</div>"
                + (
                    f'<div style="font-size:0.8rem; opacity:0.65; margin-top:0.4rem;">{detail_line}</div>'
                    if detail_line else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────
# 5-2. 자주 본 지역 TOP 3 (owner/founder 공통 - user_view_history를 누적 조회수로 정렬)
# ────────────────────────────────────────────────
def render_top_viewed_regions():
    with st.container(border=True):
        st.subheader("🔥 자주 본 지역 TOP 3")

        regions = st.session_state.top_viewed_regions
        if not regions:
            st.info("아직 지도에서 조회한 지역이 없습니다.")
            return

        # 카드를 눌러 app.py 지도로 바로 이동시키려면 진짜 st.button이 필요해서
        # (raw HTML로는 페이지 이동 클릭을 못 받음) 버튼을 텍스트 옆(같은 줄)에 둬야
        # 하는데, st.columns는 두 컬럼 높이가 다를 때 vertical_alignment가 안쪽 뱃지
        # 정렬까지 어긋나게 만드는 문제가 반복됐다(2026-08-29). 대신 render_header()의
        # header_profile과 완전히 같은 방식 — st.container(horizontal=True)로 한 줄에
        # 놓고, 정보 영역 컨테이너에 flex:1을 줘서(위 CSS) 버튼을 오른쪽 끝으로 밀어내는
        # 검증된 패턴을 재사용한다.
        for rank, r in enumerate(regions, start=1):
            location = f"{r.get('gu_name') or ''} {r.get('dong_name') or ''}".strip()
            with st.container(border=True, horizontal=True, vertical_alignment="center"):
                with st.container(key=f"top_viewed_info_{rank}"):
                    st.markdown(
                        '<div style="display:flex; align-items:center; gap:1rem;">'
                        f'<div class="mp-rank-badge" style="flex-shrink:0;">{rank}</div>'
                        '<div style="flex:1; min-width:0;">'
                        '<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">'
                        f'{location}</div>'
                        '<div style="font-size:0.8rem; opacity:0.65;">'
                        f'{r["view_count"]}회 조회</div>'
                        "</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                if st.button("🗺️ 지도에서 보기", key=f"top_viewed_goto_{rank}"):
                    point = get_dong_center_point(r.get("dong_code"))
                    if point:
                        st.session_state["region_click"] = point
                        st.switch_page("app.py")
                    else:
                        st.toast("이 지역의 좌표를 찾을 수 없어요.", icon="⚠️")


# ────────────────────────────────────────────────
# 5-3. 지금 뜨는 업종 트렌드 (founder 전용 - trend_keywords 기반, 창업 참고 정보)
# ────────────────────────────────────────────────
def render_trend_widget():
    with st.container(border=True):
        st.subheader("📈 지금 뜨는 업종 트렌드")
        st.caption("trend_keywords 테이블 기준 · 최신 집계 시점의 매장수 상위 키워드")

        keywords = st.session_state.founder_trend_keywords
        if not keywords:
            st.info("아직 집계된 트렌드 데이터가 없습니다.")
            return

        for rank, kw in enumerate(keywords, start=1):
            rate = kw.get("growth_rate")
            if rate is None:
                g_bg, g_fg, g_text = "rgba(139, 148, 158, 0.15)", "#8b949e", "변화 없음"
            elif rate >= 0:
                g_bg, g_fg, g_text = "rgba(46, 160, 67, 0.15)", "#2ea043", f"▲ {rate:.0%}"
            else:
                g_bg, g_fg, g_text = "rgba(248, 81, 73, 0.15)", "#f85149", f"▼ {abs(rate):.0%}"

            st.markdown(
                '<div style="'
                'border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; '
                'padding:1rem 1.25rem; margin-bottom:0.75rem; '
                'display:flex; align-items:center; gap:1rem;">'
                f'<div class="mp-rank-badge" style="flex-shrink:0;">{rank}</div>'
                '<div style="flex:1; min-width:0;">'
                '<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">'
                f'{kw["keyword"]}</div>'
                '<div style="font-size:0.8rem; opacity:0.65;">'
                f'{kw["store_count"]}개 매장</div>'
                "</div>"
                '<div style="flex-shrink:0;">'
                f'<span style="background:{g_bg}; color:{g_fg}; padding:0.15rem 0.55rem; '
                'border-radius:999px; font-size:0.8rem; font-weight:600; white-space:nowrap;">'
                f"{g_text}</span>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────
# 5-4. 가게 등록 (founder 전용 - stores/store_snapshots에 새 매장 추가)
# ────────────────────────────────────────────────
@st.dialog("🏪 가게 등록하기")
def show_store_registration_dialog():
    st.caption("등록하면 마이페이지에 내 매장 현황·스냅샷 추이가 바로 표시됩니다.")

    store_name = st.text_input("상호명", key="store_reg_name")

    industries = get_industry_options()
    industry_labels = [name for _, name in industries]
    industry_idx = st.selectbox(
        "업종", range(len(industry_labels)), format_func=lambda i: industry_labels[i],
        index=None, placeholder="업종을 선택해주세요", key="store_reg_industry",
    )

    dongs = get_dong_options()
    dong_labels = [name for _, name in dongs]
    dong_idx = st.selectbox(
        "위치 (동)", range(len(dong_labels)), format_func=lambda i: dong_labels[i],
        index=None, placeholder="동을 선택해주세요", key="store_reg_dong",
    )

    floor_category = st.selectbox("층 구분", _FLOOR_CATEGORIES, key="store_reg_floor")

    if st.button("✅ 등록하기", type="primary", key="store_reg_submit"):
        if not store_name or industry_idx is None or dong_idx is None:
            st.error("상호명, 업종, 위치를 모두 입력해주세요.")
            return

        user = st.session_state.user_info
        store_id = register_new_store(
            user_id=user["user_id"],
            store_name=store_name,
            industry_code=industries[industry_idx][0],
            dong_code=dongs[dong_idx][0],
            floor_category=floor_category,
        )
        if store_id is None:
            st.error("등록에 실패했어요. 잠시 후 다시 시도해주세요.")
            return

        # shared/auth.py의 세션 키를 직접 갱신 — 로그아웃 없이 이 자리에서 바로
        # "내 매장 현황" 등 owner 섹션이 보이게 한다. store_status/store_snapshots/
        # display_name은 "이미 session_state에 있으면 재조회 안 함" 캐싱 방식이라
        # 새 매장 기준으로 다시 조회되도록 지워둔다.
        st.session_state["store_id"] = store_id
        st.session_state.user_info["store_id"] = store_id
        for key in ("display_name", "store_status", "store_snapshots"):
            st.session_state.pop(key, None)

        st.success(f"가게 등록이 완료됐습니다! (매장 코드: {store_id})")
        time.sleep(1.2)
        st.rerun()


def render_store_registration_prompt():
    with st.container(border=True):
        st.subheader("🏪 아직 등록된 매장이 없어요")
        st.caption("매장을 등록하면 내 매장 현황과 스냅샷 추이를 볼 수 있어요.")
        if st.button("🏪 가게 등록하기", key="store_reg_open_btn"):
            show_store_registration_dialog()


# ────────────────────────────────────────────────
# 6. 위험 구역 (회원 탈퇴)
# ────────────────────────────────────────────────
def render_danger_zone():
    with st.container(border=True, key="danger_zone"):
        label_col, action_col = st.columns([5, 2], vertical_alignment="center")
        with label_col:
            st.caption("⚠️ 계정을 삭제하면 되돌릴 수 없습니다. 신중하게 결정해주세요.")
        with action_col:
            with st.container(horizontal_alignment="right"):
                if st.button("회원 탈퇴", key="withdraw_btn"):
                    show_delete_account_dialog()


# ────────────────────────────────────────────────
# 메인 레이아웃 (탭 없이 세로 스크롤 구성)
# ────────────────────────────────────────────────
def main():
    # 실제 로그인 세션(shared/auth.py) 확인 — 이 페이지 전체가 로그인 여부/본인 데이터를
    # 전제로 하므로, 여기서 막아두면 아래 모든 render 함수는 로그인 상태를 다시 검사할
    # 필요가 없다. 관리자는 여기 대신 관리자 대시보드를 쓰게 안내한다(app.py의 ADMIN
    # 분기와 동일한 방식).
    user = auth.current_user()
    if user is None:
        st.warning("로그인이 필요한 페이지예요.")
        st.page_link("pages/login.py", label="로그인하러 가기", icon="➡️")
        st.stop()
    if user["user_type"] == "admin":
        st.info("관리자 계정은 마이페이지 대신 관리자 대시보드를 이용해주세요.")
        st.page_link("pages/admin_dashboard.py", label="관리자 대시보드로 이동", icon="➡️")
        st.stop()
    st.session_state.user_info = user

    init_session_state()
    inject_custom_css()

    with st.container(key="top_logo"):
        # 페이지 최상단 로고. 클릭하면 앱 홈(app/app.py, 루트 경로)으로 이동한다.
        st.markdown(
            f'<a class="mp-logo-link" href="/" target="_self">'
            f'<img src="data:image/svg+xml;base64,{_LOGO_SVG_B64}" alt="서울 상권분석"></a>',
            unsafe_allow_html=True,
        )

    with st.container(key="page_title"):
        st.title("마이페이지")

    render_header()
    render_my_info()

    # store_id 존재 여부로 매장 섹션을 노출한다(user_type=='owner' 조건은 뺐다) —
    # 예비창업자가 마이페이지에서 직접 가게를 등록(register_new_store())하면 user_type은
    # 그대로 founder로 두고 store_id만 채우기 때문(2026-08-29, auth.py의 owner 로그인
    # 규칙(로그인 아이디=store_id, 고정 비밀번호)까지 맞추려면 범위 밖인 auth.py를
    # 건드려야 해서 회피). "최근 관심있게 본 지역"/"자주 본 지역"(둘 다 user_view_history
    # 기반, 메인 지도 클릭 집계)은 매장과 무관하게 로그인한 유저라면 누구나 쌓이므로
    # 공통으로 보여준다. "지금 뜨는 업종 트렌드"/가게 등록 유도는 아직 매장이 없는
    # 예비창업자(founder) 전용으로만 노출한다.
    user = st.session_state.user_info
    if user.get("store_id"):
        render_store_status()
        render_favorite_regions()
        render_top_viewed_regions()
        render_recent_views()
        render_store_snapshots()
    else:
        render_top_viewed_regions()
        render_recent_views()
        if user["user_type"] == "founder":
            render_trend_widget()
            render_store_registration_prompt()

    render_prediction_history()

    st.divider()
    render_danger_zone()


if __name__ == "__main__":
    main()
