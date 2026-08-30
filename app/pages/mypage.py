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

# 상단 로고 — 점포 아이콘 이미지(PNG) + "Hotspot" 워드마크(2026-08-30 로고 교체 요청).
_LOGO_PNG_B64 = (
    "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"
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
        .mp-logo-link, .mp-logo-link:hover, .mp-logo-link:visited {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            text-decoration: none !important;
        }
        .mp-logo-link img {
            height: 40px;
            display: block;
        }
        .mp-logo-title {
            font-size: 1.5rem;
            font-weight: 800;
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            /* Streamlit이 마크다운 안 <a> 태그에 기본 파란색+밑줄을 강하게 입혀서
               (2026-08-30 스크린샷 확인) !important로 덮어써야 실제로 검게 나온다. */
            color: #111111 !important;
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
        # predictions 테이블 (FK user_id로 필터된 결과). get_prediction_history()는 DB
        # 연결이 안 됐으면 None, 연결은 됐는데 진짜 분석 이력이 없으면 빈 리스트를 반환한다
        # — 예전엔 `db_predictions or [더미...]`로 둘을 구분 안 해서, DB가 멀쩡히 연결된
        # 계정도 분석한 적이 없으면 더미 2건이 마치 본인 이력인 것처럼 보였다(2026-08-30
        # 확인). None(DB 연결 안 됨)일 때만 더미로 폴백하고, 빈 리스트는 그대로 빈 상태로
        # 둬서 render_prediction_history()의 "아직 조회한 분석 기록이 없습니다" 안내가
        # 실제로 뜨게 한다.
        db_predictions = get_prediction_history(user["user_id"])
        if db_predictions is None:
            st.session_state.predictions = [
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
        else:
            st.session_state.predictions = db_predictions

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
        st.subheader("🏪 내 매장 현황")

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
        st.subheader("📈 내 매장 스냅샷 추이")

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

    # 로고와 로그아웃 버튼을 같은 줄에 — 관리자 대시보드(admin_dashboard.py)와 동일한
    # 배치(로고 왼쪽, 버튼 오른쪽)로 맞춘다(2026-08-30 요청).
    logo_col, logout_col = st.columns([5, 1], vertical_alignment="center")
    with logo_col:
        with st.container(key="top_logo"):
            # 페이지 최상단 로고. 클릭하면 앱 홈(app/app.py, 루트 경로)으로 이동한다.
            st.markdown(
                f'<a class="mp-logo-link" href="/" target="_self">'
                f'<img src="data:image/png;base64,{_LOGO_PNG_B64}" alt="Hotspot">'
                f'<span class="mp-logo-title">Hotspot</span></a>',
                unsafe_allow_html=True,
            )
    with logout_col:
        # admin_dashboard.py의 로그아웃 버튼과 동일한 패턴 — 로그아웃 후 app.py(메인 지도)로 이동.
        if st.button("🚪 로그아웃", key="mypage_logout_btn"):
            auth.logout()
            st.switch_page("app.py")

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
