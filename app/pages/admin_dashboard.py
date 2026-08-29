"""
관리자 페이지 (관리자 대시보드) - Streamlit UI

ERD(schema.sql)에 있는 테이블만으로 구현 가능한 기능만 남겨뒀습니다.

상단 요약 카드 그리드(2026-08-29, 참고 이미지 스타일 요청 반영):
- 신규 가입자 · 전체 회원수 : 오늘/최근 7일 가입자 수 + 전 기간 대비 증감률(users)
- 회원 유형 분포          : owner/founder/admin 비율 도넛(users)
- 유동인구 TOP 5          : population_features.total_pop_avg → administrative_dongs 집계 (막대)
- 모델 성능 추이          : models를 학습 시각순으로 ROC-AUC/정확도 라인차트
- 동별 유동인구 vs 폐업률  : population_features × stores 집계 산점도

그 아래 상세 섹션:
- 오늘의 인기 키워드 : 상권 트렌드 키워드 TOP N(trend_keywords), 30초마다 자동 재조회
  (trend_keywords 자체는 snapshot_date 단위 배치 집계 테이블이라 "실시간"은 스트리밍이 아니라
  화면이 최신 데이터를 자동으로 다시 읽어오는 수준을 의미한다)
- 인기 조회지역     : dong_view_stats(지도 클릭 집계) → administrative_dongs 조인
  (dong_view_stats는 아직 schema.sql에 없는 테이블이라, 없으면 에러 대신 빈 상태로 표시한다)
- 사용자 관리      : 가입자 목록(users), 상호명 검색(store_snapshots.store_name)
- 모델 관리        : 이탈 예측 모델 버전·성능 지표 상세 테이블(models)
  (재학습 실행 버튼은 DB/실제 파이프라인과 연결된 게 없는 가짜 UI라 제거했다, 2026-08-29)

계정 정지(상태 컬럼), 브이월드 API 연동 상태, 데이터 갱신 현황, 방문자 수/API 호출량,
에러 로그/API 실패 이력은 현재 스키마에 대응하는 테이블/컬럼이 없어 제외했습니다.
(새 테이블·컬럼이 추가되면 다시 붙일 수 있습니다.)

DB 연결이 없거나 아직 쌓인 데이터가 없는 섹션은 화면 흐름 확인용 더미 데이터로 폴백합니다
(각 폴백 지점에 TODO 주석 참고). 실제 관리자 로그인 세션 확인은 shared/auth.py를 재사용합니다.
"""

import importlib.util
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
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

# app.shared.db를 일반 `from app.shared.db import ...`(또는 상대 import `from .db import ...`)로
# 가져오면 깨진다: 이 프로젝트의 진입점이 app/app.py라서 Streamlit이 sys.path에 app/ 디렉터리를
# 넣는데, 그 안에 있는 app.py 파일과 이름이 겹쳐서 "app" 패키지 해석 자체가 실패한다
# (mypage.py에서 먼저 겪었던 문제와 동일). 그래서 db.py를 파일 경로로 직접 로드해서 우회한다.
_DB_MODULE_PATH = Path(__file__).resolve().parents[1] / "shared" / "db.py"


@st.cache_resource
def _load_db_module():
    spec = importlib.util.spec_from_file_location("_admin_db_pmh", _DB_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ────────────────────────────────────────────────
# DB 조회 헬퍼 (mypage.py의 get_*() 함수들과 동일한 컨벤션 —
# DB 연결이 없으면 None, 연결은 됐지만 데이터가 없으면 빈 리스트를 반환해서
# 호출부(init_session_state)가 폴백 여부를 판단하게 한다)
# ────────────────────────────────────────────────
def get_admin_users(limit: int = 500) -> list[dict] | None:
    """users 기준 가입자 목록. store_name은 store_snapshots(최신 스냅샷)에서 매핑."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT u.user_id, u.login_id, u.user_type, u.store_id, u.created_at, "
        "(SELECT ss.store_name FROM store_snapshots ss WHERE ss.store_id = u.store_id "
        " ORDER BY ss.snapshot_date DESC LIMIT 1) AS store_name "
        "FROM users u "
        "ORDER BY u.created_at DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()

    result = []
    for r in rows:
        d = dict(r)
        d["created_at"] = d["created_at"].strftime("%Y-%m-%d") if d["created_at"] else None
        result.append(d)
    return result


def get_models() -> list[dict] | None:
    """models 테이블 전체를 학습 시각 최신순으로."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT model_id, model_name, version, model_type, accuracy, precision_score, "
        "recall_score, f1_score, roc_auc, trained_at, is_production "
        "FROM models ORDER BY trained_at DESC"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    result = []
    for r in rows:
        d = dict(r)
        for key in ("accuracy", "precision_score", "recall_score", "f1_score", "roc_auc"):
            d[key] = float(d[key]) if d[key] is not None else None
        d["trained_at"] = d["trained_at"].strftime("%Y-%m-%d %H:%M") if d["trained_at"] else None
        d["is_production"] = bool(d["is_production"])
        result.append(d)
    return result


def get_top_regions(limit: int = 5) -> list[dict] | None:
    """population_features.total_pop_avg -> administrative_dongs 조인, 유동인구 상위 N개."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT d.dong_name, p.total_pop_avg "
        "FROM population_features p "
        "JOIN administrative_dongs d ON d.dong_code = p.dong_code "
        "ORDER BY p.total_pop_avg DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).all()
    return [{"지역": r[0], "유동인구": float(r[1])} for r in rows]


@st.cache_data(ttl=30)  # "오늘의 인기 키워드"는 30초마다 자동 갱신되는 화면이라 캐시도 그에 맞춤
def get_trend_keywords(limit: int = 5) -> list[dict]:
    """trend_keywords에서 최신 snapshot_date 기준 store_count 상위 N개. DB 연결이
    없거나 아직 적재된 스냅샷이 없으면 빈 리스트(호출부가 데모 데이터로 폴백)."""
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


def get_model_performance_trend() -> list[dict] | None:
    """models를 학습 시각순으로 정렬해 roc_auc/accuracy 추이를 만든다. get_models()와
    같은 테이블이지만 "버전이 지날수록 성능이 좋아지고 있는지" 한눈에 보려는 용도라
    별도 조회로 둔다. DB 연결이 없으면 None, 성능 지표가 있는 모델이 하나도 없으면
    빈 리스트."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT model_name, trained_at, roc_auc, accuracy FROM models "
        "WHERE roc_auc IS NOT NULL ORDER BY trained_at ASC"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [
        {
            "모델": r["model_name"],
            "학습시각": r["trained_at"].strftime("%Y-%m-%d %H:%M") if r["trained_at"] else None,
            "ROC-AUC": float(r["roc_auc"]),
            "정확도": float(r["accuracy"]) if r["accuracy"] is not None else None,
        }
        for r in rows
    ]


def get_dong_population_vs_closure(limit: int = 300) -> list[dict] | None:
    """동별 유동인구(population_features.total_pop_avg) vs 폐업률(stores 집계)
    산점도용 데이터. "유동인구가 많은 동이 실제로 폐업률도 낮은지" 한눈에 보기 위함.
    DB 연결이 없으면 None, 매장이 하나도 없는 동은 폐업률을 계산할 수 없어 제외한다."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT d.dong_name, p.total_pop_avg, "
        "COUNT(s.store_id) AS total_stores, "
        "SUM(CASE WHEN s.is_closed THEN 1 ELSE 0 END) AS closed_stores "
        "FROM population_features p "
        "JOIN administrative_dongs d ON d.dong_code = p.dong_code "
        "LEFT JOIN stores s ON s.dong_code = p.dong_code "
        "GROUP BY d.dong_name, p.total_pop_avg "
        "HAVING total_stores > 0 "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
    return [
        {
            "동": r["dong_name"],
            "유동인구": float(r["total_pop_avg"]),
            # SUM(CASE WHEN ... THEN 1 ELSE 0 END)은 MySQL/TiDB에서 DECIMAL로 나오는데,
            # Altair가 vega-lite JSON으로 직렬화할 때 Decimal을 못 받아서 float로 캐스팅한다.
            "폐업률": float(r["closed_stores"]) / r["total_stores"],
        }
        for r in rows
    ]


def get_user_type_distribution(users: list[dict]) -> list[dict]:
    """이미 세션에 로드된 admin_users를 user_type별로 집계 — 별도 쿼리 없이
    화면에 필요한 형태로만 다시 묶어준다."""
    labels = {"owner": "기존점주", "founder": "예비창업자", "admin": "관리자"}
    counts: dict[str, int] = {}
    for u in users:
        label = labels.get(u["user_type"], u["user_type"])
        counts[label] = counts.get(label, 0) + 1
    return [{"유형": k, "인원": v} for k, v in counts.items()]


# ────────────────────────────────────────────────
# 추가 인사이트 (2026-08-29 요청 — 지금까지 admin_dashboard가 전혀 안 쓰던 테이블 활용)
# predictions/industries.custom_group/industry_transitions/support_actions
# ────────────────────────────────────────────────
def get_prediction_risk_by_industry(limit: int = 5) -> list[dict] | None:
    """predictions.score를 industry_code별로 평균 내서 모델이 실제로 위험하다고
    보는 업종 TOP N을 본다 — 이 프로젝트의 핵심 산출물(모델 예측)인데 admin_dashboard가
    지금까지 predictions 테이블을 한 번도 조회하지 않고 있었다. 표본이 너무 적은
    업종(우연히 극단값)은 제외하려고 최소 10건 이상만 집계한다."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT i.industry_name, AVG(p.score) AS avg_score, COUNT(*) AS n "
        "FROM predictions p JOIN industries i ON i.industry_code = p.industry_code "
        "GROUP BY i.industry_name "
        "HAVING n >= 10 "
        "ORDER BY avg_score DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
    return [{"업종": r["industry_name"], "평균위험도": float(r["avg_score"]), "건수": r["n"]} for r in rows]


def get_industry_group_distribution() -> list[dict] | None:
    """stores.current_industry_code -> industries.custom_group(대분류 10개)로 집계.
    지금까지 industries 테이블은 이름 매핑용으로만 조인됐지 custom_group(대분류)
    자체를 보여주는 화면이 없었다."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT i.custom_group, COUNT(*) AS n "
        "FROM stores s JOIN industries i ON i.industry_code = s.current_industry_code "
        "GROUP BY i.custom_group ORDER BY n DESC"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [{"대분류": r["custom_group"], "매장수": r["n"]} for r in rows]


def get_top_industry_transitions(limit: int = 5) -> list[dict] | None:
    """industry_transitions — 업종이 어떤 업종에서 어떤 업종으로 가장 많이 바뀌었는지
    TOP N. admin_dashboard에서 지금까지 전혀 안 쓰던 테이블."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT fi.industry_name AS from_name, ti.industry_name AS to_name, COUNT(*) AS n "
        "FROM industry_transitions t "
        "JOIN industries fi ON fi.industry_code = t.from_industry_code "
        "JOIN industries ti ON ti.industry_code = t.to_industry_code "
        "GROUP BY fi.industry_name, ti.industry_name "
        "ORDER BY n DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
    return [{"전환": f"{r['from_name']} → {r['to_name']}", "건수": r["n"]} for r in rows]


def search_stores(query: str, limit: int = 20) -> list[dict]:
    """상호명 또는 store_id로 매장 검색 — 지원 조치 등록 시 대상 매장을 고르기 위함.
    매장이 수만 건이라 전체 목록 대신 검색 결과만 가져온다."""
    if not query:
        return []
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return []

    from sqlalchemy import text

    sql = text(
        "SELECT DISTINCT store_id, store_name FROM store_snapshots "
        "WHERE store_id LIKE :q OR store_name LIKE :q "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"q": f"%{query}%", "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def log_admin_support_action(store_id: str, admin_user_id: str, action_type: str, notes: str | None) -> None:
    """support_actions에 관리자 개입 조치를 기록한다. app/shared/write_support_action.py에
    이미 log_support_action()이 있지만 잘못된 import 방식(from app.shared.db import ...)을
    써서 이 프로젝트의 멀티페이지 구조에서 그대로 가져다 쓰면 깨진다 — 다른 페이지들과
    같은 방식대로 이 파일 안에 독립적으로 쿼리를 둔다."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return

    from datetime import date as _date

    from sqlalchemy import text

    sql = text(
        "INSERT INTO support_actions (store_id, admin_user_id, action_type, action_date, "
        "follow_up_closure_status, notes) "
        "VALUES (:store_id, :admin_user_id, :action_type, :action_date, NULL, :notes)"
    )
    with engine.begin() as conn:
        conn.execute(sql, {
            "store_id": store_id, "admin_user_id": admin_user_id,
            "action_type": action_type, "action_date": _date.today(), "notes": notes,
        })


def get_recent_support_actions(limit: int = 10) -> list[dict] | None:
    """support_actions 최근 조치 이력 (담당 관리자 로그인ID, 매장 상호명 포함)."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT sa.action_id, sa.store_id, sa.action_type, sa.action_date, sa.notes, "
        "u.login_id AS admin_login_id, "
        "(SELECT ss.store_name FROM store_snapshots ss WHERE ss.store_id = sa.store_id "
        " ORDER BY ss.snapshot_date DESC LIMIT 1) AS store_name "
        "FROM support_actions sa "
        "JOIN users u ON u.user_id = sa.admin_user_id "
        "ORDER BY sa.action_date DESC, sa.action_id DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
    result = []
    for r in rows:
        d = dict(r)
        d["action_date"] = d["action_date"].isoformat() if d["action_date"] else None
        result.append(d)
    return result


st.set_page_config(
    page_title="관리자 대시보드 | 상권분석",
    page_icon="🛠️",
    layout="wide",
)

# 상단 로고 (지도 핀 아이콘 + "서울 상권분석" 워드마크). mypage.py와 동일한 로고를 그대로 재사용.
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
# 스타일 (mypage.py와 통일: 로고 위치, 타이틀/서브헤더 크기)
# ────────────────────────────────────────────────
def inject_custom_css():
    st.markdown(
        """
        <style>
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
        /* 대시보드 요약 카드(신규가입자/회원유형/유동인구/전체회원수/성능추이/산점도) 공통
           스타일 — 참고 이미지처럼 각지지 않고 둥글고 옅은 그림자가 있는 카드로 통일한다.
           st-key-*는 Streamlit 공식 API라 내부 DOM 구조를 추측하는 것보다 안전하다. */
        [class*="st-key-dash_card_"] {
            border-radius: 1rem !important;
            border-color: rgba(49, 51, 63, 0.08) !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }
        .dash-card-title {
            font-size: 0.85rem;
            font-weight: 600;
            opacity: 0.7;
            margin-bottom: 0.75rem;
        }
        .dash-stat-value {
            font-size: 2.1rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .dash-stat-delta-up { color: #2ea043; font-weight: 600; font-size: 0.9rem; }
        .dash-stat-delta-down { color: #f85149; font-weight: 600; font-size: 0.9rem; }
        .dash-stat-delta-flat { color: #8b949e; font-weight: 600; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────
# 더미 데이터 (DB 연결이 없거나 아직 쌓인 데이터가 없을 때의 화면 확인용 폴백)
# ────────────────────────────────────────────────
_DUMMY_ADMIN_USERS = [
    {
        "user_id": "u_0001",
        "login_id": "parkminha76",
        "user_type": "owner",
        "store_id": "store_1042",
        "store_name": "카페 온기",
        "created_at": "2025-01-15",
    },
    {
        "user_id": "u_0002",
        "login_id": "founder_kim",
        "user_type": "founder",
        "store_id": None,
        "store_name": None,
        "created_at": "2025-03-02",
    },
    {
        "user_id": "u_0003",
        "login_id": "admin_lee",
        "user_type": "admin",
        "store_id": None,
        "store_name": None,
        "created_at": "2024-11-20",
    },
    {
        "user_id": "u_0004",
        "login_id": "test_user",
        "user_type": "owner",
        "store_id": "store_2091",
        "store_name": "역삼 분식",
        "created_at": "2025-06-10",
    },
    {
        "user_id": "u_0005",
        "login_id": "new_owner",
        "user_type": "owner",
        "store_id": "store_3087",
        "store_name": "성수 베이커리",
        "created_at": date.today().isoformat(),  # "오늘 가입" 데모용
    },
]

_DUMMY_MODELS = [
    {
        "model_id": "lightgbm_v3",
        "model_name": "LightGBM 폐업예측",
        "version": "v3",
        "model_type": "ML",
        "accuracy": 0.891,
        "precision_score": 0.870,
        "recall_score": 0.850,
        "f1_score": 0.860,
        "roc_auc": 0.920,
        "trained_at": "2026-08-15 10:00",
        "is_production": True,
    },
    {
        "model_id": "catboost_v2",
        "model_name": "CatBoost 폐업예측",
        "version": "v2",
        "model_type": "ML",
        "accuracy": 0.879,
        "precision_score": 0.850,
        "recall_score": 0.840,
        "f1_score": 0.845,
        "roc_auc": 0.905,
        "trained_at": "2026-07-20 10:00",
        "is_production": False,
    },
    {
        "model_id": "mlp_v1",
        "model_name": "MLP 폐업예측",
        "version": "v1",
        "model_type": "DL",
        "accuracy": 0.862,
        "precision_score": 0.830,
        "recall_score": 0.810,
        "f1_score": 0.820,
        "roc_auc": 0.889,
        "trained_at": "2026-06-01 10:00",
        "is_production": False,
    },
]

_DUMMY_TOP_REGIONS = [
    {"지역": "여의동", "유동인구": 108739},
    {"지역": "역삼1동", "유동인구": 108306},
    {"지역": "화곡8동", "유동인구": 86418},
    {"지역": "서교동", "유동인구": 84851},
    {"지역": "서초3동", "유동인구": 70607},
]

_DUMMY_TREND_KEYWORDS = [
    {"keyword": "무인카페", "store_count": 128, "growth_rate": 0.42},
    {"keyword": "반려동물동반", "store_count": 96, "growth_rate": 0.31},
    {"keyword": "회전초밥", "store_count": 87, "growth_rate": 0.12},
    {"keyword": "저속노화식단", "store_count": 74, "growth_rate": 0.58},
    {"keyword": "노키즈존", "store_count": 63, "growth_rate": -0.08},
]

_DUMMY_MODEL_PERFORMANCE_TREND = [
    {"모델": "MLP v1", "학습시각": "2026-06-01 10:00", "ROC-AUC": 0.889, "정확도": 0.862},
    {"모델": "CatBoost v2", "학습시각": "2026-07-20 10:00", "ROC-AUC": 0.905, "정확도": 0.879},
    {"모델": "LightGBM v3", "학습시각": "2026-08-15 10:00", "ROC-AUC": 0.920, "정확도": 0.891},
]

_DUMMY_DONG_POPULATION_VS_CLOSURE = [
    {"동": "역삼1동", "유동인구": 108306, "폐업률": 0.09},
    {"동": "서교동", "유동인구": 84851, "폐업률": 0.12},
    {"동": "성수동", "유동인구": 62340, "폐업률": 0.14},
    {"동": "화곡8동", "유동인구": 86418, "폐업률": 0.11},
    {"동": "서초3동", "유동인구": 70607, "폐업률": 0.10},
    {"동": "신월6동", "유동인구": 3524, "폐업률": 0.31},
    {"동": "홍은2동", "유동인구": 21590, "폐업률": 0.22},
    {"동": "방학3동", "유동인구": 17796, "폐업률": 0.29},
]

_DUMMY_PREDICTION_RISK_BY_INDUSTRY = [
    {"업종": "가축 사료 소매업", "평균위험도": 0.854, "건수": 333},
    {"업종": "가발 소매업", "평균위험도": 0.518, "건수": 419},
    {"업종": "그 외 기타 상품 전문 소매업", "평균위험도": 0.270, "건수": 2794},
    {"업종": "치과병원", "평균위험도": 0.257, "건수": 163},
    {"업종": "종합병원", "평균위험도": 0.197, "건수": 108},
]

_DUMMY_INDUSTRY_GROUP_DISTRIBUTION = [
    {"대분류": "음식", "매장수": 217427},
    {"대분류": "소매", "매장수": 166398},
    {"대분류": "과학·기술", "매장수": 122821},
    {"대분류": "수리·개인", "매장수": 69234},
    {"대분류": "교육", "매장수": 67488},
    {"대분류": "시설관리·임대", "매장수": 33381},
    {"대분류": "부동산", "매장수": 31872},
    {"대분류": "예술·스포츠", "매장수": 30969},
    {"대분류": "보건의료", "매장수": 23699},
    {"대분류": "숙박", "매장수": 15606},
]

_DUMMY_TOP_INDUSTRY_TRANSITIONS = [
    {"전환": "백반/한정식 → 돼지고기 구이/찜", "건수": 997},
    {"전환": "백반/한정식 → 요리 주점", "건수": 696},
    {"전환": "기타 의류 소매업 → 여성 의류 소매업", "건수": 326},
    {"전환": "생맥주 전문 → 요리 주점", "건수": 289},
    {"전환": "경양식 → 요리 주점", "건수": 276},
]


# ────────────────────────────────────────────────
# 세션 상태 초기화
# 관리자 로그인 세션 확인은 main()에서 shared/auth.py로 처리하고, 여기서는 각 섹션을
# DB에서 조회해 세션 상태를 채운다. trend_keywords는 30초마다 자동 갱신돼야 해서
# session_state 캐싱 없이 render_trend_keywords()에서 매번 get_trend_keywords()를 부른다.
# ────────────────────────────────────────────────
def init_session_state():
    if "admin_users" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없거나 가입자가 없으면 화면 확인용 더미로.
        st.session_state.admin_users = get_admin_users() or _DUMMY_ADMIN_USERS

    if "models" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없거나 등록된 모델이 없으면 화면 확인용 더미로.
        st.session_state.models = get_models() or _DUMMY_MODELS

    if "top_regions" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없으면 화면 확인용 더미로.
        st.session_state.top_regions = get_top_regions() or _DUMMY_TOP_REGIONS

    if "model_performance_trend" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없거나 성능 지표가 있는 모델이 없으면 더미로.
        st.session_state.model_performance_trend = (
            get_model_performance_trend() or _DUMMY_MODEL_PERFORMANCE_TREND
        )

    if "dong_population_vs_closure" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없으면 화면 확인용 더미로.
        st.session_state.dong_population_vs_closure = (
            get_dong_population_vs_closure() or _DUMMY_DONG_POPULATION_VS_CLOSURE
        )

    if "prediction_risk_by_industry" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없으면 화면 확인용 더미로.
        st.session_state.prediction_risk_by_industry = (
            get_prediction_risk_by_industry() or _DUMMY_PREDICTION_RISK_BY_INDUSTRY
        )

    if "industry_group_distribution" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없으면 화면 확인용 더미로.
        st.session_state.industry_group_distribution = (
            get_industry_group_distribution() or _DUMMY_INDUSTRY_GROUP_DISTRIBUTION
        )

    if "top_industry_transitions" not in st.session_state:
        # TODO(데모용 임시 폴백): DB 연결이 없으면 화면 확인용 더미로.
        st.session_state.top_industry_transitions = (
            get_top_industry_transitions() or _DUMMY_TOP_INDUSTRY_TRANSITIONS
        )


# ────────────────────────────────────────────────
# 1. 회원수 조회
# ────────────────────────────────────────────────
def _signup_counts(users: list[dict]) -> dict:
    """오늘/어제/최근7일/그 이전7일 가입자 수를 한 번에 계산 — 신규가입자 카드와
    전체회원수 카드 둘 다 여기서 재사용한다."""
    today = date.today()
    week_start = (today - timedelta(days=7)).isoformat()
    prev_week_start = (today - timedelta(days=14)).isoformat()
    return {
        "today": sum(1 for u in users if u["created_at"] == today.isoformat()),
        "yesterday": sum(1 for u in users if u["created_at"] == (today - timedelta(days=1)).isoformat()),
        "this_week": sum(1 for u in users if u["created_at"] >= week_start),
        "prev_week": sum(1 for u in users if prev_week_start <= u["created_at"] < week_start),
    }


def _render_stat_delta(current: int, previous: int, unit: str = "명") -> None:
    """참고 이미지의 "+2.3%" 스타일 — 이전 값 대비 증감률을 초록/빨강 텍스트로 보여준다.
    이전 값이 0이면 퍼센트가 정의되지 않으니 절대값 증감(+N명)으로 대신 표시한다."""
    if previous > 0:
        pct = (current - previous) / previous * 100
        cls = "dash-stat-delta-up" if pct > 0 else "dash-stat-delta-down" if pct < 0 else "dash-stat-delta-flat"
        text = f"{pct:+.1f}%"
    else:
        diff = current - previous
        cls = "dash-stat-delta-up" if diff > 0 else "dash-stat-delta-down" if diff < 0 else "dash-stat-delta-flat"
        text = f"{diff:+d}{unit}"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def render_signup_stats_card():
    """신규 가입자·전체 회원수를 카드 하나에 나란히 — 원래 카드 2개로 나뉘어 있었는데,
    옆의 회원유형 도넛 카드 하나와 높이/개수 균형이 안 맞아 보인다는 피드백(2026-08-29
    스크린샷)으로 하나로 합쳤다."""
    with st.container(border=True, key="dash_card_signup_stats"):
        users = st.session_state.admin_users
        counts = _signup_counts(users)

        # CSS min-height는 Streamlit이 컬럼-카드 사이에 끼워 넣는 height:auto 중간
        # 래퍼 때문에 안 먹혔다(2026-08-29 확인) — 그래서 CSS 대신 실제 빈 콘텐츠로
        # 카드 높이를 옆 도넛 카드(차트 260px + 제목 + 범례 + 패딩)에 맞췄다. 처음엔
        # 여백을 전부 아래에만 둬서 텍스트가 위로 쏠려 보였는데(2026-08-29 스크린샷),
        # 위/아래로 절반씩 나눠서 내용이 카드 세로 가운데에 오도록 한다.
        st.markdown('<div style="height:95px;"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="dash-card-title">신규 가입자 · 오늘</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dash-stat-value">{counts["today"]:,}명</div>', unsafe_allow_html=True)
            _render_stat_delta(counts["today"], counts["yesterday"])
            st.caption("전일 대비")
        with col2:
            st.markdown('<div class="dash-card-title">전체 회원수</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dash-stat-value">{len(users):,}명</div>', unsafe_allow_html=True)
            _render_stat_delta(counts["this_week"], counts["prev_week"])
            st.caption("최근 7일 신규가입 · 이전 7일 대비")

        st.markdown('<div style="height:95px;"></div>', unsafe_allow_html=True)


# 회원 유형 3종 고정 순서/색상 — 매번 다르게 배정되지 않도록 카테고리 색은 고정한다
# (dataviz 원칙: 카테고리 색은 고정 순서로 배정, 매번 다르게 순환하지 않음).
_USER_TYPE_COLORS = {"기존점주": "#3D6FD9", "예비창업자": "#6C63FF", "관리자": "#F2A65A"}


def render_user_type_donut():
    with st.container(border=True, key="dash_card_user_type"):
        st.markdown('<div class="dash-card-title">회원 유형 분포</div>', unsafe_allow_html=True)
        df = pd.DataFrame(get_user_type_distribution(st.session_state.admin_users))
        order = [k for k in _USER_TYPE_COLORS if k in df["유형"].values]
        # outerRadius=90(지름 180px)에 height=220만 주면 범례가 차지하는 공간 때문에
        # 실제 그려지는 원 영역이 180px보다 좁아져서 위쪽이 잘렸다(2026-08-29 스크린샷).
        # 반지름을 줄이고 카드 높이를 다른 2행 차트와 맞춰(260) 여유를 준다.
        chart = (
            alt.Chart(df)
            .mark_arc(innerRadius=48, outerRadius=78)
            .encode(
                theta=alt.Theta("인원:Q", stack=True),
                color=alt.Color(
                    "유형:N",
                    scale=alt.Scale(domain=order, range=[_USER_TYPE_COLORS[k] for k in order]),
                    sort=order,
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["유형", "인원"],
            )
            .properties(height=260, padding={"top": 15, "bottom": 5, "left": 5, "right": 5})
        )
        st.altair_chart(chart, use_container_width=True)


def render_model_performance_chart():
    with st.container(border=True, key="dash_card_performance"):
        st.markdown('<div class="dash-card-title">모델 성능 추이</div>', unsafe_allow_html=True)
        st.caption("models 테이블 · 학습 시각순 ROC-AUC / 정확도 추이")
        df = pd.DataFrame(st.session_state.model_performance_trend)
        if df.empty:
            st.info("아직 성능 지표가 있는 모델이 없습니다.")
            return
        melted = df.melt(
            id_vars=["모델", "학습시각"], value_vars=["ROC-AUC", "정확도"],
            var_name="지표", value_name="값",
        )
        chart = (
            alt.Chart(melted)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("학습시각:N", title=None, sort=list(df["학습시각"])),
                y=alt.Y("값:Q", scale=alt.Scale(zero=False), title=None),
                color=alt.Color(
                    "지표:N",
                    scale=alt.Scale(domain=["ROC-AUC", "정확도"], range=["#3D6FD9", "#A78BFA"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["모델", "학습시각", "지표", alt.Tooltip("값:Q", format=".3f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


def render_dong_closure_scatter():
    with st.container(border=True, key="dash_card_scatter"):
        st.markdown('<div class="dash-card-title">동별 유동인구 vs 폐업률</div>', unsafe_allow_html=True)
        st.caption("population_features.total_pop_avg × stores 폐업률 (동 단위)")
        df = pd.DataFrame(st.session_state.dong_population_vs_closure)
        chart = (
            alt.Chart(df)
            .mark_circle(size=70, opacity=0.55, color="#6C63FF")
            .encode(
                # y축 제목("폐업률")을 세로로 회전시키면 좁은 카드 폭에서 글자가 한 자씩
                # 줄바꿈돼 깨져 보였다(2026-08-29 스크린샷) — 카드 제목/캡션에 이미 축
                # 의미가 나와 있으니 title=None으로 없애고 % 포맷 라벨만 남긴다.
                x=alt.X("유동인구:Q", title=None),
                y=alt.Y("폐업률:Q", axis=alt.Axis(format="%"), title=None),
                tooltip=["동", "유동인구", alt.Tooltip("폐업률:Q", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


def render_prediction_risk_chart():
    with st.container(border=True, key="dash_card_prediction_risk"):
        st.markdown('<div class="dash-card-title">업종별 평균 폐업위험도 TOP 5</div>', unsafe_allow_html=True)
        st.caption("predictions.score 평균 (표본 10건 이상 업종만) — 모델이 실제로 위험하다고 보는 업종")
        df = pd.DataFrame(st.session_state.prediction_risk_by_industry)
        chart = (
            alt.Chart(df)
            .mark_bar(color="#D8454A")
            .encode(
                x=alt.X("업종", sort="-y", axis=alt.Axis(labelAngle=-20), title=None),
                y=alt.Y("평균위험도:Q", axis=alt.Axis(format="%"), title=None),
                tooltip=["업종", alt.Tooltip("평균위험도:Q", format=".1%"), "건수"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


def render_industry_group_chart():
    with st.container(border=True, key="dash_card_industry_group"):
        st.markdown('<div class="dash-card-title">업종 대분류별 매장 분포</div>', unsafe_allow_html=True)
        st.caption("industries.custom_group(대분류 10종) 기준 stores 집계")
        df = pd.DataFrame(st.session_state.industry_group_distribution)
        chart = (
            alt.Chart(df)
            .mark_bar(color="#3D6FD9")
            .encode(
                x=alt.X("매장수:Q", title=None),
                y=alt.Y("대분류", sort="-x", title=None),
                tooltip=["대분류", "매장수"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


# ────────────────────────────────────────────────
# 2. 인기 키워드
# ────────────────────────────────────────────────
@st.fragment(run_every="30s")
def render_trend_keywords():
    with st.container(border=True):
        # "마지막 갱신" 시각을 캡션 줄로 따로 두지 않고 타이틀 옆에 바로 붙인다(2026-08-29
        # 요청) — st.subheader는 다른 텍스트를 같은 줄에 못 넣어서 h3 태그로 직접 그린다.
        st.markdown(
            '<div style="display:flex; align-items:baseline; justify-content:space-between; gap:0.75rem;">'
            '<h3 style="margin:0;">🔑 오늘의 인기 키워드 TOP 5</h3>'
            f'<span style="font-size:0.75rem; opacity:0.55; white-space:nowrap;">'
            f'⏱️ 마지막 갱신: {datetime.now():%H:%M:%S}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "trend_keywords 테이블 기준 (최신 snapshot_date의 store_count 상위 N개) "
            "· 30초마다 자동 갱신"
        )

        # session_state에 한 번 캐싱해두면 30초 자동 갱신이 무의미해지므로, 매 프래그먼트
        # 실행마다 get_trend_keywords()(자체 ttl=30 캐시)를 새로 부른다.
        keywords = get_trend_keywords(limit=5) or _DUMMY_TREND_KEYWORDS

        for rank, kw in enumerate(keywords, start=1):
            rate = kw["growth_rate"]
            if rate is None:
                g_bg, g_fg, g_text = "rgba(139, 148, 158, 0.15)", "#8b949e", "변화 없음"
            elif rate >= 0:
                g_bg, g_fg, g_text = "rgba(46, 160, 67, 0.15)", "#2ea043", f"▲ {rate:.0%}"
            else:
                g_bg, g_fg, g_text = "rgba(248, 81, 73, 0.15)", "#f85149", f"▼ {abs(rate):.0%}"

            # Streamlit 컨테이너 키에 CSS를 거는 대신, 카드 테두리까지 통째로 직접 그린다.
            # (내부 DOM 구조를 추측해서 맞추는 방식이 신뢰할 수 없어서 자체 완결형으로 전환)
            st.markdown(
                '<div style="'
                'border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; '
                'padding:1rem 1.25rem; margin-bottom:0.75rem; '
                'display:flex; align-items:center; gap:1rem;">'
                f'<div class="mp-rank-badge" style="flex-shrink:0;">{rank}</div>'
                '<div style="flex:1; min-width:0;">'
                f'<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">'
                f'{kw["keyword"]}</div>'
                '<div style="font-size:0.8rem; opacity:0.65;">'
                f'{kw["store_count"]}개 매장</div>'
                "</div>"
                '<div style="flex-shrink:0;">'
                f'<span style="background:{g_bg}; color:{g_fg}; padding:0.15rem 0.55rem; '
                "border-radius:999px; font-size:0.8rem; font-weight:600; white-space:nowrap;\">"
                f"{g_text}</span>"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────
# 3. 사용자 관리
# ────────────────────────────────────────────────
def render_user_management():
    with st.container(border=True):
        st.subheader("👥 사용자 관리")
        st.caption("상호명으로 가입자를 검색합니다 (가입자가 많아 전체 목록은 기본으로 안 보여줍니다)")

        search = st.text_input(
            "🔍 상호명 검색", placeholder="상호명으로 검색...", key="admin_user_search"
        )

        if not search:
            # 가입자가 많아지면 전체 목록을 매번 렌더링하는 게 부담이라, 검색했을
            # 때만 결과를 보여주는 방식으로 바꿨다(2026-08-29 요청) — 기존엔 검색 전에도
            # 항상 전체 목록이 테이블로 떠 있었음.
            st.info("상호명을 입력하면 가입자를 검색합니다.")
            return

        users = st.session_state.admin_users
        df = pd.DataFrame(users)
        df["store_name"] = df["store_name"].fillna("-")
        df["store_id"] = df["store_id"].fillna("-")

        # TODO: 실제로는 WHERE store_snapshots.store_name LIKE :search 로 DB에서 검색
        df = df[df["store_name"].str.contains(search, case=False, na=False)]

        if df.empty:
            st.info("검색 결과가 없습니다.")
        else:
            df = df[["store_name", "store_id", "user_type", "created_at"]].rename(
                columns={
                    "store_name": "상호명",
                    "store_id": "매장 ID",
                    "user_type": "권한",
                    "created_at": "가입일",
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────
# 4. 모델 관리
# ────────────────────────────────────────────────
def render_model_management():
    with st.container(border=True):
        st.subheader("🤖 모델 관리")
        st.caption("이탈 예측 모델 버전 관리, 재학습 실행, 성능 지표(정확도 등) 모니터링")

        models = st.session_state.models
        df = pd.DataFrame(models).rename(
            columns={
                "model_id": "모델 ID",
                "model_name": "모델명",
                "version": "버전",
                "model_type": "유형",
                "accuracy": "정확도",
                "precision_score": "정밀도",
                "recall_score": "재현율",
                "f1_score": "F1",
                "roc_auc": "ROC-AUC",
                "trained_at": "학습 시각",
                "is_production": "운영 반영",
            }
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "정확도": st.column_config.ProgressColumn(
                    "정확도", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "운영 반영": st.column_config.CheckboxColumn("운영 반영", disabled=True),
            },
        )
        # "재학습 실행" 버튼(선택한 모델 이름으로 st.info만 띄우고 실제로는 아무 것도
        # 하지 않는 가짜 버튼)은 실제 재학습 파이프라인이나 DB 쓰기와 전혀 연결돼
        # 있지 않아서 제거했다(2026-08-29 요청 — DB와 무관한 기능 정리).


# ────────────────────────────────────────────────
# 5. 인기 조회 지역 TOP N
# ────────────────────────────────────────────────
def render_top_regions():
    with st.container(border=True, key="dash_card_top_regions"):
        st.markdown('<div class="dash-card-title">유동인구 TOP 5</div>', unsafe_allow_html=True)
        st.caption("population_features.total_pop_avg → administrative_dongs 집계 기준")

        top_df = pd.DataFrame(st.session_state.top_regions)
        region_chart = (
            alt.Chart(top_df)
            .mark_bar(color="#3D6FD9")
            .encode(
                x=alt.X("지역", sort="-y", axis=alt.Axis(labelAngle=0), title=None),
                y=alt.Y("유동인구", title=None),
                tooltip=["지역", "유동인구"],
            )
            .properties(height=260)
        )
        st.altair_chart(region_chart, use_container_width=True)


# ────────────────────────────────────────────────
# 6. 인기 조회 지역 TOP N
# ────────────────────────────────────────────────
# dong_view_stats: 지도 클릭을 직접 집계하는 테이블 (아직 schema.sql에는 없음).
# predictions 기반 집계와 달리 existing_store/new_location 조회를 가리지 않고 전부 잡힌다.
_POPULAR_AREAS_SQL = """
    SELECT
        d.dong_code,
        d.gu_name,
        d.dong_name,
        v.view_count AS total_views,
        v.last_viewed_at
    FROM dong_view_stats v
    JOIN administrative_dongs d ON d.dong_code = v.dong_code
    ORDER BY v.view_count DESC
    LIMIT :limit
"""


@st.cache_data(ttl=30)  # 30초 캐시("오늘의 인기 키워드"와 동일) — 5분은 너무 길어서
# 클릭 직후 관리자 페이지를 보면 카운트가 안 된 것처럼 보인다는 피드백(2026-08-29)
def _fetch_popular_areas(limit: int = 5) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["dong_code", "gu_name", "dong_name", "total_views", "last_viewed_at"])

    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return empty

    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(_POPULAR_AREAS_SQL), conn, params={"limit": limit})
        # DB 서버(TiDB Cloud) 시스템 시간대가 UTC라서 last_viewed_at이 NOW()로 UTC 기준
        # 저장돼 있다 — 한국 시간(KST, UTC+9)으로 보정한다.
        if not df.empty:
            df["last_viewed_at"] = df["last_viewed_at"] + timedelta(hours=9)
        return df
    except pd.errors.DatabaseError:
        # pd.read_sql은 SQLAlchemy 예외(테이블 없음 등)를 pandas.errors.DatabaseError로 감싸서
        # 던진다 — sqlalchemy.exc.DatabaseError를 잡으면 안 걸린다 (실제로 안 걸렸던 원인).
        # dong_view_stats 테이블이 아직 스키마에 없는 경우(마이그레이션 전)도 에러 없이 빈 화면으로.
        return empty


# TODO(데모용 임시 폴백): dong_view_stats가 아직 없거나 비어 있을 때 화면 확인용으로 보여주는
# 더미. 실제 클릭 집계가 쌓이기 시작하면(테이블이 생기고 데이터가 들어오면) 자동으로 안 쓰인다 —
# 그래도 실제 운영 전엔 이 폴백 자체를 지우는 게 맞다.
_DUMMY_POPULAR_AREAS = pd.DataFrame(
    [
        {"gu_name": "강남구", "dong_name": "역삼동", "total_views": 482, "last_viewed_at": "2026-08-28 09:12"},
        {"gu_name": "마포구", "dong_name": "서교동", "total_views": 401, "last_viewed_at": "2026-08-28 08:47"},
        {"gu_name": "성동구", "dong_name": "성수동", "total_views": 356, "last_viewed_at": "2026-08-27 22:03"},
        {"gu_name": "강남구", "dong_name": "논현동", "total_views": 312, "last_viewed_at": "2026-08-27 20:15"},
        {"gu_name": "마포구", "dong_name": "합정동", "total_views": 289, "last_viewed_at": "2026-08-27 19:40"},
    ]
)


def render_popular_query_regions():
    with st.container(border=True):
        st.subheader("📍 인기 조회지역")
        # 원래 캡션이 길어서 3열 그리드의 좁은 컬럼 폭에서 2줄로 줄바꿈돼, 옆 카드들과
        # 헤더 높이가 안 맞았다(2026-08-29 스크린샷) — 한 줄에 들어가게 줄였다.
        st.caption("dong_view_stats 기준 · 지도 클릭 조회 랭킹 (게스트 제외)")

        df = _fetch_popular_areas(limit=5)
        if df.empty:
            df = _DUMMY_POPULAR_AREAS

        for rank, row in enumerate(df.itertuples(index=False), start=1):
            location = f"{row.gu_name or ''} {row.dong_name or ''}".strip()
            detail = f"{int(row.total_views)}회 조회"
            if row.last_viewed_at:
                detail += f" · 최근 {row.last_viewed_at}"
            # 카드 테두리까지 통째로 직접 그린다 (컨테이너 키 CSS는 신뢰할 수 없어서 자체 완결형).
            st.markdown(
                '<div style="'
                'border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; '
                'padding:1rem 1.25rem; margin-bottom:0.75rem; '
                'display:flex; align-items:center; gap:1rem;">'
                f'<div class="mp-rank-badge" style="flex-shrink:0;">{rank}</div>'
                '<div style="flex:1; min-width:0;">'
                '<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">'
                f'{location}</div>'
                '<div style="font-size:0.8rem; opacity:0.65;">'
                f'{detail}</div>'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_top_industry_transitions():
    with st.container(border=True):
        st.subheader("🔄 업종 전환 TOP 5")
        st.caption("industry_transitions 기준: 가장 흔하게 관측된 업종 전환")

        # 옆 두 카드(인기 키워드/인기 조회지역)는 항목마다 "제목 + 설명 캡션" 2줄
        # 구조인데, 여기는 제목 1줄 + 오른쪽 건수만 있어서 항목당 높이가 짧아 카드
        # 전체 높이가 안 맞았다(2026-08-29 스크린샷) — 건수를 오른쪽 대신 제목 아래
        # 캡션 줄로 옮겨서 같은 2줄 구조로 맞춘다.
        transitions = st.session_state.top_industry_transitions
        for rank, t in enumerate(transitions, start=1):
            st.markdown(
                '<div style="'
                'border:1px solid rgba(49, 51, 63, 0.2); border-radius:0.5rem; '
                'padding:1rem 1.25rem; margin-bottom:0.75rem; '
                'display:flex; align-items:center; gap:1rem;">'
                f'<div class="mp-rank-badge" style="flex-shrink:0;">{rank}</div>'
                '<div style="flex:1; min-width:0;">'
                '<div style="font-weight:700; font-size:1.05rem; margin-bottom:0.2rem;">'
                f'{t["전환"]}</div>'
                '<div style="font-size:0.8rem; opacity:0.65;">'
                f'{t["건수"]:,}건 관측</div>'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


_SUPPORT_ACTION_TYPES = ["전화상담", "현장방문", "임대료 지원 안내", "컨설팅 연계", "기타"]


def render_support_actions():
    with st.container(border=True):
        st.subheader("🛟 지원 조치 등록 · 이력")
        st.caption(
            "support_actions 테이블 — 위험 매장에 대한 관리자 개입 조치를 등록하고 "
            "이력을 확인합니다 (지금까지 admin_dashboard가 안 쓰던 테이블)"
        )

        search = st.text_input(
            "🔍 매장 검색 (상호명 또는 매장 ID)", key="support_action_store_search"
        )
        selected_store_id = None
        if search:
            matches = search_stores(search)
            if not matches:
                st.info("검색 결과가 없습니다.")
            else:
                options = {f"{m['store_name']} ({m['store_id']})": m["store_id"] for m in matches}
                picked_label = st.selectbox(
                    "매장 선택", list(options.keys()), key="support_action_store_select"
                )
                selected_store_id = options[picked_label]

        form_col1, form_col2 = st.columns([1, 2])
        with form_col1:
            action_type = st.selectbox("조치 유형", _SUPPORT_ACTION_TYPES, key="support_action_type")
        with form_col2:
            notes = st.text_input(
                "메모", key="support_action_notes", placeholder="예: 임대료 지원 프로그램 안내함"
            )

        if st.button("✅ 조치 등록", key="support_action_submit", disabled=not selected_store_id):
            admin_user = auth.current_user()
            log_admin_support_action(selected_store_id, admin_user["user_id"], action_type, notes or None)
            st.success(f"{selected_store_id} 매장에 조치를 등록했습니다.")
            st.rerun()

        st.divider()
        st.markdown("**최근 조치 이력**")
        # 등록 직후 바로 반영돼야 하는 값이라 session_state 캐싱 없이 매번 새로 조회한다.
        actions = get_recent_support_actions(limit=10)
        if not actions:
            st.info("아직 등록된 조치가 없습니다.")
        else:
            df = pd.DataFrame(actions)
            df["store_name"] = df["store_name"].fillna("-")
            df["notes"] = df["notes"].fillna("-")
            df = df[["store_name", "action_type", "action_date", "admin_login_id", "notes"]].rename(
                columns={
                    "store_name": "매장",
                    "action_type": "조치 유형",
                    "action_date": "일자",
                    "admin_login_id": "담당 관리자",
                    "notes": "메모",
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────
# 메인 레이아웃
# ────────────────────────────────────────────────
def main():
    # 실제 관리자 로그인 세션 확인(shared/auth.py) — app.py의 ADMIN 분기, mypage.py의
    # 로그인 가드와 동일한 방식. 관리자가 아니면 로그인 페이지로 안내하고 멈춘다.
    user = auth.current_user()
    if user is None:
        st.warning("관리자 로그인이 필요한 페이지예요.")
        st.page_link("pages/login.py", label="로그인하러 가기", icon="➡️")
        st.stop()
    if user["user_type"] != "admin":
        st.warning("관리자 계정으로만 접근할 수 있어요.")
        st.page_link("app.py", label="메인으로 이동", icon="➡️")
        st.stop()

    init_session_state()
    inject_custom_css()

    # 로고와 로그아웃 버튼을 같은 줄에 — 로고는 왼쪽, 버튼은 오른쪽(2026-08-29 요청,
    # 원래는 "관리자 대시보드" 타이틀 줄에 있었는데 로고 줄로 옮겼다).
    logo_col, logout_col = st.columns([5, 1], vertical_alignment="center")
    with logo_col:
        with st.container(key="top_logo"):
            # 페이지 최상단 로고. 클릭하면 앱 홈(app/app.py, 루트 경로)으로 이동한다.
            st.markdown(
                f'<a class="mp-logo-link" href="/" target="_self">'
                f'<img src="data:image/svg+xml;base64,{_LOGO_SVG_B64}" alt="서울 상권분석"></a>',
                unsafe_allow_html=True,
            )
    with logout_col:
        # mypage.py 회원탈퇴 처리와 동일한 패턴 — 로그아웃 후 app.py(메인 지도)로 이동.
        if st.button("🚪 로그아웃", key="admin_logout_btn"):
            auth.logout()
            st.switch_page("app.py")

    with st.container(key="page_title"):
        st.title("관리자 대시보드")

    st.caption("서비스 운영 현황을 한눈에 확인하고 관리합니다.")

    # 요약 카드 그리드 (참고 이미지 스타일, 2026-08-29 요청). 1행은 신규가입자+전체회원수를
    # 카드 하나로 합치고 회원유형 도넛과 2등분(2026-08-29 피드백 — 작은 스탯 카드 2개가
    # 옆 도넛 카드 1개와 개수/높이가 안 맞아 보였음). 2행은 카드 3개를 균등 배치.
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        render_signup_stats_card()
    with row1_col2:
        render_user_type_donut()

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        render_top_regions()
    with row2_col2:
        render_model_performance_chart()
    with row2_col3:
        render_dong_closure_scatter()

    # 3행: 지금까지 admin_dashboard가 안 쓰던 predictions/industries.custom_group 활용
    # (2026-08-29 요청 — DB 스키마 대비 활용 안 된 부분 추가).
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        render_prediction_risk_chart()
    with row3_col2:
        render_industry_group_chart()

    # 4행: 목록형 카드 3개(인기 키워드/인기 조회지역/업종 전환)를 한 줄에 나란히 배치
    # (2026-08-29 요청). 셋 다 이미 각자 st.container(border=True)로 카드를 그리고
    # 있어서, st.columns 안에 그대로 넣기만 하면 된다.
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        render_trend_keywords()
    with row4_col2:
        render_popular_query_regions()
    with row4_col3:
        render_top_industry_transitions()

    render_support_actions()
    render_user_management()
    render_model_management()


if __name__ == "__main__":
    main()
