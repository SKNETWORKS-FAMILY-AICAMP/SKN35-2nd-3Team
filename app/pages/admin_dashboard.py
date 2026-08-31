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
- 업종 트렌드 TOP 5 : 상권 트렌드 키워드 TOP N(trend_keywords), 30초마다 자동 재조회
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


def set_production_model(model_id: str) -> None:
    """models.is_production을 model_id 하나만 TRUE로, 나머지는 전부 FALSE로 바꾼다.
    app/shared/query_predictions.py가 프로덕션 모델을 "is_production=TRUE LIMIT 1"로
    찾는 방식과 맞춰, 항상 정확히 하나만 TRUE가 되도록 단일 UPDATE로 원자적으로
    처리한다(둘 다 켜지는 순간이 생기지 않게)."""
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return

    from sqlalchemy import text

    sql = text("UPDATE models SET is_production = (model_id = :model_id)")
    with engine.begin() as conn:
        conn.execute(sql, {"model_id": model_id})


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


@st.cache_data(ttl=30)  # "업종 트렌드 TOP 5"는 30초마다 자동 갱신되는 화면이라 캐시도 그에 맞춤
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
    매장이 수만 건이라 전체 목록 대신 검색 결과만 가져온다.

    store_snapshots는 매장(store_id) 하나에 스냅샷별로 여러 행이 쌓이고, 폐업 후 다른
    업종으로 교체되면 같은 store_id라도 store_name이 스냅샷마다 달라진다("업종 전환"과
    동일한 현상). 예전에는 여기서 그냥 DISTINCT로 아무 스냅샷 행이나 매칭했는데,
    get_recent_support_actions()의 이력 표시는 항상 "최신 스냅샷" 이름을 쓰다 보니
    검색·선택 시 본 이름(과거 스냅샷)과 등록 후 이력에 뜨는 이름(최신 스냅샷)이 달라
    보이는 문제가 있었다(2026-08-30 확인). 최신 스냅샷 기준으로만 매칭·표시하도록 맞춘다.

    ORDER BY도 반드시 필요하다 — 검색창/조치유형/메모 입력이 st.form 없이 각자
    바로 rerun을 일으키는 위젯들이라, "매장 선택" 드롭다운을 고른 뒤 메모를 입력하는
    등 다른 조작을 하나만 해도 이 쿼리가 같은 검색어로 다시 실행된다. ORDER BY 없이
    LIMIT만 걸면 같은 검색어라도 실행마다 다른 20건이 뽑힐 수 있어서(실제로 동일
    검색어를 5번 연속 조회해 확인함, 2026-08-31), 방금 고른 매장이 다음 rerun에서
    옵션 목록에서 사라지면 selectbox가 아무 안내 없이 목록 첫 번째 매장으로 조용히
    바뀌어버린다 — "검색해서 고른 매장과 실제 등록된 이력이 다르다"는 제보의 원인.
    """
    if not query:
        return []
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return []

    from sqlalchemy import text

    sql = text(
        "SELECT store_id, store_name FROM ("
        "  SELECT store_id, store_name, "
        "         ROW_NUMBER() OVER (PARTITION BY store_id ORDER BY snapshot_date DESC) AS rn "
        "  FROM store_snapshots"
        ") latest "
        "WHERE rn = 1 AND (store_id LIKE :q OR store_name LIKE :q) "
        "ORDER BY store_id "
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

# 상단 로고 — mypage.py와 동일한 로고(점포 아이콘 PNG + "hoTSpot" 워드마크)를 그대로 재사용.
# TODO: 팀에서 최종 로고 이미지 파일을 확정하면 이 상수 대신 그 파일을 base64 인코딩해서 교체.
_LOGO_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAJAAAACoCAYAAAAPb2d4AABlq0lEQVR42u29d7wdV3U2/Ky9p5x6q656t61quXdjJAMGAyEOAV06BBISgoEEQpIPAlwJCAlJXkJCXt5AgIQOV4BtQrHBtizj3i1Lsnovt5fTp+y9vj9mzpw55aq5g0e/rXNPmzOzZ83aqzzrWcCL24vbi9uL24vbi9uL24vbi9uL24vbKW30u3jSzEzrAFq5YQNt3dpDdwAA7gCwBpu2DTM2bGWsXUnYsJWBlYS1te+uXtFD169cw8AGrF27lgEwEfGLovRbuvX1sejvZ7m6b6OBtf3yZL5T/ZBBJ32HEQBau7Zf9vezZGZ6UQO9gLULAFqzbp3YtH69AlCnHVK2QLEyOHv3Qb9z277x2UcHc2eP5/wuloJKrqp4rpjuuP7CQknn2rKyLZtO54XQBbdSlrYk27aNvXNmtW2d1kWHlizoKi6eMWNP0qKi1oCr6ud29eo+ef3163jr1nW8fv16/aIAPW+1TJ9YuXId9faSjguMJYG9O/csuWPr6KWHxlTXRIGXjOW9VeM59+xCGVmfLcPXJjxNUFpDa0D7gAIAIcFKQRAgiEGCQEQwJWAKH5JcJEyohKUPZFPGwc60VD0Z2jd/pn33237/kp8nbRquuHVHKfr61ghgjV6/nvSLAvR8EBxmsZ7WAQjubssAnMmDc379RGHeIzsm1+w+XL56ZFxdVVGJZL4i4PqA5yv4vg9iDYCZSGgiMIUzQUREYDAD4V8I/gveZoC1YsFgYhAJYUAICUGMhCWQshTSCW9oRofYPH+G/cSiaeYTb33l7FvNzNxDflVs1vbLvv61vB5gvMDtJ3ohapttK1fSht5eBqAFgNFj+xfd8ljulY/tmHjb3iOlVWMl6ih6SVRcwPdcEEEFWgQsiAisRXjqFCgsqk0FAWCOnnNsBaxOFgdrJYjAADEjEALNzEqBiKQUhkTKEsgmFNqTbm52j/2L8xZn77v6wuytCxactRWRLPXLFVu3vmCXOHoh2Ta9vRvEhg29CgBsE7j5Nzsvu/W+ox/Ze8y5dqhgZScKGp6vQWAIQUoKgJhFcJtz3elS9JzDZxy+S3VGU3yCgs9QJEShfoq+VxMvZiahWTMrDWKQtCwLHWlgWsYpLZqd/M/XrJn1vZesOuNRomA5W7u2X65YsZZfaMsbvdAEh8vHFv3XL49d9sDmsbcfHvReOeGkjErFARErQwICgcAgri3qdQcAahCTQKMwtZqUQEkx1QtKs/jVnreYYGZA+4qZmYxk0kbWKugFMxOPr1qUuumD71jyNaJpR8J1WfC6F05ogF4wgsM868sbHn7vI09MfPjAIHXkygTtuxASShJEdR0hqi4xqK1S0YXl6IQbXyWgScjqnofSxfEfiO2f0HrZaxJHYtZMWimWJCXaMyZmtDnHLlyaufEt1875j9mzF24LluqNxrp1a9TzXZCelwLU398ve3urgjMy5wvfePIjj+8uvevgkOjOFz1IASUlAGZRExmaQgegtUCc7OzwcV+of41OZffMANjXYIaQmZSNed1+6ZwzU//10T+58mNEVK45Cs/fZY2eb1qHejcIbOhVzJz55k8eWfubh0Y+d2DEmDmZdyEllBQswEzxpaleNLj5rudIgTT+4pTTcCJxOxVZoYYlr/69QJ0phvIVjLZsAjPanIeuvrT7i+9de8mNRFQMApRr9fNRG9HzKWK8fj0xAL5546N/cOvdI/+w9YBeNp5XEEIrQ0BoBlXtF25x8Fx3wajh/WDJYW48eW7xWcSWtmZDuf63T6B96qxwqjvKxr+IwIqhGVJ2Zk2cMVvf8/bXn/GXl64640GuzZF+UYAatrVr++WGQOvIL3z93r77Hxv/5NFRgKGUISDA8VuPT7yghFd4KgFr/X2eejpamUWNBncQOELc4tahuX4cndu0o8CQY+0rsCApZ3bDP39526c+9v7F/0U0Z2R1X59xx7p1zxvbiJ7rmM769SsJ6FW33P7olTfeNvQPuw+rq0oll01DMAiiOs88xf2PuL8V3uHcsLRNtR5xzKjGKS5JcS0FqkaCTnZaG0S/xccDWWTt+SxSSQtnzaEDr3/VvN5XrD77gdrcPfexI3qOlywtCfjcF297xwNb8/82XrQ7tar4QpARD+Y1iw+3sJ+odlmoJnVEFPlF8b2Jqt9EseWu6v23EgbicPnj2uBWVlRt0aOT0Twn+AQRs9LQRKZsT3vjl5zTvu5jH7z6a0RU6u9n2dtL6ndOgPr6Nhrr11/tl0p75n/iH3Z8fMcB789KDkMKrQCSTRPMx7Ne6y0fbtIQtcgPUTCCC6/B2gdrH1orMCto1uBw1PYhwu8JkBAQwoAgAyQkiGSUA2HmqmNVUyF8grU2rl0pZiNRsyQRoH1NIpO2sWS+vvfP377wXWeeuXJXdfn/HRKgPgGs1/c99tiib3xn3/8eHrRWOm5FSQoCgITmeacGHdR8fze8W9U8JEIzWIG1gu87UH4Fyq9AKwfMHph1uATxFDEjqmk0QriqCoAkhDAhZALSSMCQNqS0A/kHh4LIkVajBlOpMS5el3BreWsABGatWRlmwpjZ7e540+vmv+/V11x8R+TBPgd2ET3bgcEbbniT+sa3br/2l7cPfnUkZ84TUD4RGXxKzjG1MmiiO59IgFlBqTI8twjPK0JrD8whukNrKKXh+xrKV9Bag3W9mqOqJuBYeIAEpCBIacAwBQxJIFkNRRFIWJBGEpaVgWGkIIQZJjbUcSIGdFJhBdQnUJRSkHNmSKy+pOuz73nn6k8R9Yq+vhXPek6Nni3hWbNmndy0ab3/b//vl+seerzSNzTsQkro4Go3G7cntd84mivcjfLLcN08XDcPrVwAGlozPFfBdXwwK1imRFvGRnd3BtN7MpjWnUF3dzva25NIp5JIJkwIKUAAXM9HqeQgXyhifLyMsdEChkdyGBjKYXyihHLFBzPBtAxYlglpUKh1DJhmCqbdDstIBfAQraNzo5gNdgrBx7h+0o6rMHtmm7jk3OQX//L6l39Y62dfE9Gz9BsMAJ/7/M/f9+DmiS+VK1IYksEg0Vqr8JR3J9XS4QAEBAlo9uC4RVScSfh+CQQN1oBT8eA6LmxLYPasdixbMgurVs7HsmVzMXfuNHR3ZWFZ1uncEpjMFTEwMI69+45hy5aD2PLkYezbN4xczoU0TCSTFixDQAPQZMGy22DbbRDSAjMfJ/p0nCmk+LkHcSPPZZ3NSnnpBe1f+ehfvOrDRFR+NuNF9IxHlqlXMPfb//gvv/jCw48W/qziKAjBDBbEFMY9porYNNrRXAvvBcuUj4qTQ7kyAaUrIACe66NSqiCZkFh61kxcfvkSXHrJEiw5aw7S6VTTMWrN0FpHg+OuVYAPioYQIhzU4lwVDh4axiOP7Mbd92zH5scPYnysDNOyYCcNMBEYRiBIiU4YMlGzk1oEP+ssMqpaZNw0NUSAp1i3ZZNi1dnWzR//6GveQESlZ8vNp2d232sFsEF99h9v+o8dO/n6iYm8LyUks6DWQX1ufUDxOy80Uh1nEpXSOJSuQBChXHLguhXMnd2Bq69ehVdecz5WLJ8PIY1oN0op+L6C73tQStWEhhnMVRO+JjSBpkAUvq5qjUCIJAzDgGEagU0kRd0hHzs2grvu2oZbbnkUW7cehlJAJpsESYKGAdtqh53ohJAmWD+160zE8BX8ZNI2zl1hf/fvPv669xCRF54Tv0AFaK2UcoP6+0/f9DcPb578vOsKTwod3Io0dSSE64Kz3OSGu34RpeIIfL8MIYBSyYHvuVh59hz8wXWX4+o156KjIxtqFw3X9eD7HnzfhwrQXpAyvPiGASklpJSR8Bxv01pDaQ3l+/C82j4BQEoJy7JgGGa4XxEKrY+HH96NG2+8B/fcvR2VikamraoJTdjJTiTtDgACmrkaQKzL9NdUEtdBTMIYZmRLKcV+Jp0wVqyw/+djH3vdH69Zs07cccczG7WmZyo18eOf9Kp//qf//T+PP5L/SKHgaSGZwnt7Cpux3mGvEx4SUOyjVBpFxRmHkIBT9lAqFbHq7Hl4xztfgTVrzoFhmKF2cuG6HjzPBTPDMAxYlgXLsiClfNrP1/d9VCoVOI4Dz/NgGAaSiSRM04Q0BGSoBXdsP4gffP8O3HbrE/A9INuWgmIFKZNIp2fAMFJBWKEuYUcnndwnApTPfjabMFadZ//HRz/6Bx8E1kpgg3rBCFD/2n7Zu6FXfe0rv/r9u+8avmliwlGGAVE3EyfyPKi2bAgh4XhF5EtDYO0CTJgcz2H+vA68892vwGteexksy4Lv+3BdF67rwvf94CImk7Bt+3g2WrRUnUgDncxnqlqqXC6jVCpBaw3btmHbdqihTACEzY/vxv98/Ve4967tSKbSSKZM+BpI2F1IJDrDCdAAU4C8DnNrzejHhihroLlY+dDpjJCXX9X9J9df/3tf/+Qnnzmjmp7+3NZ6/Ysb71zxow27NuXzsksKjTDydhxHvClaGP1RroyjVBmDlIxSvgyGjzesvRLvfPe16Oxqg+d5ocZxwMxIJBJIJpMQQpxQIKr2zkk4AyclYI378zwPxWIBlUoFpmlFx5VIWGAGfn3zg/jvr96MQ4cm0dGZhdYK0kgjleqBFGbo9oexrTpUZMMF5AY9Tsy+Bz1jVlpeflXXu972zmu+FcdYPS8FqArH2LNnz7wvff6euwYHvHmCWAMsIrBXqFm4BTCUGzKJzAqF4jA8vwBBhImxCZy1bBb+4qNrcf5FS+H5gYvuui601kgmk0ilUid90U9Wo5yi19lyn1pr5PN5lEol2JaNZDIJECGZTGBiPIev/MdP8fMbH0QqnYJpG9BaIJOaDstMBwY2tcrm1P6vBztFbpr2fMaM2WbxXX96/iUXXXTOdu5jQU+zJnraDILpm7aJneZ2bTnnf31kEJcBvk/EkijwEogAgg7/DqtZCIGqDqtbCAxBBK0dFAoD0LoErRTyuRze+Oar8KnPvhvzF85AoVCCU6nA8zwkEgm0t7fDNM26i3gi4ThZzXOyn53qc8wcap0E0uk0Kk4FhUIBUgoopWFZBta8/ALMXzgNjzy4HYXJClJJCcfJQ5CEZSaCeQvUOEQoNBQ9BvMrYnMaziUZAtotycT+vQNz7r7/hh/T1es4KOFe//wSoLVr++WGbevVZz/2nY8cOej9pee7vhDaqJ0oh3dLkHMi6HAiOJocQMMggvJLKBQGQeShUipDkMJff+IteMd7XgOtFYrFMlzXgWEYaG9vjwKBJys4p6KFTnZ/x9sPxZKtQojILisUCnBdF5ZpwfVcLF2+AFe+9Gzs2LYHh/YNIJOy4bp5AAzLTEbzSFGog2tzyDq8Qes/IwDB7CvPMVbs2L5z+u13fvBnb3zjCrlt2wZ+3ixhVaP5pu/fdd2v/nfHjwo5l0gGIDDUpT0bjpnqly1BBM8rIV8chpBAfrKAnllt+Pjf/zFWrlqMfL4Qus0+stnscY1jPL+rTCKhyuVyyOVyyGazkNKAZRnwfYV/+4cf4Jab7kP3tC4wANtqQyrRHQY5T+2SMYG1z7q9KyGvvHbuNW9+2zW3Pp32kPHUI83rmJnTH333f/1DJceGJNaka+JBLYBbiIc6wCCScL0CiqVhmBIYH5/EkhVz8cnP/xl6ZnZgcjIH33dhGAY6OrqfdtvlWU0+xo69ra0NiUQCIyMj4RKcBBHhb9e/E9N6svjhf9+Kzq5ueF4OFWakEt1BrCiGiqKa4RyLoVEcFUJSgiqTPj942/7PM/OVROQ+XTmzp7SEbdu2Um7b9gGdLJ316ZGj/uu1chUBsrZe12ARIlaLFasDhSBA+SWUSiOQkjE+NoFzLz4Dn/7C9WhrT6FYKMP3PSSTSWSzbS9o4WmljQzDQDabRblcRrlchmlacBwXV6w+D1bCwH13PoaEbUNpB2AN20wCHCxfgqKlKnzkaN5FKECBWcQEVlq5cs6Wx7Z1337vjT9j3Sc2bVr/3AlQFcj0rf97w+t2PT7+JbfsaSEhBBGJMCFNVD0pioQmfpJSELSqoFQahpSMibFxnHvxmej7l+th2hLlUiA8bW1tgefyW7bFb4ZUKhV5a7adQLlcwYWXrUQiaeLejQ8jlUhAqTIIhISRDGwgqs1r4xCRgR0Z2qQ8X3sVvuRjf/3Re9/3Fwt396/tlxueoj0kTnfpWrFhKzNzZtfjQ19wC0pIoUkyk2SNugGGhIoNDQkNgzRIOagUhyHJx8TYGJafOx+f+Kf3QZoC5VIFSil0dna8YO2dU9VG7e3taG9vRy43CSEEJiZyeOM7r8G7/vx1mBgbBYHgVMbhezmYAhCx+ZQI5luE8y5ir4XPyZBMTt7Dg7/Z+XlmFr3BNaRnXQNt27ZSfnnbB/Qce+UHBnYV36J8VxGRFJGXFfO4QqinCNGeorqGs0axNAKGg3w+jzkLe/CpL/wFkmkL5XIFzBqdnZ0wDAO/C1s1Il6NWk9MTsK2bVTKFVxw+UqU8kVsvm8rMpkUPK8EU5owyACxhgjnl6qeWQjJjXtlIph7YlZaV3jW7u17R36x6YP3AxCbNm3iZ02A+vr6xJe//AF+4IEH5t33v09+vzzp20KyECFZhYi5l/WPiD0CFWcMviqh4jhIpAx88l8/hGmzOlEqlgBodHT87ghPoxBZlgVDSkxOTMC2E6g4FVx8xbk4tPcw9mw/iHQ6Ac+rICFTkBC1EEldjKjheVg5IgCwz6iUndX/8l+f+v7b3/H2CfSdvhCJ09W4v/nh5k+onOiUpFjqcOlCoEKrg1hD6OodoiFYQbKG7xXgegUQM3zPwQc+9W7MXTwLhXwRzBrtHR0wDCOCWfwujWrkOp1Oo729Hfl8DgDB8Sp438ffjgVLZqFcKEGQj4o7BoHq/DJkNM8cey18rqPrQoJ9rfOUvrN/63sJxNu2raRnxQaq5rr2b9+/aPxw4a1epcwSLKKD1rED1wypNSRXH3WwjGkXFWcCkhjjYyN4wx+/GhevPhe5iTyYNdra2mAaZjSZz5EqOLnPPEMeIRFB62AuUqkUysUSlK+QTFu4/u/eAWkByvPh+yV4XhEGo3bDcoubOCZQghUIWrhORU8eLvz5bTfdtnLDhl7V19cnnhUNJA2Bm7+16TMqrzJSaC2gSUCB4ENABYN9iPA5wQdBQbCCAYWyOwESPnK5Say6fDl+/+2vQm4iB80+0ukUbNuGngpgxc+aRXvi343KeE4LEXtSm1IK3d3dkIaE5/kolxwsXDYXb/qz12FychKCCBV3AswVGNCQrELDWkVGdPBYNbZV4NSwJgGPdUl3bLnnwGeFpGc+El1NxD10993Lfvale58oDLtSGoQqQKzKBEcx7FMcrSog4Koi8t4EXM+DNjU++Z9/ixlzpqFUKiORsNDR0RGBvlrBLKYGofHUQnfSrBwnU6NKp1G/GtTxSCFOySNrfH706FEkEgkwM1KpFL748a9i82+2obOzDSbZyJqdwTJIx6nI5RgWLYhQIz3N9l71vsvPu3zN5du5j+lUk60nbaVu2LaBAOChX+64DgUYJkGBIcG6gUqpFc6AwPBR8XIQpJEvTOJtH3kz5i6ejcmxCRimQFtbW53AVCeRTwJcVcUqP9/ddJ4KIx8XXG7+npQS06ZNw/DwMJKJJFzPxZv//A+wd/Me+E4FwlTwhQ1LJANAGlpU2HIzc4QQrHSBrcdv3tELYH3vtl7xjKQywrC3Zmb779/yL3+kHB9CcF28PNI+rTArBJTdPEA+CrkClpx/Jq6+bjVykzmAGNm2NgghjqNFGsrfq1BOZggSGB2fwE9v3gTXVRCCmkv0GKgSwtQzZLTSPFwPGY0VLEaVFERhISKaSqZrtYgEIsD1PCycNxOvftlLapG9U1jJqvZQMplEOp1GuVyGoU3MXDAdr3n7K9D/7z9BT3cXHD8H2zRAJEKEA6bgEKirbyHfcTByYPiPD08e/o+57XNHTzXFcVIC1NvbKwCo7/39996EvL8M2tNSxOynKmIugj+gTlcq7cLzi0GNFvn4g/f8HqQhoEoKmWwKtmWfWHjidyhHTIRgYvzDF7+On996LxKWAV/5TexkEYXdSUMz6uveI43RuDZMWeLO1RwCLNOG5/kAE157zVXwPD/CSxOdvBmldRAXK5fLABjFfBEvfd1LcP+vHsTIvmFk0kn4qoSETMe0Nh9ndSWAWGhmJYrmvI3/edfrAXxt3Zp1EoD/dAoQbdiwQTOz9S/v+ae/4ooPSTUDsnqsxLGbm2uMYQRCxSuChEZ+IocLX3EBll+4HMV8AYYlkclkoE/TGBVCwPM9HDg8gJ5pXTj77EuQSGQgiKE4ZPcJClHrSGECnEDdK010VdX3mXSt2IapXlPVJccJUgTphagcgCSGBg9g+7bN2HPwcB1PEU7WBo8pTCEEOjo6MDY2BtO0kcyYePXbXolvrPtvECXg6TISwg6II6ZQsgGFMUWoYQmGLrs8unvg90jS17Bp/dNrA/Wv7Re9G3rVzd/52aXuaOUcX/lMoUVIDaZOPb1BEMhSWsFXFSj2YaYkrnnTK+B7HrT20Z4+yaWrhR0cv/kNU8L3FJYsvxwdnbNhSg3DEIDWYBAcj6F1vPK+eoER4I3RCs9TFSTdoPupybhgAIYgmEZIxABC2XFBlMSjj96MJzY/ACsEvJ0ygobrtVA6nUY+n4dSPkp5jXOuXIWzzl+Mg5sPoD2bgVLl0BaaivmoAVlNEL5yqTTKr3jy4ScXLTtv2b5TqSk7oQBtwAaAgIOPHHiTLDFMgobWsiX7AdckqooEclUZEArFyRzOv/YSLFi2APnJXIRd1iddE9XaA6LwH4MxMTYI1/WhtIdM0oQMuJuhNOAp3bAPilGocguweng7xBirqJFKOjxPSUF4gygoh/aVRqnkwLaTcMp5EMnaEk9PLRxBROjo6MDQ0DCIBIQhsPr1q/E/j34N0Ao+VZAge4oVjOrocap1mgxWRoXS9/zwzj8G8ImVpxBYNE7CeFbMnP3i2/7h95XrQkgi5niVJKH+aILDE0TQ7MPnMlh7MCzCla+7Cr6nABAymczJQ0i5fjaiuytg+obnKRAYG2/7QQQgEUQQ1VtMSmit4fmq/vqFVBmmYYBZQ2sVr7avuiqQIvi+r7zQ3hN1VaOWaUAHFPah0cvRe1JImKYF1/FCahluMLWqhvnJeXJKKSSTSSQSFhzXQ7lYwbKLV2DesvkY2XkMVjYNpR2YZIMRLr/crM4YtWoPApMqu5g4NLqWmT9HROWTNaaPK0AbNmwQANQvvvaTyzjvzgP7TCxF1bpBzJaoXxwYxAI+OwB8lAtFnHnJUixYvhClYgnJZALWcQKGx41Ch9dWKQ0pJW77zf3Yd/AI2rJtsBOZ0F0OdJIUAsp3Ua6UYBoSM3vaavZWaBRrzRgbz4GEgVSqHXUVeyBo7aNSLsI0DXR3tkNrHWkaEgRfKYyN5yGFRCqVDb+jYzUEGrbt4LY7H8Bb3/BqdLW3QbMO6Wda0ALzyRjUHGCIhkZAUEikLVz0ykvwv1t+COIkFCqwSdax6texS3KsKj8gHRFa+YwCLbnl2z8/D8A9G3o3SIRtQ05bgLb+360EgMb3jJxrOAyLSAEwuIpx5jonrE5ZCFKo6ApAgKd9nP+KSwEhwFojlUq0JIM6qZUrdN+lkMgXivj6d24Ca40LLliDFWe/BJVKKbC+tEYqmcID9/8M99x7K979lutw/R+/GZ6vIGVgdxlC4rGtO3H9334WixetwDWvfEdMS2kYhonx8QH85MdfwZyZ3fjaF/sghIixhwGO6+K9H/4MDh4ZwDWveiumT18I1ykHmgiAZQjcvrEfm594BP/9/Zvw1+9/F5SnAloYcGsCLRyH1jUUykQiCcsy4Xs+KkUHKy9fhd/M/BW8ggtpE4j9YOms+40pvLNA3rUssZzYObQawD1PSypj/ab1GgJcPDa+mhwPAkzVhKjUNeyJ1CrKvUjWMMBg7UKzD9dx0D13Gs66YBnKxQos2wy1D9f5O9GghhFzfqqf0cyQUuC7P/4ldu7Zj4ULzsCqVZdCMMOUFgxpwjITcN0KDh/ag6Rl42VXXQLLNJFOJpCwLCRtG5ZlYnxiEhXHQSbTDttOQgoThmHBMCxIYSCdakcykcTYxCRcz0PCtpC0LSRsC5Zpoi2TwRUXn4t8PodDh3bBNGxIaUEaFkzDRDqZxgXnr0ZHWzt++stN2L57P0zTgPZVvScWP1/U/m5ld1fDEel0Gkpr+J6Ptp4OLL1kBZxSCYIYmj0Y4XVqwmiFiVcZJWAZBjTpioPc0eHXM7Ps3dCr+STub3G8xCkAvXf33gV+rnSF7zsgUoLC/Er1EVAB/oRV3fDhAYLhVYpYevEKZDrb4LkOEskEBIlYBjqWV6oO3fy8SiGntYZpGDh4ZAD9N96CdDqFc85fA8tMANqDFIDWCmQaGBg8gMHhozjjjHlYvmRxXdZbKRWkCAaGAQ0kEpnw5/yQECqwiUzDQjKVRqFYwOj4ZPDdkJBBh8e15sqLkE5ncGD/DpTLOZAI0jCmJDhOGbNmLcaZS85FLp/Hf337J1GUg2OkDeGdUTt3XTtv1g0Z+3AeqoWKzAHJ9Morz4EwASgfGm4IpdFh5YsGQQUjvEYifB4mWEkrF85E/pw9e/YsBsDoY3oqGkgAwJab7r7WdHQXWCmpNUkdQDRk+MO1TLuOIBvECj57ABSECSy9dBV834cgIJkIXMwpzTNuCApz4xIWuNxf/+4NGB4dxRlnnI2585ajWC6DJEEIwJSBEX3k8A6UK2Vceen5sEwztF+obgwOjYFIIpVuq/O0qqTShmkincqgXHEwNjYRuukISRoC4OiKpYuxZNF8DA0fxcDgIViWDVNW3WSBiutj1bkvRU/PDNx578O4875HYJpmQMzAHC3Lp2IHVVMctm1Ds4bnuJi9dD66ZnfDd8vQpMDshRAaBanDa6WDpKtkBdKx66cVCWgWJc/efdtjc+Ppq9MSoG3rtzEATB4cupJLFTYExzK6obCAW2R8GSAPIA/KddAxoxuzzpwPp+zAthMwDCMyMk86DEK1OIhlWXjo8W245fZ70d3Vg3PPWw1ogq8AxTWguufkcOjwbrRlMrj6iouacmlVT29odAyGYSKd6UCV2TlurAphIJXOwvN8DI+ONxn5gZYycMUl56FcLuHwoe1hPCj4DaUJruOhu2Mmzj77cmit8F/f/gnKlcoJiR74JBAlqVRwQ/q+j0Q2jbkrF6FSLoOhAHiQCDFCcehrbEQICmgYxMrygdyewYsBYOvQ1tMXoA3YoIVtADl/ATxF9QDuEBgfxk8CuGSQH5LMACsIAXhOBbOWLkCqPQvf85BMJUPvAyf0sqKcV5xpVxA838dXv/VjOK6Dc1ZditmzFqLiVQAieArQYJiGiaHB/RgdHsSSMxZhyZkLo9qzqlNSVf1j4+OwbRPJZAaqwSvk0I1Pp9vBYAwMj04JjH/pFRcik0nhyJHdKJcmIYQBXzE8FezDdSs4/5xLMW/uQjzx5C7c8IvbYRhG8JtUf94cO3+eCjWPIMKeSNgwjIA+T7PGvHPOCggZoMHsR4A+yTGgPWrA+xDqGthCBNKOi9JE7veYWa7ftF6dlgCFQGtWFW+GVyisUH4Fgn0htEJt+IHxrH0IVpGqFKygyYcghu97mLtsYVhuS7AsKyK0nBKVp4MBHVv7NcNXPkzDxC9u/Q0eeOQJzJwxG6tWXQbtVyAJQctKpeH7CoDGvn1PouJUcNlFZwe8Qq4X8PsoFQ6NyVwBI+MTSCSTsO0ElO+Fx6FDyt9A62Uy7SASGBwehVIaSnO0H2aGrxTOWjwPS85YiKHhYxgcOgAhJRzPD6LhWkOQj0QiifPOvQK2ZeC7P/4FBkfGAo2sdHTe1XOPD24xAvtLQwgByzLBYHiuh5mL5yLZlgy6uIiwCphD+0f7wfXSfoDZqj5y1QnyhfYcqHzxgr1Hj84BwCcCmonjxH9w949uXoJiZZqAZgkmCYYkHS1VwRJWW9oMaAjSYChopUEJE9PPmA/XcWFaZm35iuANJwfz1FrDkAYmCwV884c/g2lauPDCV6CnZw5M00BHWxqpZAoJOwU72YZcMYdDh3aju6sD1716DWQ4yVUyKcOQkFLA8TxM5grIZtowfdp0WHYCdiIJ204hkUhF9ew9PbMgpYGRsUlIKWCZNWIqKSUMKWGZJl73qpfCdT3s3vMkSFqw7RSsRBLpTBLZbAYQhGUrLsOSpRfg8NEBfKv/p5BhKiduIJ8c9JXDgCXBsm0wA57rIdvThY4506FdL7y6qnaNSNeuH9euowyNbcFMgpi5UEkduefRWQCwcuXxo9LHjQP5I/lppuNDExis6yqUKRbKrydB1iChoV0H2Wlt6JzVA891kUmnIIQI+pUSnRJGq2rXfP8nv8S+A8cwrbsbg8OHcOttPwYoKFpUGvC1hpASuYkRVMoldHZm8ZOf3x5wGnJg0FaBGZIEBoZGoZng+x4efOBX8JSGr+IJVoZlmSgVJ5FOp7Fn/2H8v29uiHGJh1X9HCRRR8Yn0dHeiUMHd2Hjxp9EhFeGDIglguygAa0ZHe2d+Nktd+HVL3sJVi49A47rRktsXca8BUaoER5lmTZADK0U7EwC0xfOwb5dRyApMBlk1YulRsRlM7E5s2bbYyEm9VwA959WIDEMIKIyWlgifQWC1kFRRSMNZDy6GQi8BwUihvY8dM2eDzuTRKXsBGqWW0DjTpDKqHobxVIJt915P5JJG8ViDo88fBuIY7RDMR5F20ohkUxjcHgcX/zKd2tljVRNTwSBxqSdwLRpnSgUcrjzzp+Doo/JKNelWUNKE9lsG8Yn8vjif34fmlXUxgNUy/pLIdEzrRNK+Xji8U1Qvh9qgbCJb/VzhkAqlUW+UMatd96Ps5edGXABSQlqSt/UY5saQXda61CjSni+Bwaja/4M7KfAdWdiiMC1qeNUZBDAOnxEhOUCMUvPR3FgaEZcFk5JgLZt2sYgoDw8ukq4gSVPEVl2YxMlHZ2sAMFjDUkAKx/tM6dBGCaYK0HwLFq+UOfxRExhDTzRFBOqSsWBU/FgmAZWnf0SGGYSSmuYhkBQ/UNQmuH5wMF9j2Fg4BBeueZyrFy6CErpELFYa/kkCNi28wB+tfFuzJ41H1dc8WoopeB6Cj4TpCCYBkFIgVKxiK1b7oNtmfir978DmbRdIyasch5xkCK4+Y77sGvPfpy19CJMn7EAknwIEeTHKm5wZqZpYGz4IDY/cR9KpUoTZqkl+oD1lDghIYJ8nlvxoJVG5+zpEEbAckZEkCwaejNyLAalGznzWbgeyqPj54NCWTgVAYoSqJrlDe/99NlwHUgDohYa5eYWWLHGbhS2DWCt0T6jO7jjKGB3j5euNCX34vCD0OOrvq21RjJhI51OYnhsEmeceR46u+fCcx2QIKRtCSECQ7riCRw79CTK5RKWLJ6Pt7z+1VOe/A9uuBk3/OxWWJaN889bjZLjouIGNVS2aUBKgpQmjh3biyc234WKw3jtNVdizszpU+7z7gcfh+OU0dkxAxdf/HKQKkEIiYrHAayENVKpNB598JfwPA9tmXQt7EVNVNAtQUONuUIigiENAEFAMdvdAStpgrQChIRkFQ9h1++HAkGL/aaA58PNF1eRZWKDc3x+xeNZ2Fny/B4oDYMRlOnEQt9N5Ts6CCKCNMAKJIC2aV1QSkGGRutJlerE82pR4lQhlUph6RkLUCoVsHPHQ1DKRamcR6lYwNjkJAqFHAgaI4M7MTx8CG3t7fjlrb+B6waUvo7rwvN8OCGHoq8Ufv7r3yCbbcPg4CHsP7gDjqfhVPKAqsBzSygUc3B9F3t2PQZmH47jo//GWwKN6DhwPQ+e56NScQAAW7bvxuZtO9HW1ob9+x6HWx6F6zqYzOcxmcujXMrBdSrIT45h394tME0DS8MQAxOa+LBPDnEWmAMyLMLUyoedTSOZSgCeFxV2Vq8XaR2MMI5X88DC6wsGlA9yvRm64mZiXvkpC5DmsusLVEuTa6XK8RHUHqH2GgUHJyUh0ZYNItAi6HTTMpjRaj7i6Yxq5IkZ1778CliWhR07HsT+vZuRzbYhkUhCGgmYdhalUg7333cLfN+DIIHdB47gS1//HoQUsC0LpmnAtiwYhoH/+7XvY/uugzCkhOt5uOeeX6JUmkBneweSqQxMOwE70YZ9e57A9u0PAixgmTZu+OUm3HbX/UjYNizThGkaSCRsDI+O45+/9D9wPR+aCaOjg3jwgVthWhbISMG0Ukgm25CwbTz+yK04fGQvlpyxEJddtAqu60KA6lMZPEWArGFU636rQUnlK8iEBTOdAPtu+K6uqxsjZpCujarwBNcQRFrBYJEEYJ66Eb1uXVWLZoTyswQPAkY9HJjRCBKOnAURRj6lKWGmEtBaQ9pWyHQU8LlQS5BTHCoa9z4YQhDKlQouvfBcvPUNr8E3vncT7rrzBgwO7MfsOWdCCBPjY4PYs+tBHDpyCMvOnIc1l52NH/7vnfjOj2/G/oNH8ZprXorZM6ZhaGQMP//1Xbjz3keQSafwluuuwp0PbMX23Xtw6y3/g5UrL0P3tFlwXQcHD+zGrp2PIl8s4g+vvRxaa9xwy/3o+/xX8OCjW/GSS85DMpHAk7v24Yaf345dew/jnOULcdUlK/Hdm+7Egw/fjdHxESxYeA5S6Q5USgXs3fM49u3bCkMa+Is/eRuymRSKpUpY9tPUJyhGenwcBcQI0iqEaL7NdAKuDgK6RBpGuFC1pHLn+mshSYOLJf9ksNFTuvEDhwa6yPczQgceBzV4ThH2J+bW65DOhZhhGhKGZUIrFU0OQ4FiBh0a+MuIqgZ1i/ZNACrlCj7wnjfDMg3033Qbtm29D9u23QvPV/BcD1IKnLNsId7y+y/BnBmdSCYM/OiX9+COex/FHfc+Ass04XlB8G/+7On4w2svw8XnLML82dPwg5/dhR27D+G226rud8D/nEmncN01l+Laq86Ngowb792C72y4BT/4yW0BZ7XjwrZNnL9yEd563UsxZ3onEpaJ/p/fje07nsDOnU/AMgOOaqU0pk/rxIf//O244pJzUCyWIKRsMGa5zpWoOSot4h4cRFgodCG1DsIkVspGiVW0fBFrCGrotNgCR8IIqlzhVFJHjx7tBJA/JQHaEAaOBrZumSa1MhUzE3PY/YinKFiruZqSAKEY0pYQpgwYtcJSm8AG0g1lOrV+lhHWi1tngpgZjlPB9e95M85cvADr/+mraG/LYlZPOzKpJJafOQerls5HKmHBsCwsP2MuPvRHr8OjW/di/5FhVCoVZFIpzJ8zHecum4+ObAKGaWFmTwf+9M3X4Mk9R7Bz71GMT+YhhMTs6d04b/l8zJreDtu2AGi8+qXn4dzli7B5+wEcGRiD5/voas/gzAUzsfKsOchmkjBME2efNQdz/uR1eGTLXuw9OISJfBGHj41hwbwZ+LfPfhQzpnehUCgGKZUohcInkRHjFlRvFJEpVOM9hm0GqYqQ3FRwqx5IDG7RL4uVrxOG7BzcsmslgINVYOEpaaDyWJkRZtAR2jmNy1V9p/bARZQABDGEIcLyRxV1CZwSLE8nX7LODCjfx+wZ0+ApD90dKfzZW6+NiDwZjFmzpqOzow3HBoahlMLqi5bhqouWRTElIYLj6u7uxKyZ05HLF3HoyADOWTof5y5dAKUDA9M0ROA1pdNYMG82NDMOHj6KWSDMmR5U0SqNsE8GwzAkZs+eiUwmjcOHj8H1NdZcthLXXHUedu8fxD995Udoa0uju6s9Eh48XXX8VIvHMQDDMAMbJyhlhgxXiEZ4a4Rbi5kmHJIxaM8Tp72EzZw7g/NVAWA6bsvGaAkTGpJkSL3GtbhDpKHEKWDmp+hbSAQhgtxWgHwUAcKRNdLpJGbN6IZtW1C+j4XzZ2HWzGkYGZ1AsVgMllNpIJ1OoaurAwnbhO8r9HS3o7uzDWPjk5jMFeCr0NAUAp2dbcimU1BaI2mbWLH0DBRyRYyOjaPiOGAGLMtGe0cWne3ZoN2A1jjrjHno6S5gYHgcRAzTCBZrrRiu559YeAgtCDVbh+0jaB7roJVn1AEo9LQoAJGRjvchoZYgxeAnNcF3tZ1KT5yyAK1du5YBIDNtel6zUhIsAR2WEtWA2GCui2JWT08QQwgGsQ+tNDR0HT5sqvJk4uPVqlPd/HHYv1SzRiKRwFlnzIdWHgzTxJFjIxidKEAaEsQUtWdiNqBZAgpw8g5GcgO1sAIjgl+ACFoBbdk05s7qhud68LWCEBK7Dwyh4nowRIA30jABAhwFDI5O4ujQeC2CrYEFc3qweMEsmKaJYkVF0fCgVZWeooyIWpL415b2mGldtRlBCKwMCueZoX0vuBahBhLxtlHcylQPxSqElCutvHQmPQIAa7du5VPxwgAAJUkTmn3HIk7pxkrHFkGusLsDBACDBKAUlOdBMkErFcoNN2uZqAluC8+uoXwnMpS4PpLqKwVTSuw7PIgPfeI/kC84oKrhyRRiY6hWzRr1VA32HWTzg71JEhCmAcMg/PMn/hSrlsyHNC3cfMcj+Pt/+y6kYcHzub5eLCIMlZEd6PsuXnLxcnzmr98FoVQYqmuFkENTr9bmZYZaIu44VujIQIheCHBMyq1ACIIghhHGd+pb2LRqfc6RJiMpJSeq3ILrpiQnn9KIzo8dnZ2wrSRrxTJITqBqSE9lzghGEIwSBOErKMcJQFXKj7oht/xmExqv2UOIKkW1BmsVO9lAoAxpYGQ0h/GJIto7O2BlO2sJEQomxRKAJWVwMcM7l4RA0fGRNAASAhVF8IqTGBsewdHBMZy3YhEECRw6OoKyo7C4y0S7pUDCCJKfETSVoooMxxc4OM44NDCKiuMhYSXDAsaaMxH1wThBFeXJlZERtA57v7KG9nyokgMpBSRRiPmpMqZwywJNbqisISInO7OrfAL5mdoGYsAwhCBP6UA5xn6XohB4vQZCNVotCPAV3GIJaQpKX6YsJudT8zy4SZ3XXjCMgDMwM20GLn7re4MKi2oinjVWTk+jM2UBDCjNMARh50gZRcfFJfPasXO0jCNlYMuN38Ho4GDA3RwuCZYp4WnG6jNNXLcyjf3DlUAIpYTiIBMOALM6Mxh3BT7+kxFYhh3kwdBcVcIniUPkk7SjddhAD0RQrgdVLMOUEgYAQwWciRFKntFUFRN/NAB4Wo339PQMnOgwmgRoa7jeTTvvvP2j+HbOArcprTig2aOascL1WWIKI8iG1jCEgFAK7mQeRATP80+DcYyaBKe+dIibBauKMdIcwFvDN3ytMb/DRsKSKLo+LEGwpcBQycORyTIumJPFUNHD/vEyrEQSVZBgtNwxQpAZIWUZeOxwBV/fNQ2dM2fC91zMaUugPWlitOAguedRXLcq6PulwU2tK1t44E+Rhyj0TJUP5SuQIeGVy9DFIpJSwtAh6WaVH7Fayt1K6YerAfk+ZCZhAkicchxo3bp1vH79esxKJof2ZdJjxNxGzMwcdP2MDiD2w4F2IhA0DC0hhYTUCuXR8QCGWnEaMvE0RSfiRiwMNSl1bjIw6xmVKNZNAgg0TXvCwOxsgIb0NEEzw2eNXSMlLOpKImEKPHqsGCRxRQCdbY4uUFSQV/EYsy9+GeafdxFMVcG5M1LoSJrYfGgS279/AEpXYnHd5iWjNnGtIs3csIzTSeXCfM+H1gqGMODkC9ClMkzTDG5orcLOExwLwYRiHbJiBBXFOmiwAQWdEMbB00llUC0M7ErbHpSEhWDFulofHouY1hMrBDQYhgqappiCUB4eAaCDlpOeDxkhErlF/wjUt3NsMpGaxKi+JCY2+dXKiQCjA0zPWHBVNcLN0JDYOVKGJQXmdiaxY7iIoqdhCKo1vm3MZXINRA9WUE4Ztl9ERmoMTuSxd0jh8GgJpqwx9DdSToRlxKEdFEc0cOslvar+jitEweddzwuDthLO+CTIdWFlbBhlL0ieSmoZJuEwjxm9oilIeyST5flA7vQw0QDINhXZomgKiioworjClAOQKghaJUwDlcEhKFfBV4yK4wR9QXWgRuMDiD9H5IrW10QFBiJrQDO1WBJrUE8OL5hmRk/GghACJV/DVQG0ZLzkYrzkYXa7hSOTFQwWPZhSRKsiM6J9UGM4gQNvRzOhJ2MjawlMusBQSYPJqBHsMrWIaMXPtRkHXvWiglHdj2gqm6sfgQHtVBywDm6e/LFBmNqHKQmGr8IsfI2zuzqomiSPdU4SUNo0JJLt2ScAFPsAcbwa+ZYCtC5AZ4E70hVhMIT2YUDDZA0D8REQZxocDmgYngepNWzTgjMyBiefB4hQLlcgRGPBl25RAHY8dzeWxYknALmev0dQsHS12QLdqQBC6irAVYyCyzgy6WB61oKvGQcmgq7PteBbY8PbOJ8XR1F1UwbHUHZ95MpuaKTqpjwWuD7n12yScguGtPjcHH+OiAha+XA9N/RSGaVjg7CEgCTA9Pzadateo/CaSYQ1Y6xhsIYJDUNpNk0Tpp14gIh4zerVpw6qX7N6tYDSsHrat8uEBaE0B+znuqEKNXxerXwkDUP5MHwF2zTAEzmUh0cgTYlSqVg3OdWe6YFrXz+47rkCczDqPkf1k85MYcM1gg6Xro6UCdfXkdD5TDgy6cCUAl0pE4MFF55GQ5KxqV1Sna2hEUSa2TDgSAtjygLMBKxEEsK04SkVdtSp1SVRg8XLoVBwnZDUzi/StlHoo9WozZ3neXBdL7I3K4ePIGmaEErB8LwAmcghc2soMIJ0VA8moqpVDWKfYAsY07sPBsKw5tRTGWuuv56xaRM6zzrrNzn79g8L5QmWdtQZmBsTYhyLabKG6bow7SRMt4L8wcPInrEwbJyiauXMJ/DkeUo21pCRNaLzbxGQY0Y2YUBpQokBSQxDCuQqPkqewtwOG+NlDzlXQ4p6Ni9qFQuPcRsyM6Z3JlHacjceObQFjhsEMYmAQsXDGf4kSCRrmoNR13s+UEoqSqDyabAO15bsAAFarlQCHJBlojw8Cn9oOMA/+QoyLGIgRgs8Otdj25lAvgIlbVbdaQ+nzQ8UpjOyZy46Op6ySWpNHCQkqCnQ1xhcZIbtKrhCIy0Jud17MXv1FXA9F5VKBclEAl5jZcbxuoAfHwZTt9RUnxthFUbJ9SCIYAgB+ApjBRftyaCYb6jghLZOzbjUzPBjfHjcsHdDaIxVCO5IGcvlPqzusWFChemJIHTQlbUxXObmbrjUAv//lLdgFoIO0T5MM4HCwcMwCiVY3V2wciWYSgEhmC+6SlzDhtct1MSAUkKlbTIXzjkMAMMrV/LpJFMZAJJLFh9BZ/uIwXqazzpsktJKR4SspSJwSE3HhamTSNkWRvYfhJcrAFIgn8sjnU4Bvp7qXo8B7GseVb1Ho0PiJKAxyV9Nl5AUMFMpKOVH+aei46O93UZHysRwwQHZEomGtUuzDhg6DKOJZUUzkLJN3L8rh95XekiUNIbyFi4/I4GioyEEQjAWwweFVRt0nJCFPs4dwieAKVRbogO+8kPiTQFmgeLuvcgIwCCGUXFBUVCL6nBWdXyP1fwdERuCqJxND6SvuOKJMDeqT1mAiIj7AZkW4siW93/sl2nLfkdJ+woQRl3aP0aXBhAoLHQTngfT8ZGwbdDoOPIHDiGzbAkmJ3OYMXN6aPjGbkuun9w6gD03MPxVnQjNTVpKh2QIXjGPHXf+GlrVek94SiNtSewHkHd8SKqBsOLa0zANuJOjMEwzYPmINcZzfY0lM5MYOeTB0TYuWGjB8YIigKr2k7JaXBB3z+vJpDT4BHBnOo5KrtF4SRIolwoolx0IQ8DP5+Ds2YPupAVTKViOCxmxojXmIXVECF+VJ6GhbduQPHvmpiWGHA49MH1acI6e1X2ETevJXLzgAb7v0XeInEPaAISO0bLF23ZTNcUBkGLYFQdeykJSaYw98SQ6Vi5HOV9AsVhCIpGA53mN2ZDjKHZugkxzE40HYEgJx/Vx9OBhHNj1nXAVqWfn4rrgY/1ZVAFHViKBiuvDNGQd7snxGGf2mJjf1olfPFnBr5wyXL9aysAo+4QzewysmJ2KJSlj3mRViPTUoLmTY9wMBVAAuVwevucjYaeR27UPNDIKO5uFVfZgOl4UFG2MqUUtMoEwiEgg7bPIZJCYP/d+KE1rVveJ9SdgbZ1SgNZcv5KxCazPnF9QbSmI8TGQIeoBJBzPGIfh8TBxlyg4cNrTyCYsDGzbATeXB6TE2Ng45s2bA9dtZqU/YRvtWPlvI91vxXGxZNFs/PX7e3FkcDTkPawVKtIUOLw6nsLws0optGVSuGjVWSiXHWSz6UhTCUEYLZZxRrfGy5akUHKDjskSwLgjcNfuEpbOql+8Ig6kuuAnN3OeT9Htear3Pd9HLpcPCm7ByG1+AinWsKSEXchDKgVELGU11EMdSTrFcvLKF6otCT1/5kEAHMrAaVLchYZ05YyZ2/z2tE4qLTiIdNXCSsQN9Sg1BWxWXFgVF6mECWN4CJM7dqL7ovMwMT6OmTOnRwzs8YlqFJiWkxcnpgrTC1UNLQlY+9qrwriObrkkUH1Hj5YWezV2XCpX4Lqq3h5mAMJAZ8LDjLRCXjIEBbaPKRgpA1Fciut1Rh3+6EQdmOM1dK3mRUqJyfFJOI4DaVpwRsdR2b4dM5JJWJ6CXSgFGArdwqBvCSQjFlqL4oxON7Hmwq1xGTjd2ngGgI6V522enDNzp715xzI/8J9rKw9zvaEYLaoEoRWSxTK8dDs6BDDxwIPovuAceJ6HsbFx9PT0oFwu1yHzTibh2qpJjiBAh/vJ54stQAo4NWO1ihMKwWiRDcOAjnW7VzoIWHJINurpppBhBGirliZVQ92nsoS1irprVhgbGweYIU0TY1ufhDk6jlR3NxL5MswwLsRQsZu8BiqjxjSLZm3alnRmz9y4NDttR+C/nLjxijgOHzHz2rVyEVElc/by/xbtGQjfZQoDWRTR+WsE3JsqMJCCIjGQYFiFEixfoT2bhtq9B/n9B2EkEhgaHI6oSWpZdZ56QjnqbQAdUq/oME/EzHC9oHDQcVyosGjQV9XHKp2LXzfi7/vh8+rryg+G67pwXReO6wQ5sDAE7uvA3in7IhoVJVDWBioKERNsUATgwnW9GqQlBHwFfD7HA8xzg7nAUbpGSgOFfDHAVRsGtOMg98CD6LBMmADMyUIQ8qSqHRZesxCTU5/SCNGYymPKptmb2fMAEfEdq1fLp0403t/PIEL2D19zy/gNv/iMPTlmaWkwcUTwXmNp5fpGAZoI5HswyxXYXRmkS0WM3vsA5r3pjSgWJjE8PIqeadNQKpchRX2pT3NbEIpcd8s0YJsGDEOCSMBxXOzaewDK98IGJ1RXwdsCunTSXZ8oZCnLpBMYHZuAFAH2Z1G3iZ9udjFSdKG4Rtbk+Aqz2yUSRvBjnuth7/5DSCcsHDs2BCECxGLAK01gzfB8v24pJWrUOvXM8lUbcHh4GKwVDCuFic1bYBw9gnR7O4yKA7NUBowQlUB1uIDQ/qG6dg1aEMBKlns6SJ636r54MPmpMtUzAEyfPn378KK5h1P79y3WQrAODQ+m+AFyHVCbQ4yxmSvA7UijI5PGgS1PoHjlZbCmT8fgwAC6ujohRUDyXUdn0gBzCgQiKI3ef2gAtm1jcGgMhpTwfB9HB8fBSjVH7BpKFlu2WKIa72IdkiQWUkjmK8gXK7BMYKgIXJww8Jpz2moArXgE3AYO5zSENKAU48jAOJK2geGxPKQgeJ6HA0cG4Xs+kskEerrbY5Q3VJfja7oYGjBMExMTE8jlcpDSgPI8jN19NzoNAcswYA9NBASh0qiHvtSg7DGkJldPVRsCYmL2jL1dL1t9b2j/6Kel4Rz39Qn69Kf1ti986Rtt3/vRuyuFomIpZONd3artEYUQ1MrsHpTaMzg2PgbnwguxoLcX5UIeM2fOxJw5s1EoFKIONq2AVUprZDNpfOVbP8F3f3wzspl0sAw0BBDryxRbAPiZpw7s0VSFe1RPshLl/hp3H28LRdAIqVpYR92LGPVYKq18/OMnPoRzV5yJUrlSYxCZEuIaEHvu2LETlUoFdiqDic2bkf/BD7CgowMpVyF14FjQWpRqTP5Thkqq7ayUr6x0VrrveOunzvzQez/Da9dK2rBBPT3detatA9avR/a66/4tf8ttb7PyOZOlEWYvOVZpwY3xwCA4BYI1kYOTTaEzm8bBJzYjf9FFSC2Yj4GBY+jq6oJlWfA8N+x11eJkmaAVo2daF+bPnYVkMmC5pxaBx6licHGEMVEj2rGVJ8n1tmesZ0bUpJGbwwP16EOKYZRi5nVEMUNIJm0opY/jRNTsKds2MTAwhFKpHBCJlksYu/NOzDRtWIYFe3gIgjXYkK2Dh9TApgKABZg8JYozeryZb7ju5/jQe6lqujxd7Z6CtMaiuftG584ZtfcfmMUSHHD3oS6VzY0rCABtCJDrwi4U4Xe2odupYPiOjVjwjndAa8b+ffuwfOVyuK7bupkKBbGXUrmM33vFlXjFVZfUXHRu3XSLKR4ijDsgDY1yG5xHjocWqy82tJqghh5p8SRpc4VFxMYT6Z4acjTAWUsiVMLig1Z2T42hTaJSrmDg2AAECMKyMXLnb5A8fATZadNgViqQ+QLYEFEnouYbihoJ6kEEloYkd9Hcwx2ze7afanRTnER3GOa+PjFNihwvP+NW0dMB1korIYKDlQSWAMugPoYlQYeDJUEJgCXBmMjBVhqd7e0Q+/Zg5JFHYCbTGBufwJHDR5HJpGPNThrd9mBCHceFFEElqCEFDEPAkBQ8xoYpw9dD/kJDyqBWv/p5STCkgClE9GjK6vcQ7jP8jJSwpIQlZPh+uM/wfSPab/ieIWAaAqYkmBIwZMBbHewreF5tDaWVgut7UySWY+liAqSU2L//AHzlQ5gmnKEhlO6/B90d2eBYJsYD50UKsCBoQWBB4fUhsBTRdapeGxgBG4Mxs4d4xdIfEVGJ166Vp9Kx8KRqazesXEnQDPvVr/pyYc4MH8oTLAAWcTpaEWgjUX0MTiAQcQnyPVgTk7AMAz3pJEY33Y7y8CDsZAqHDh5GPl9AIpEKyPTqEpEieh5Qzwlo3TgCKIJmgmYRjuprgNYUjoAaV4evK64+ZyhdhYqE+wnTDUGsJ/5ZhJ9D9BkdoSQDQ7eKuqz/Xa69Fn4u4HSTDUnXeiLegMo3hSNHjiKXL0CKwLYavu3X6HTLQafrYgmyUIQ2ZExthwJE4RAIngsCwmunBTGzkrnZM3LT3/eu/44876e77Xdvb6/qA8SKc865DwsW/thOJylAlNUOrHpQYXlqJAPBawwyDIhcHka5glQmjQ6njKGNt4W8hISdO3eBWUNKARWCsqrYH806eAxHI+AqajsQUt9ySCVcoxSOA9ji8FGOPluDmIbfj/ap617jGKNqdZ9AGNfRUx1nyDYbVY2E72sdniPHhJIjAVRKwbYtjI+P4ejRgSCwadsYefhByF070NnRAYsZxuhYgKATCLUMRTdxfHD0twATQQDaTCVRmTvn0dlm5smTDR6eVt/4df39BN9Hx7Wv/Fq5uwta+0LJ6gEztAS0ALRkaBEOyeDwbxX+LUdHYWlgWmc75L7dGHv4QdjpFNyKgx07diKRsINEn/ZDlF4AYg+oatSJB7d6vRHNF1Lf8FT79FvsszUyMEBLNr4e30/8PBp/02/+TDiUcmGaEq7rYM+evSBiCNNA6fAhFO/chBltGSRME+boKKA8aFPUzT3L+qEFomuhBYMNQGsXanoXyfPP2cC+T3f09UngmeobHzYgY+bk49e//4nkAw+c4duWhmZBLcBXzV2LwrpwreB2dcGZNR1518GRiotp170RqfkLUMrlMH3GdCxZclaQJCSKbGQiPIWNTrGJwLO9v1Z9MAxIKbBly1a4jgtpGmDXxbEffA8zchPo7uyEPTkJ4+gxQMgWraPiQYNGJgwAIG14HpUuv2xv6t++dMESonyYj+RnRANVUxtEVMaK5T8wujohtNIk4yqyumRRSNcf2kDhmgwCICWM8XHY+SLSyRSmmQLDv/4lnLFRWMkkBgcGsXfvXrS1ZSN1HpTuNg60eK32Hjd9Nm7D1N6r2SpTvd5qP1xnB9X/Xovjqtpn+vgjWLYYRAKmaeDJ7TvgVJwwnyYw+Oub0TY+io6OdlhuBXJ4CCREHTFE1ZyoQlZqjWUQIiepWsoKmjGdOl72ss8vIcrx2rXiVIXnlDvAMrMgIr3j6NHlxXWffMze/JjJdiKYPa6VPdcIXWPxkAiXG6AXlSHhL5yPsm1jeHIC4109mP26PwQZJtxKGfPmz8PChQsxMTFZmyBuqmY+AR42ROHVqTCuCxCeNqa2KYLAU2T/j4P2jn2/2j3RMEzYtoVt255EPp+HISWkbWPojtthPPwg5kzrQpIA8/AhyLITOC3hTnRVaIIUZUOJXYwMmkhb0CJ/9qqt53/5qxeByMFpaJ9T0kChFtLc1yeWzpnzpH3hRV+wujoI2tMcou+YwrWW4sY0RY4Ui9AmMgChPRhHjyKpfExrb0P72BAGfv0LQGuYto2DBw5h7969aG/LhKW7OoRENBvQcYO2CvWoew3c2qBt+d3GfUw1dFgkWd1vnGGtxsYW4g/Df7HjidWxAQETrWkasCwDW7duRS43EdjFloXRB+4HHn8Es3q6kLQMyOFBoFyGlgKaAE0ETTWvmKlqi1bnPPTACIGtyj5TRxusCy7oJ6IKTlP7nLIAVYkawEwr/vR9/1SYMeuwJAgOWIzC9sUEEuEI/64uX1x18QmAIUGVMsTRo0gSoaejHeljBzCw8WYQCKadwKFDR7Bj1y60tWWDzjaqSpgfFNzVH34jxjpexCfCUft7Kkz2ySntwMWmqCZNREtF/W/XfourIQkSsdBEMJTSSCYzEEJi8+bNyOXykNKEtFMYf+QhePdtwuzOLFIJE3JsFCKXBxlGICzRboJ5pvBnWIYotyhOFy5vUjApXxRnzZ5c+Sfv+5/Tcd2fkgCtX79eh+vluHn2yu+ZmdClDw+QRS2QqMO/IUV0QtW7QRPApgEqFSAHjyFpSMzo6kLy8B4MbLwZYA3LsjFwbAhPPLE1bMmUgOd74V3b7GFx9KiO8/7xvLNGD0sfx8NSoXap1q7p1vVr4X7AKtqfDv/WWsH3PbS1ZeG4ZTz2+OMolcqQwoA0bYw+cj/cezdhTkcb0ikLcnIcNDYCmLIWICSE8x3GdqqBwjCYGAUUBQefh9ZWW5bsCy66iaQ8yH194lRd96ckQDGJpa53vfebxbnzS+z7UkvBkaqMRaJZhNHoqnCJ0N0nBO6kaQC5CcihASQNgZldHUge2oVjt/4M2nORSCYxOTGJRx/dDM/z0dHeHmJ6dGyZQdR+UnMNsMUhq0YUr6l7vWGg4ZGbQWytBo677LVeVgHAVwF3ZFdnJ4aGhvD4Y5uhPB/SMCFNAyP33Qnvwd9gTncHMtkUZG4SPDQIliKcQw61TzUGVzWgRcyJQS2YSwSWgqF9Kiyc77kXX/7v0Jo2rFz5lPxbcXq8jqS5r49mT5++LXXFS75ld2SJWWkWIlpr0aJxGlc5RigATnOV+cg0gPExiMFjSEqJmV2daBs6hIFbfwq3kIOdSsFzXTz22GMYCGEgpmnB93VdfX31B6vLVbDcyYbli9Dceq36PQqbEsU/I6Ym+a4updF+Kfo7/ns1gQy8Ld/TSKfSyGYyeHL7Djz55M6glbhhgRgYuvNXoC0PYn5XB7JpG8bkODAwECAMq9okFIpAmCg0D8IpqMbnqnaPQJDKYKXtznbhL1v+zXMuuuhh7uuj3t5ehacxoHGqHhlPMC/e84F3bzG2b7G1nQTF3Zs4ZrqJ7oJq166agFQ+uKMTevZMuCQwmsthxEii/bI1SM9eAM914Hsepk3rwllnnQnDMJGbzENpBRHvdENTQTdifEYNrQVqHhVPCXGlGL9OSyw1HQ/rHDSpsywLbe1tmBifwI6dO1EqlmEaBoRlQRXzGLvndmSGD2NmezsSpgEaHQUNDIVQD6rVk7VAfbSg/Ir3H2HpufBXrBqf+cX/umA20UHu6yNav14/6xqoqoX6164VHST28LKzv5no6iCwp4O1VgdDMFjoYBBHcSKOewehz8mkQaYE5cYhjhyCzRrTujowS/iYvPPnGHniAQghkUilMDE+iYcffgxHjx5De0c72trawDrgSozz7zR7VTV2sKCrAEfPa0sZNeOuIzgtGpq+ob7zDTczq1b7mRIJdHV1IZVMYvv2nXj0sc1wKy4sy4KwLBQP78Por29E9+QA5vR0I2lboOFBYGgAZFAU8Y/mLooyV18PI8zRCOe9qunZ02ZHlrwzl31vNtGB/rVrxVMVnqckQACwdsUKZjB1fuCvPleYv2hMkE/aII5UaPXkZJjWiIfWDQ0tNVgyEA4WDFgSKBWAQ/tgORV0tmWwIJuE2PoAhu/8ObzJMVipoO5q967deOihhzAxOY6urk60tWXDC6aibn4tR1OT+pjATGn7tICNVF+PCR1HjGE6Qhp2dXagrS2DI0eO4t77H8DRI0cgBYUtIBTGHrkHpXtuwWzhYXpXBywC+OgBYGwIZMlgjoQOln3JiEgPJYPC+YzmMPZ3MDRggCV7ND5z7qT9xrd9gQFau2LF0xI+p6e6g/61a2Xvhg1q8/e/9rfy5h//Y3l8TJE0JKYoyWkCHTcTtlajatBSAtNnQXd2wNMa47k8RmHCXnoeMouXgUwLyq1Aa4229nbMmzsX3d3d0JpRLBbgOAHliZQUA6txi6Z9UzDDEp8waBmPT1YTwEIAiUQCqVQazBrHjg3i0OGDKJecgOjckCAiVAYOo7DlQWSK45ieySBl25ClItTRwxBOGQhbODU2CDgeLyc3NDskIsDzlD1zljTWvvsjS171B/9avWbPCwFiZsI6ou3rOD3xqQ89mNrywFKPDN2s3bgJl1wPg6ZmiGmY/UZnFzBjJpRhoVgsYihfRKVzBtLLLkB69jwQCXiVMjzPRyqdwuzZszFj+nRYtgXX9VGpVOC6bsDALqghZsPHyW/RcSGm1Y6B1S7QlmUimUyGbTJLODYwiGPHjqJSqcA0TBiWCQgJZ2IMue2PQhzdg+kJCx3ZNlgE0Ogw1OBAUFEhZAs77rQusLKJpXfBVT897+P/+AcbenvF2v5+fbqBw6ddgAKvvl/29vaqzb/62Vp94zd/qA8f0NqyZZDW4Jo9fRLUFHVI0LAHBCsFnUyAZsyGzmbh+QqTxQJGXQ3VMw+ZM8+G3TUdDMAPS3tMy0RnZyemT+9BZ2cHLCuAjlZLdXzfC5e5+krQVuCuOhsnhMRW81WWZcGybBABlUoF4+NjGBwcxuREDlqrgGrXNEFCwC/lUdy/E/7BHWhXDrqzGSQSNqRTAQ8cAyYnIKRsYo3g0+XzIGLhucxzF3sL/u4fL5g2e+G2ajrq6Ur8Gk/HTnp7e1V/f78851W/v2HzV/9ljZkffX+lUlYspGwo5Y9ubJoi4xS3Q6MqUmGAPAf60D5QZyfsnpno7upExvMwMXkE4/cdQa5rJlLzlyIxfTasRBbK9zEyMoKBwQGYhoG2bBadXd3o6GhHOpWGaWXD6lgOW3kHnabjHZSrmkoIEQhC2KW5Cn53XRf5fB5Hjx7F2Nh4VKdumBKGacAwUgBruJOjKB3aDX1sHzJ+Bd2ZDFKpLAzlg0cHoYeHQEqDLDP0sRo9OY6VC3Azwz81vhfeDFprK5uR+sJLvzxt7uJt/f39koIivqdto6drR9zXJ2j9embmnof+7s+elDse6VBWAmGgpIY3Zj7h0fCU6UgGawU2LVDXNIiuLnhSouK4yBVKKPiAm+2BNWcx7OlzIBPpgLHM9+F7bsC2QUHjuGQygVQ6hVQqaO9t2zZMM2gNTiLW1ooD99vzPDiOg3KpjGKpiFKpBMepwHU9sAakEbQCJ2kEJpznwBkbgntsH3j0CDLsoiOVRCaZggWAcxPQw4PgSqlB6zTMVQOBSd2iW9XSHGuZWZsuLX2PsOz8w+d/5j/PJaIJPs2E6TOugQCA1q/XoYQP7bj5R1/QuWOfzQ8dU2TaVWx6PaMGt2BeiruGsSx3nCaFpAFiBR4+CpUbhejsRrq9E6mebriej2JpArld96OwNwHdMQNWzxxY7dNgJLOB8VrlESyXkS8UIuRgtfcEQCEDfe2C6rqe7gENsJRBcWMimQm7EmkotwRnbBDu8GHo8WOwnQK6TYFMewpJqwNSK4jCJNToELhYCDScZdQqKKr8RkwRVyPXcfrEe+ygiZ+ktvJRUGk6fYawr3jFJ4lo/JnQPk+rBqoZ1OsI69ZZ2776uY3ub356mQupiWIGdVjtEGcubXbOGC3bF8eqLCkkDmCtAduG6OgEtXWB7SQ8FRjOhXIFZZ9RkQlwpgNGWw/M9mmQ6SzITkEYBig0VoOIsUY9gTk10KOEUW/WYOVDeS60U4SfG4M3MQzOjcB2i0iSRjZpI5lMwTJMSN8D5yehx4ehy/mggrbqYcWJ2inOFhJbxKc0gaieeaxW5qSThhTuBS+/c9+H+l62dt06xrp1/HRrn6dVA0WgswC5WGHmdz56dPejcs/mpJIJpiBHEGMtRRRZrSsHi7OXE6YodQ6VvESABVYuePgo9PgwKNsGI9uBtlQGmWwKvmK4joOKM4ry0UF4RwiOMOFbSXAyDbKDAdOGMBOANGNR37AttlJg5UJ7DrTjgJ0iUCmAnBIMrwyTFbKmgVTCRLK9A7ZhBB2TKyVgbAAqPwG4Afm4NKrLlW5xczQrZAK3bAMeX+yprvyU2PAc+PNWDq/8UN87LyZSYcL0aReep12AojxZoC53PfaTb/y9mBj4nJoYUSxMyYgX1VOTZokDvwgEbiLYoCmMJQKkCYIG58fAhXFoywalsjDTbbCSaWTSqaD6Qil4ngvHK8Ep5eHlFHwGfBZBzwuS0BQjHA+rUCUAkxiSCJYUQalP0oSVzcIwAsOatA9yykAuB1XIQVeKEEoFDYfDigmOU4o0RS6pqe9VQ61tA5M9murN4LvKnjHL8M9f89UU0YH+/n5JTzHf9awtYfGd/nDtWvmmG29Sj/7Hx27EI7deV/aVCnXG6Xfm48bePS30U3Umw5obTQQyTAg7AZFMgxJJkJ0AG2aQqQ4p1BkUY81ouMMFQZAIHwmiWsuj/MA7dMrQlRK4UoB2nECQSIS2VGO782rFK4H4ZDqF1Lfe5UauJKrFzUhDWxKics7Lt5z1gc9cMW3duuIztXQ9owIUh79WmJc+/tk/eUzvfsRiK1lP4X4qTeQbqVOnalBHXM9nESW6dM1TkRIwTJBlgkwbwrBBhgkSRkDiIER9UxetAe0HbaZ8H+y7YNcBey7Yd4P3gaiZXGS0cD2rYaPoM+gkOmFUbSBqIV21pStIllaYzzivkOj929XnL136WOgZazyDm/GMSSaR7u9fKxNEO+7b8I3/k8gP/l1l5JgP0zaackp1qppbVnfUjOdqfTo3cYo1P6s2pCWAZK3QmBnsVcBuuQoHC7yvGCFmHZ8i69hyGS7DoZ0kBAFS1sg6w9YBYQ/0umWmmX6qdSCcYp3PRbUhHzXcT3E2fSKQ56rEnEVG+byXf+r8pUsf42d46XrGNVCDVyYf+96//pIe+t+Xl4pFRdKQU3L/VfktuN6aZCLUdY6O/uIGhU+xbn7clCrhpl71aOiKwU2XvD6DQc0Na6cIilILa4ZjjdeoVffGFlGOlsZzPGKulUpl22Rl1at/eOm7PvIWbOgV1LvhGReeZ1QD1byyPiIij5nf9eDY4QeNbZtm+UKEuTKe0rto7hRWo0mp5/3hxrlvYS9wPUFO1XPheOdpinEJ0pQNJ+MeEbWCDcV7Y1Br2huOBQeprtktT91uOF5UEauyYM3asiD1kosfuPRdH3lfCGzSz7BueHrgHCcnRFGA8Qid9+q36rnLXILDuh7EF+FcdATPRKyeOwDkR5BYqlV4kAhdpKbv1QDnEcCcEAL+q4wFNfKB6j6CxvfhPsPB1ZLh6j5lw/MQK08iYBKp/Qaikm+WqCv/rlZQ6IZ9s6ijA4i+Q+GxUOx9FkIbQpE7c+n4wnd8rpeIJhAWgOJZ2sSz8SO9vb1q48Y+4+KrrrmDzn7Fx+0Z8ySxo2BQPX7FiONYEFUVBI8cMoAwYDRcEFmrzw96PAJcfZT1whD8Ru1zMBrel7HXqpjjanVJ9fvVC2owqO6ztd9jGR5D9XxEDatD8fM1QsxO3TmHnwmPtfpb0XEZAAwCqbKWMxaQPOe1H+tK0YEwfKLxLG70bP0QM9OGDb3iTW/9qbr3fz79Q2PrL3rLpaIiYQS9uOviQVRPTcvU4L5TrK1S8zpC3CK9X5dAwnFSuWhauKih7XjrymY+Tt/TRvvqeAyxNDV3dOQ8ENh3/bZpM4zMVe/88uKXvf16/uH3nxWj+TkToMioJsJ+5vZj//m399Ou25f4LDQJEsfH5TQaBtTQ3JDjxkFTgP8ky1kb8EhUV1iLFtX+rcFyfByMSqPNRrXvUWNAMfZWYxCVtTIFZPqy3gdXvuEjV68jKq8LUgD8bAuQeFalNeDXp0VEE9mXva/XXnJFXrITwPDrau2osfaupt6rS4kM+owTBXESEuFjSDMc2FfxytgYnaqoVoU0LF3xKlqKfSb2ndrf8aINRvPxU+1voqhyN7JrqgSl8c9Viy6r+5cBXXLdMZPWhtRSLr3qkPmSP34zERXBfXguhOcZ98KOm+pYuvjxgf2b//xgafQ76tDjvrbSEmESoZlEvqE3Uay1AouGzjN1Tj439CxtbB/OMW+p1smYW8SB40nNuEdfl/Ol6j50fbqlCYbZCtnTuNpyRENXpf9lJm3AE+aiS45d8Cf/fC0R7e3v75e99OwvXc+ZAAEA9faqjX19xsyF53z3gV98+4wMl9bnj+3RZCUJrE/QS70+GUtRY1pMnXSk+n7piOeOGrqONS5z1Ru7LqAZ9RmNLS/UlE+p64tBdJwlkFq0zWsC3hFLVWFz5jJv/us//HYi2raxr8+4urfXx3O4Gc/VD69Zt071r9wmL/m993x6661fS1judz7m5oYVS1NWk6o0FRa5ygJSlwyoMYpzhKehOgIDoBVbJh3HNqJ6PE68U2CdScZT7KuFTVTthRYKPzcVdDVAV4hBECC/ohLT5xu07Nq/6Zm17PaNG/uMq69e/5wKz7NuRLcyqtcR0XpI/dBPPvd99eRP31wuFTwI06QmDucamrGx5SZV+1AQTp7zqanNOB+n5KI599RkeBOfkHOqPqVFLXp4NXhixEHgx3f9THu34S9+zX9f1vv/vYd/+HpJvRv0085s9Xw3olsa1X196F+rJC79mw/6c1/6WCqVMQlakRRRcK6an6wG6IQIcMpCBsRLJCj6OxryBEMIUNgIRUhq8RkZ/k4wSBKCYxIR80jd94SAELLF78vob5Lxz1B0DNT4HVH7LrHyk5ms4c5Zc9ulb/yrP+/7lC+wtv95ITzPuQZqzNwz84Jt//vZn048dtM5PkMRhKyakS3SpJja8GkFro7nt2K9FqeoZK41buD6qoiWUFyqt3q4DvHecFQNpOzAFDEmAliphGVKd9ZLt6z6o8+v7iAa6+vrE+uf4Qz7C8IGas7c90siOsDMr713dORhcfjO6YoRdkzj+gBggzfTyvjlusLAVoLXnNHkWFsXgg6hZPXoAFCrvmMNgkQtmObrI0ENty83QRGZlbJNKd3pF23rfOn/99oOorGnuyTnt0aAqukO5n5JRIe3PPbQW8qG+m5x/29mgkxNRCKIFXID/IOnIKLjhpufT9RJM9Z3sf6xtcNdFbd67BFPHUI+MZypmoQVBM2sTWIp5lx+OHPp3/zh2Wd0HeRnCBT/W7GE1c9jvyTqVS7zRZu/f/0dhb13pmAkOAg1toj+xsBbda0KpgCKAK1S4q0uLdXDN+IIXGpOVbT8TaYGlEBjFW6D9xfGekzyBHeeM9JxyYdffc6FFz70bGF7XtAaqLacBYlXi+ihfTseeGcpP/YjNbKFYCQ5wN2FqMO6OsWwdRG3cM+biV3q8mjUFAmilviksHNMA1aHWmRBqCXWkBorXVt2tiJtsCu4a2XZOKv3jc934XleaqDqVo1zHNhx7x8NP/jVr+UPPSDISAJQQTqxFeKKG0D6DbGjxvIZ1KfV6nt1UY11NqYdptA3gc2EWCFAnFS26bNcDywLviC00A5ZPWdXui95f++SVS/92fMl1vOCFKC4EI0defRtu279/DeLA48BZlpQjFa4tZaJJ1cp5hlxaxpeii9x3BDdqI9815Qc1/ecDwWIGoLR1LKrUINRT6ThV8joXKrl4t4/uPKat/xsY1+fcfX657fwPC+XsPh29dXr/Y19fUbXnPO/+9g9N3QJU/z75OFHNMlUrcdmg0joWAS5ri9nXZ94inq+Mtdrjanc7Ci7Rdxgujc0Hm64LXUdnLUBIhJoNSauIDX7XMLc17//spf1vmCE53mvgapsGT/84Rtlb++P1eDBuz64/85/+/fCwGaGkQSxJj5hX2ZqShPUJy2pHszeCpxPDcBW5qboDldZOzhIfOoWy1ajzDGRFroCo3O5mHPZh963ePnqr2zsW21cvX7TC0J4XhAChIjICrJ3A9SxfXe/e/emf/maN7KdYCTiOfHmk4tILqg5M99qImJmSRzXFYeDMRrasFMN6NUElG9IwdQvb0JDO8LsWg5z3hv+7LKXveWrLzThec5TGacUJ9pAauPGPmPWoiv/u2PFu/8oO+8CZi7VsETUgCWmgDMwwAaFHI0UcAZWW5PXWpQjYtKvYnEiposAgxO1Mqfw7zgmKEqmRvtEDdsTO64IY0TQhIpIzTy3nFn2R+++7GVv+erGjX0vOOF5QWmgRsN6cnjrO3bf+fmvTxy63yAzycQsWuqW41V8cH1yob4fRysfik7cTmOqTHy1PTJDm9IXnFlZ6lj2x28479Jrb34hap4XhBE9tWG92mjvWfntfbsfKJYrut8ff0hqskIWEJ6KaChKh1Oc/awhi1BbZppzZ626VNfCP0ESpH5FpQbWNVa2oaVuO2/cnPWWt5x36bW3vBBc9d8qDdSoiY4e3X7t4KP//oPJg7e3KzYUkZA8FUziqcwQn1wHH26oPI6AZKyVIbU0ui8bm3vJ3123YMEZd73QhecFLUDxtAczv+ThX3z4puLhX3UpJh+QRpWirgmoFdcbzGjVj21qATrJFteNkSatlGULqduuOGif9cHrLj3/7MdeSK76b9US1irtQUR3bd264zVa8/fV6KZFpXLFJ2EZiAo0mwuQOc4INuXtFGPSaIBl13vltehSHRkLEbR2/VQ6YXDbVU+0rfj0dect69zX398vn2so6osaqLUmWrj97s9+Z2LvjVcWy0WfhGVU6euOt9RQU7IBaARt6LBejdDUG6G5AX0Yv2Jd8VOZDkNlXvrI8mv++bppaTr826J5fis0UFwThVCQ/cz82nsnnBvTE79eUyyMKTKSkkJql2YhoYaUK8VL+EKm1ni8g5tw1RzCOhAv+oNg6JJq65hhuJmX39L5knVvnZamsd8mzfNbJUBVIQpBaZMHmV8zec+Mr1qDN719fGSvD5mSQUV9i8x5XQZfx3gGY1HBFu020QA2qwUjBROXkOpaYHDnq789/xUf+bP5ROUQDKbwW7YZv00nE4DSWBBR2bDS79j24FcHfP7WRwvjuzSLJAQx1bntVKuw4OqygxblOdQafNYCNaZJl4SRXcHofv3HLn35+/4R+Cv0PcWmbi/aQM9BtQcRESD0/Ru/8Ql/9MbPuLknoGFpIiFO1cfnKR0yrhFSaa1MQ8tE1yWO0X3duy68/I0/5D4IrHtuSo5fFKCnow4fRETQIwM7Xr/3kX/+dmX4zrSvhR9w7HILHFCLZFoDvIOaQIQCWrt+ImEaKnXZwZ5lf/lHy5ev2sj9ayX1/kg9T4on8DufCzudkiEi6I0bVxvTZi69wZr+3jcY3dceTiVtA+woIhH2Ug9zXK0GISy7QViDzw2fF9C64rd1dBmi67VbUmd+7mXLl6/a2Ne32ggYwn67hee3WgOhRdT6gSdK8zDwT/+J/K2vKUwe0RCpsGH2cRavKehWmIlZl7mtc5FoX/SmG84690/fS0Sj1ZACfkc243fhJK++er3f398vL1mVOsTMv3ff7e3/nqRffaAysYOZ7KDqo0lKdMta9wCVqLWAI5Jdq8hve/UXzzr3T/+qVpr0uyM8vzMaqKY1+gTRegaI77/rB+/zhn/6JZQfNlxHh0RXTX0u69GGJMDa821LGKnpLx2ZuexPPjhr7iU/YAZh3VPvP/qiAL3AjOvtu7e8bHzH//sy8ncvLZfzPglbxqvX67Nngtkv60xbu0Rm9fbzrv3XN9tEj3P/Wom1T18DtxeN6BeMcd1nLDvz7NuTS//jpbr9929u65xvaFUGM7VAEJFWfgHt0xfLacv+/Mb0Bf96tU30+MaNfQb1blC/q8LzO6mBpsihmft33vDZoR3f+Wh5fLPQ2lAQIugNpZQSwpepaRepeave87E5i1//z9ovRV0a8Tu+Gb/LJ0/Uq8IosQfgb7c8tnGjq7/1zYT78PRCseiDgVQ6bbTPed2hWUvf/8GunsU39fVBAH14UXhe3Orsoo19qw0A2Lk/t2L3o//6ywdvWs0P/XQ179vy5V8x84IgHLDaaNVT9cXtxS0QpP5+CQCmlcX2zd/+5OP3f3U9M4vqcvfiDL24nRRX0cm89uIWbP8/3eHk1MD1i0MAAAAASUVORK5CYII="
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


def _render_stat_delta(current: int, previous: int) -> None:
    """이전 값 대비 증감을 "증가/감소/유지" 태그로 보여준다. 원래는 증감률(%)을 그대로
    보여줬는데, "오늘 0명·어제 1명" 같은 작은 값에서도 "-100.0%"처럼 과장돼 보이는
    문제가 있었다(2026-08-30 지적) — 퍼센트 대신 방향성만 알약(pill) 태그로 표시한다.
    """
    diff = current - previous
    if diff > 0:
        bg, fg, text = "rgba(46, 160, 67, 0.15)", "#2ea043", "▲ 증가"
    elif diff < 0:
        bg, fg, text = "rgba(248, 81, 73, 0.15)", "#f85149", "▼ 감소"
    else:
        bg, fg, text = "rgba(139, 148, 158, 0.15)", "#8b949e", "– 유지"
    st.markdown(
        f'<span style="background:{bg}; color:{fg}; padding:0.15rem 0.55rem; '
        'border-radius:999px; font-size:0.8rem; font-weight:600; white-space:nowrap;">'
        f"{text}</span>",
        unsafe_allow_html=True,
    )


def render_signup_stats_card():
    """신규 가입자·전체 회원수를 카드 하나에 나란히 — 원래 카드 2개로 나뉘어 있었는데,
    옆의 회원유형 도넛 카드 하나와 높이/개수 균형이 안 맞아 보인다는 피드백(2026-08-29
    스크린샷)으로 하나로 합쳤다."""
    with st.container(border=True, key="dash_card_signup_stats"):
        users = st.session_state.admin_users
        counts = _signup_counts(users)

        # 옆 도넛 카드(회원 유형 분포)와 카드 테두리 높이를 맞추는 CSS(height:100%,
        # flex-grow 등)를 여러 조합으로 시도했지만 Streamlit이 이 컨테이너에 이미 걸어둔
        # flex-basis:0%가 위에서 덮어써져서 안 먹혔다(2026-08-30, 실제 브라우저 렌더링
        # 픽셀 측정으로 확인). 대신 위/아래 여백을 콘텐츠처럼 실제로 넣어 높이를 맞추고,
        # 그 여백을 정확히 절반씩 나눠 콘텐츠가 세로 가운데에 오도록 한다. 87px는 실제
        # 렌더링 높이(도넛 카드 325.75px vs 이 카드 콘텐츠만 151.5px, 차이 174.25px)를
        # 헤드리스 브라우저로 직접 재서 정한 값이라, 카드 내용이 바뀌면 다시 재야 한다.
        st.markdown('<div style="height:87px;"></div>', unsafe_allow_html=True)

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

        st.markdown('<div style="height:87px;"></div>', unsafe_allow_html=True)


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
                x=alt.X(
                    "학습시각:N",
                    title=None,
                    sort=list(df["학습시각"]),
                    axis=alt.Axis(labelAngle=0),
                ),
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
        df["유동인구_label"] = df["유동인구"].apply(lambda n: f"{n:,.0f}명")
        chart = (
            alt.Chart(df)
            .mark_circle(size=70, opacity=0.55, color="#6C63FF")
            .encode(
                # y축 제목("폐업률")을 세로로 회전시키면 좁은 카드 폭에서 글자가 한 자씩
                # 줄바꿈돼 깨져 보였다(2026-08-29 스크린샷) — 카드 제목/캡션에 이미 축
                # 의미가 나와 있으니 title=None으로 없애고 % 포맷 라벨만 남긴다.
                x=alt.X("유동인구:Q", title=None),
                y=alt.Y("폐업률:Q", axis=alt.Axis(format="%"), title=None),
                tooltip=[
                    "동",
                    alt.Tooltip("유동인구_label:N", title="유동인구"),
                    alt.Tooltip("폐업률:Q", format=".1%"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


def render_prediction_risk_chart():
    with st.container(border=True, key="dash_card_prediction_risk"):
        st.markdown('<div class="dash-card-title">업종별 평균 폐업위험도 TOP 5</div>', unsafe_allow_html=True)
        st.caption("predictions.score 평균 (표본 10건 이상 업종만) — 모델이 실제로 위험하다고 보는 업종")
        df = pd.DataFrame(st.session_state.prediction_risk_by_industry)
        df["건수_label"] = df["건수"].apply(lambda n: f"{n:,}건")
        chart = (
            alt.Chart(df)
            .mark_bar(color="#D8454A")
            .encode(
                x=alt.X("업종", sort="-y", axis=alt.Axis(labelAngle=0), title=None),
                y=alt.Y("평균위험도:Q", axis=alt.Axis(format="%"), title=None),
                tooltip=[
                    "업종",
                    alt.Tooltip("평균위험도:Q", format=".1%"),
                    alt.Tooltip("건수_label:N", title="건수"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)


def render_industry_group_chart():
    with st.container(border=True, key="dash_card_industry_group"):
        st.markdown('<div class="dash-card-title">업종 대분류별 매장 분포</div>', unsafe_allow_html=True)
        st.caption("industries.custom_group(대분류 10종) 기준 stores 집계")
        df = pd.DataFrame(st.session_state.industry_group_distribution)
        df["매장수_label"] = df["매장수"].apply(lambda n: f"{n:,}개")
        chart = (
            alt.Chart(df)
            .mark_bar(color="#3D6FD9")
            .encode(
                x=alt.X("매장수:Q", title=None),
                y=alt.Y("대분류", sort="-x", title=None),
                tooltip=["대분류", alt.Tooltip("매장수_label:N", title="매장수")],
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
            '<h3 style="margin:0; font-size:1.25rem;">📈 업종 트렌드 TOP 5</h3>'
            f'<span style="font-size:0.75rem; opacity:0.55; white-space:nowrap;">'
            f'⏱️ 마지막 갱신: {datetime.now():%H:%M:%S}</span>'
            "</div>",
            unsafe_allow_html=True,
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
        st.caption("상호명으로 가입자를 검색합니다.")

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
        # "운영 반영" 체크박스만 편집 가능하게 열어서, 여기서 켠 모델이 실제
        # models.is_production으로 반영되게 한다(2026-08-30 요청 — 이전엔 상태만
        # 보여주는 읽기 전용이었음). 체크박스 자체는 여러 개를 동시에 켤 수 있지만,
        # is_production은 "한 번에 하나"가 원칙(query_predictions.py가 LIMIT 1로
        # 조회)이라, 방금 새로 켜진 행 하나만 반영하고 나머지는 DB에서 자동으로 꺼서
        # 라디오 버튼처럼 동작하게 만든다.
        other_cols = [c for c in df.columns if c != "운영 반영"]
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            disabled=other_cols,
            column_config={
                "정확도": st.column_config.ProgressColumn(
                    "정확도", min_value=0.0, max_value=1.0, format="%.3f"
                ),
                "운영 반영": st.column_config.CheckboxColumn("운영 반영"),
            },
            key="model_management_editor",
        )

        if not edited_df["운영 반영"].equals(df["운영 반영"]):
            newly_checked = edited_df[edited_df["운영 반영"] & ~df["운영 반영"]]
            if not newly_checked.empty:
                # 새로 체크된 모델이 있으면 그것만 반영(나머지는 DB에서 자동으로 꺼짐).
                set_production_model(newly_checked.iloc[0]["모델 ID"])
                del st.session_state["models"]
            # 기존에 켜져 있던 모델을 새로 켠 것 없이 그냥 끄기만 한 경우(운영 모델이
            # 하나도 없는 상태는 허용하지 않음)는 DB에 반영하지 않고 그대로 새로고침해서
            # 체크박스를 원래 상태로 되돌린다.
            st.rerun()
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
        top_df["유동인구_label"] = top_df["유동인구"].apply(lambda n: f"{n:,.0f}명")
        region_chart = (
            alt.Chart(top_df)
            .mark_bar(color="#3D6FD9")
            .encode(
                x=alt.X("지역", sort="-y", axis=alt.Axis(labelAngle=0), title=None),
                y=alt.Y("유동인구", title=None),
                tooltip=["지역", alt.Tooltip("유동인구_label:N", title="유동인구")],
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


@st.cache_data(ttl=30)  # 30초 캐시("업종 트렌드 TOP 5"와 동일) — 5분은 너무 길어서
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
            "위험 매장에 대한 관리자 개입 조치를 등록하고 "
            "이력을 확인합니다."
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
            # 한 화면에 5행만 보이게 높이를 고정하고, 그 이상은 표 안에서 스크롤한다
            # (row_height 35px + header 38px 기준 — st.dataframe 공식 높이 계산 방식).
            visible_rows = min(len(df), 5)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=visible_rows * 35 + 38,
            )


# ────────────────────────────────────────────────
# 메인 레이아웃
# ────────────────────────────────────────────────
def main():
    # 실제 관리자 로그인 세션 확인(shared/auth.py) — app.py의 ADMIN 분기, mypage.py의
    # 로그인 가드와 동일한 방식. 관리자가 아니면 로그인 페이지로 안내하고 멈춘다.
    auth.restore_session_from_url()
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
                f'<a class="mp-logo-link" href="{auth.home_url()}" target="_self">'
                f'<img src="data:image/png;base64,{_LOGO_PNG_B64}" alt="hoTSpot">'
                f'<span class="mp-logo-title">hoTSpot</span></a>',
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
