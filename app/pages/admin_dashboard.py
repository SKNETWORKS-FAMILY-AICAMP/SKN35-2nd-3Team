"""
관리자 페이지 (관리자 대시보드) - Streamlit UI

ERD(schema.sql)에 있는 테이블만으로 구현 가능한 기능만 남겨뒀습니다.
- 회원수 조회      : 전체 회원수, 오늘 가입한 회원 수, 권한별 분포(users)
- 오늘의 인기 키워드 : 상권 트렌드 키워드 TOP N(trend_keywords), 30초마다 자동 재조회
  (trend_keywords 자체는 snapshot_date 단위 배치 집계 테이블이라 "실시간"은 스트리밍이 아니라
  화면이 최신 데이터를 자동으로 다시 읽어오는 수준을 의미한다)
- 유동인구 TOP N   : population_features.total_pop_avg → administrative_dongs 집계
- 인기 조회지역     : dong_view_stats(지도 클릭 집계) → administrative_dongs 조인
  (dong_view_stats는 아직 schema.sql에 없는 테이블이라, 없으면 에러 대신 빈 상태로 표시한다)
- 사용자 관리      : 가입자 목록(users), 상호명 검색(store_snapshots.store_name)
- 모델 관리        : 이탈 예측 모델 버전 관리, 재학습 실행, 성능 지표 모니터링(models)

계정 정지(상태 컬럼), 브이월드 API 연동 상태, 데이터 갱신 현황, 방문자 수/API 호출량,
에러 로그/API 실패 이력은 현재 스키마에 대응하는 테이블/컬럼이 없어 제외했습니다.
(새 테이블·컬럼이 추가되면 다시 붙일 수 있습니다.)

실제 DB 연동 부분은 TODO로 표시했고, 지금은 더미 데이터로 화면 흐름만 확인할 수 있습니다.
"""

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

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
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────
# 세션 상태 초기화 (더미 데이터)
# TODO: 실제로는 관리자 로그인 세션 확인 후 각 섹션을 DB에서 조회
# ────────────────────────────────────────────────
def init_session_state():
    if "admin_users" not in st.session_state:
        # users 테이블: user_id, login_id, user_type, store_id(owner만), created_at
        # store_name은 users/stores에 없고 store_snapshots.store_name(최신 스냅샷)에서 조인해온다.
        st.session_state.admin_users = [
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

    if "models" not in st.session_state:
        # models 테이블 기준
        st.session_state.models = [
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

    if "top_regions" not in st.session_state:
        # population_features.total_pop_avg JOIN administrative_dongs, 유동인구(평균 총 인구)
        # 상위 5개 동. 아래 값은 실제 DB에서 조회한 상위 5개를 그대로 옮겨온 것.
        st.session_state.top_regions = [
            {"지역": "여의동", "유동인구": 108739},
            {"지역": "역삼1동", "유동인구": 108306},
            {"지역": "화곡8동", "유동인구": 86418},
            {"지역": "서교동", "유동인구": 84851},
            {"지역": "서초3동", "유동인구": 70607},
        ]

    if "trend_keywords" not in st.session_state:
        # trend_keywords 테이블: keyword, snapshot_date, store_count, growth_rate
        # (최신 snapshot_date 기준 store_count 상위 N개, growth_rate는 최초값이 0이면 NULL)
        st.session_state.trend_keywords = [
            {"keyword": "무인카페", "store_count": 128, "growth_rate": 0.42},
            {"keyword": "반려동물동반", "store_count": 96, "growth_rate": 0.31},
            {"keyword": "회전초밥", "store_count": 87, "growth_rate": 0.12},
            {"keyword": "저속노화식단", "store_count": 74, "growth_rate": 0.58},
            {"keyword": "노키즈존", "store_count": 63, "growth_rate": -0.08},
        ]


# ────────────────────────────────────────────────
# 1. 회원수 조회
# ────────────────────────────────────────────────
def render_member_count():
    with st.container(border=True):
        st.subheader("👤 회원수 조회")
        st.caption("users 테이블 기준 전체 회원수 및 오늘 가입 수 (전일 대비)")

        users = st.session_state.admin_users
        today_signups = sum(1 for u in users if u["created_at"] == date.today().isoformat())
        yesterday_signups = sum(
            1 for u in users if u["created_at"] == (date.today() - timedelta(days=1)).isoformat()
        )
        diff = today_signups - yesterday_signups

        if diff > 0:
            badge_bg, badge_fg, badge_text = "rgba(46, 160, 67, 0.15)", "#2ea043", f"▲ {diff}"
        elif diff < 0:
            badge_bg, badge_fg, badge_text = "rgba(248, 81, 73, 0.15)", "#f85149", f"▼ {abs(diff)}"
        else:
            badge_bg, badge_fg, badge_text = "rgba(139, 148, 158, 0.15)", "#8b949e", "유지"

        total_col, today_col = st.columns(2)
        with total_col:
            st.caption("전체 회원수")
            st.markdown(
                f'<div style="font-size:2.25rem; font-weight:600; line-height:1.2; '
                f'margin-bottom:0.75rem;">{len(users):,}명</div>',
                unsafe_allow_html=True,
            )
        with today_col:
            st.caption("오늘 가입한 회원 수")
            st.markdown(
                '<div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap; '
                'margin-bottom:0.75rem;">'
                '<div style="font-size:2.25rem; font-weight:600; line-height:1.2;">'
                f"{today_signups:,}명</div>"
                f'<span style="background:{badge_bg}; color:{badge_fg}; padding:0.15rem 0.55rem; '
                'border-radius:999px; font-size:0.8rem; font-weight:600; white-space:nowrap;">'
                f"{badge_text}</span>"
                "</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────
# 2. 인기 키워드
# ────────────────────────────────────────────────
@st.fragment(run_every="30s")
def render_trend_keywords():
    with st.container(border=True):
        st.subheader("🔑 오늘의 인기 키워드 TOP 5")
        st.caption(
            "trend_keywords 테이블 기준 (최신 snapshot_date의 store_count 상위 N개) "
            "· 30초마다 자동 갱신"
        )
        st.caption(f"⏱️ 마지막 갱신: {datetime.now():%H:%M:%S}")

        keywords = st.session_state.trend_keywords

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
        st.caption("가입자 목록 (상호명 검색 가능)")

        users = st.session_state.admin_users
        search = st.text_input(
            "🔍 상호명 검색", placeholder="상호명으로 검색...", key="admin_user_search"
        )

        df = pd.DataFrame(users)
        df["store_name"] = df["store_name"].fillna("-")
        df["store_id"] = df["store_id"].fillna("-")

        if search:
            # TODO: 실제로는 WHERE store_snapshots.store_name LIKE :search 로 DB에서 검색
            df = df[df["store_name"].str.contains(search, case=False, na=False)]

        if search and df.empty:
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

        model_ids = [m["model_id"] for m in models]
        select_col, action_col, _spacer_col = st.columns(
            [1.5, 2, 3.5], vertical_alignment="bottom"
        )
        with select_col:
            selected_model = st.selectbox("재학습 대상 모델", model_ids, key="admin_retrain_select")
        with action_col:
            retrain_clicked = st.button("♻️ 재학습 실행", key="admin_retrain_btn")

        if retrain_clicked:
            # TODO: 실제 재학습 파이프라인(app/build_features_and_model.py 등) 트리거
            st.info(f"{selected_model} 재학습 작업을 시작했습니다. (구현 예정)")


# ────────────────────────────────────────────────
# 5. 인기 조회 지역 TOP N
# ────────────────────────────────────────────────
def render_top_regions():
    with st.container(border=True):
        st.subheader("🚶 유동인구 TOP 5")
        st.caption("population_features.total_pop_avg → administrative_dongs 집계 기준")

        top_df = pd.DataFrame(st.session_state.top_regions)
        region_chart = (
            alt.Chart(top_df)
            .mark_bar()
            .encode(
                x=alt.X("지역", sort="-y", axis=alt.Axis(labelAngle=0)),
                y="유동인구",
            )
            .properties(height=200)
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


@st.cache_data(ttl=300)  # 5분 캐시: 클릭마다 실시간 반영이 필요한 화면은 아니라서 부하를 줄인다
def _fetch_popular_areas(limit: int = 5) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["dong_code", "gu_name", "dong_name", "total_views", "last_viewed_at"])

    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return empty

    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            return pd.read_sql(text(_POPULAR_AREAS_SQL), conn, params={"limit": limit})
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
        st.caption(
            "dong_view_stats 기준: 기존점주·예비창업자가 지도에서 클릭해 조회한 지역 랭킹 "
            "(게스트 클릭 제외) · 지도 클릭 연동 전이거나 테이블이 없으면 데모 데이터로 표시됩니다"
        )

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


# ────────────────────────────────────────────────
# 메인 레이아웃
# ────────────────────────────────────────────────
def main():
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
        st.title("관리자 대시보드")

    st.caption("서비스 운영 현황을 한눈에 확인하고 관리합니다.")

    render_member_count()
    render_trend_keywords()
    render_top_regions()
    render_popular_query_regions()
    render_user_management()
    render_model_management()


if __name__ == "__main__":
    main()
