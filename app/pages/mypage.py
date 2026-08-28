"""
마이페이지 (My Page) - Streamlit UI
ERD에 존재하는 테이블(users, stores, store_snapshots, predictions)만 사용하여 구성했습니다.
- 내 정보           ← users
- 내 매장 현황       ← stores (owner만 해당)
- 내 매장 스냅샷 추이 ← store_snapshots (owner만 해당)
- 내 분석 히스토리   ← predictions (user_id로 필터)

실제 DB 연동 부분은 TODO로 표시했고, 지금은 더미 데이터로 화면 흐름만 확인할 수 있습니다.
"""

import importlib.util
from pathlib import Path

import streamlit as st

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
    """predictions에서 query_type='new_location'인 좌표 조회 이력을 (query_lat, query_lng)
    단위로 묶어서 최근 순으로 가져온다. DB 연결이 없으면 None, 연결은 됐지만 이력이 없으면
    (또는 지도 클릭 → predictions 실시간 기록 연동이 아직 없으면) 빈 리스트를 반환한다.
    클릭 1번이 업종 수만큼 여러 행으로 남는 설계라 좌표로 GROUP BY해서 중복을 없앴다.

    TODO(미연결): 반환값엔 좌표만 있고 동 이름이 없다. app.py에 _nearest_dong()/
    _dong_name_map()이 실제로 추가되면, 호출부에서 그 함수로 좌표 → 동 이름 변환 후
    화면에 붙일 것 — 지도 클릭 판정 로직(A-1, 최근접 중심점 방식)과 기준을 맞추기 위해
    일부러 SQL에서 좌표 매칭을 따로 흉내내지 않았다. 지금은 아직 그 함수들이 없어서
    어느 render 함수에서도 호출하지 않는다.
    """
    if not user_id:
        return None
    db = _load_db_module()
    engine = db.get_engine()
    if engine is None:
        return None

    from sqlalchemy import text

    sql = text(
        "SELECT query_lat, query_lng, MAX(created_at) AS last_viewed "
        "FROM predictions "
        "WHERE user_id = :user_id AND query_type = 'new_location' "
        "GROUP BY query_lat, query_lng "
        "ORDER BY last_viewed DESC "
        "LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"user_id": user_id, "limit": limit}).mappings().all()
    return [dict(row) for row in rows]


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
# 더미 데이터: 내 매장
# 계정당 매장 1개 (users.store_id가 단일 FK). 상권이 여러 개인 점주는 계정을 따로 관리한다는
# 전제라서 매장 선택 UI는 두지 않는다.
# TODO: 실제로는 stores/store_snapshots 테이블을 store_id로 조회해서 대체
# ────────────────────────────────────────────────
DUMMY_STORES = {
    "MA010120220813467754": {
        "store_id": "MA010120220813467754",
        "store_name": "카페 온기",  # DB 조회 실패 시에만 쓰는 폴백 이름 (실제로는 store_snapshots.store_name 사용)
        "current_industry_code": "카페",
        "dong_code": "역삼동",
        "is_closed": False,
        "first_seen_snapshot": "2025-03-01",
        "last_seen_snapshot": "2026-08-20",
        "snapshots": [
            {
                "snapshot_date": "2026-08-01",
                "industry_code": "카페",
                "is_closed_next": False,
                "transitioned_next": None,
            },
            {
                "snapshot_date": "2026-05-01",
                "industry_code": "카페",
                "is_closed_next": False,
                "transitioned_next": None,
            },
            {
                "snapshot_date": "2025-11-01",
                "industry_code": "음식점",
                "is_closed_next": False,
                "transitioned_next": "카페",
            },
        ],
    },
}


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
# TODO: 실제로는 로그인 세션에서 user_id를 받아 DB 조회로 대체
# ────────────────────────────────────────────────
def init_session_state():
    if "user_info" not in st.session_state:
        # users 테이블: user_id, user_type, store_id(owner only), login_id, password_hash
        st.session_state.user_info = {
            "user_id": "u_0001",
            "login_id": "parkminha76",
            "user_type": "owner",  # "owner" | "admin" | 그 외
            "store_id": "MA010120220813467754",  # owner가 아니면 None
        }

    if "display_name" not in st.session_state:
        # 헤더 인사말 등에 쓰는 이름. login_id 대신 실제 상호명(store_snapshots.store_name)을
        # DB에서 가져와 쓰고, DB 연결이 없거나 못 찾으면 더미 상호명 → login_id 순으로 폴백한다.
        store_id = st.session_state.user_info.get("store_id")
        fallback_name = DUMMY_STORES.get(store_id, {}).get("store_name")
        st.session_state.display_name = (
            get_store_name(store_id)
            or fallback_name
            or st.session_state.user_info["login_id"]
        )

    if "predictions" not in st.session_state:
        # predictions 테이블 (FK user_id로 필터된 결과)
        st.session_state.predictions = [
            {
                "query_type": "폐업 예측",
                "store_id": "MA010120220813467754",
                "store_name": "카페 온기",
                "industry_code": "카페",
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
                "industry_code": "분식점",
                "score": 0.67,
                "shap_top_features": [
                    {"feature": "유동인구", "shap_value": 0.15},
                    {"feature": "동일업종 밀집도", "shap_value": -0.10},
                ],
            },
        ]

    if "favorite_regions" not in st.session_state:
        # predictions(user_id=본인) → stores → administrative_dongs 집계 (관제실의 인기 조회
        # 지역과 같은 방식 + WHERE user_id=본인).
        # TODO(데모용 임시 폴백): 지금은 predictions가 비어 있어서 항상 빈 리스트가 오는데,
        # 화면 확인을 위해 "DB 연결은 됐지만 이력이 없는" 경우도 더미로 채워서 보여준다.
        # 실제로 조회 이력이 쌓이기 시작하면 db_regions가 더 이상 빈 리스트가 아니게 되므로
        # 이 폴백은 자동으로 안 쓰이게 된다 — 그래도 실제 운영 전엔 이 폴백 자체를 지우는 게 맞다.
        db_regions = get_favorite_regions(st.session_state.user_info["user_id"])
        st.session_state.favorite_regions = db_regions or [
            {"dong_name": "역삼동", "count": 8},
            {"dong_name": "서교동", "count": 5},
            {"dong_name": "성수동", "count": 3},
        ]


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
        st.subheader("👤 내 정보", help="users 테이블: login_id, user_type, store_id")

        user = st.session_state.user_info

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("아이디 (store_id)", value=user["store_id"], disabled=True)
        with col2:
            st.text_input("사용자 유형 (user_type)", value=user["user_type"], disabled=True)


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
                st.success("비밀번호가 변경되었습니다. (구현 예정)")


@st.dialog("⚠️ 회원 탈퇴")
def show_delete_account_dialog():
    st.error("탈퇴 시 계정 정보가 삭제되며 복구할 수 없습니다.")
    confirm = st.checkbox("안내 사항을 확인했습니다.")
    if st.button("회원 탈퇴", type="primary", disabled=not confirm):
        # TODO: 실제 탈퇴 처리 로직 (users 레코드 삭제/비활성화)
        st.error("회원 탈퇴 처리 (구현 예정)")


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

        store = DUMMY_STORES[st.session_state.user_info["store_id"]]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("상호명", st.session_state.display_name)
        col2.metric("현재 업종", store["current_industry_code"])
        col3.metric("위치 (동)", store["dong_code"])
        col4.metric("폐업 여부", "폐업" if store["is_closed"] else "영업 중")

        st.caption(
            f"📅 데이터 수집 기간: {store['first_seen_snapshot']} ~ {store['last_seen_snapshot']}"
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

        store = DUMMY_STORES[st.session_state.user_info["store_id"]]
        snapshots = store["snapshots"]
        if not snapshots:
            st.info("아직 기록된 스냅샷이 없습니다.")
            return

        for snap in snapshots:
            with st.container(border=True):
                date_col, badge_col = st.columns([2, 5], vertical_alignment="center")
                with date_col:
                    st.markdown(f"**{snap['snapshot_date']}**")
                with badge_col:
                    with st.container(horizontal=True):
                        st.badge(snap["industry_code"], icon="🏷️", color="blue")
                        if snap["is_closed_next"]:
                            st.badge("다음 폐업 예정", icon="⚠️", color="red")
                        else:
                            st.badge("정상 운영 유지", icon="✅", color="green")
                        if snap["transitioned_next"]:
                            st.badge(f"→ {snap['transitioned_next']} 전환", icon="🔄", color="orange")


# ────────────────────────────────────────────────
# 4. 내 분석 히스토리 (predictions)
# ────────────────────────────────────────────────
def render_prediction_history():
    with st.container(border=True):
        st.subheader(
            "📜 내 분석 히스토리",
            help=(
                "predictions 테이블: query_type, store_id, industry_code, score (FK user_id로 필터) "
                "· 날짜 컬럼이 없어 최근 조회순(prediction_id 역순)으로 정렬"
            ),
        )

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
                    st.caption(f"{_prediction_target_label(pred)} · {pred['industry_code']}")
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
    st.caption(f"{_prediction_target_label(pred)} · {pred['industry_code']}")

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
# 5. 내가 관심있는 지역 TOP 3
# ────────────────────────────────────────────────
def render_favorite_regions():
    with st.container(border=True):
        st.subheader(
            "📍 내가 관심있는 지역 TOP 3",
            help=(
                "predictions(user_id=본인) → stores → administrative_dongs 집계 "
                "· dong_name별 조회 횟수 상위 3개"
            ),
        )

        regions = st.session_state.favorite_regions
        if not regions:
            st.info("아직 조회 이력이 없습니다.")
            return

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

    # store_id가 있는 owner에게만 매장 관련 섹션 노출
    user = st.session_state.user_info
    if user["user_type"] == "owner" and user.get("store_id"):
        render_store_status()
        render_favorite_regions()
        render_store_snapshots()

    render_prediction_history()

    st.divider()
    render_danger_zone()


if __name__ == "__main__":
    main()
