"""로그인/세션 관리 (기존점주: store_id + 고정 비밀번호, 관리자: 고정 계정 5개)

설계 근거 (write_user.py 사용 예 + schema.sql + seoul-biz-ui-logic.md 6번, 2026-08-27
사용자 확정): password_hash 컬럼이 NOT NULL이라 "비밀번호 없이 로그인"은 설계상
어색하다는 지적을 반영해서, 전부 실제로 입력받는 고정 데모 비밀번호로 통일했다.
- 기존점주(owner): 아이디=store_id, 비밀번호는 전부 "1234"로 통일. 처음 로그인하는
  store_id면 users에 계정을 그 자리에서 자동 생성.
- 예비창업자(founder): 아직 가게가 없으니 store_id가 없음 -> 아이디/비밀번호를
  직접 입력하는 일반적인 회원가입/로그인.
- 관리자(admin): "admin1"~"admin5" 5개 고정 계정, 비밀번호는 아이디와 동일
  (admin1/admin1 ... admin5/admin5). support_actions.admin_user_id가
  users.user_id를 FK로 참조하므로, 관리자도 users 테이블에 실제 행 하나는
  있어야 함 -> 최초 로그인 시 자동 생성.
"""

import hashlib
import secrets

import streamlit as st
from sqlalchemy import text

from .db import get_engine

# ---------------------------------------------------------------
# 고정 데모 계정 규칙 (전부 사용자 확정, 2026-08-27)
# ---------------------------------------------------------------
OWNER_DEMO_PASSWORD = "1234"
ADMIN_LOGIN_IDS = [f"admin{i}" for i in range(1, 6)]  # admin1~admin5


def _hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------
_SESSION_KEYS = ("user_id", "user_type", "store_id", "login_id")

# st.session_state는 WebSocket 연결 하나에만 붙어 있는 값이라, 새로고침(F5)처럼
# 연결이 새로 맺어지면 그냥 다 사라져서 로그인이 풀린다(2026-08-31 사용자 제보).
# 반면 URL은 새로고침해도 그대로 남아있으므로, 로그인할 때 발급한 토큰을 URL 쿼리
# 파라미터에 붙여두고 그 토큰으로 세션 정보를 찾을 수 있는 서버 메모리 저장소를
# 둔다. 모듈 전역 dict라 앱 프로세스가 살아있는 동안(재배포/재시작 전까지)만
# 유지되지만, 그 정도면 "새로고침에 로그인이 풀리는" 원래 문제는 충분히 해결된다.
_SESSION_STORE: dict[str, dict] = {}
_SESSION_TOKEN_PARAM = "st"

# 로그인/로그아웃/역할 전환 시 같이 비워야 하는 화면 상태(2026-08-30 추가).
# chatbot_history/chatbot_context를 안 비우면, 예비창업자로 로그인해서 챗봇과
# 대화한 뒤 기존점주로 재로그인해도 이전 역할일 때의 대화 기록이 그대로
# 남아있는 문제가 있었음(사용자 실사용 중 발견) — app.py가 아니라 로그인
# 함수 쪽에서 지워야 owner/founder/admin 어느 경로로 재로그인해도 빠짐없이
# 처리된다.
_CHAT_SESSION_KEYS = ("chatbot_history", "chatbot_context")


def is_logged_in() -> bool:
    return st.session_state.get("user_id") is not None


def home_url() -> str:
    """로고 클릭 등으로 앱 홈("/")으로 돌아갈 때 쓸 링크. 로그인 상태면 세션 토큰을
    쿼리 파라미터로 붙여서 돌려준다 — 로고가 지금까지 그냥 <a href="/">라 클릭하면
    브라우저가 진짜 페이지를 새로 요청했는데(세션 유지용 st.page_link가 아니라 raw
    HTML 링크라 Streamlit 프론트엔드가 클릭을 가로채지 못함), 그 URL에 토큰이 아예
    없어서 restore_session_from_url()이 복원할 게 없어 로그아웃된 것처럼 보였다
    (2026-08-31 확인). 토큰을 링크에 미리 심어두면 새로 열리는 세션에서 그대로
    복원된다."""
    token = st.session_state.get("_session_token")
    return f"/?{_SESSION_TOKEN_PARAM}={token}" if token else "/"


def current_user() -> dict | None:
    """로그인 상태면 {'user_id','user_type','store_id','login_id'}, 아니면 None."""
    if not is_logged_in():
        return None
    return {k: st.session_state.get(k) for k in _SESSION_KEYS}


def get_screen_mode() -> str:
    """진입 라우팅 (seoul-biz-ui-logic.md 1번): GUEST/NEW_MEMBER/OWNER/ADMIN"""
    user = current_user()
    if user is None:
        return "GUEST"
    if user["user_type"] == "admin":
        return "ADMIN"
    if user["user_type"] == "owner" and user["store_id"]:
        return "OWNER"
    return "NEW_MEMBER"


def _set_session(user_id: str, user_type: str, store_id: str | None, login_id: str) -> None:
    st.session_state["user_id"] = user_id
    st.session_state["user_type"] = user_type
    st.session_state["store_id"] = store_id
    st.session_state["login_id"] = login_id
    _clear_chat_session()

    token = secrets.token_urlsafe(24)
    _SESSION_STORE[token] = {
        "user_id": user_id, "user_type": user_type, "store_id": store_id, "login_id": login_id,
    }
    st.session_state["_session_token"] = token
    st.query_params[_SESSION_TOKEN_PARAM] = token


def restore_session_from_url() -> None:
    """새로고침 등으로 session_state가 비어있을 때 URL의 토큰으로 로그인 상태를
    복원한다. 로그인 여부를 확인하는 페이지라면(app.py, login.py, mypage.py,
    admin_dashboard.py 전부) auth.current_user()를 부르기 전에 가장 먼저 호출해야
    한다.

    로그인 직후에는 반대 방향도 챙겨야 한다 — login.py는 로그인 성공 즉시
    st.switch_page()로 다른 페이지로 넘어가는데, 그 이동이 직전에 _set_session()이
    URL에 심어둔 토큰을 들고 가지 않고 도착 페이지의 맨 URL로 가버린다(확인함,
    2026-08-31). 그래서 로그인은 됐는데 URL에는 토큰이 없는 상태로 도착하는데,
    이 함수가 도착한 페이지에서 한 번 더 실행되니 그때 토큰을 다시 심어준다."""
    if not is_logged_in():
        token = st.query_params.get(_SESSION_TOKEN_PARAM)
        if not token:
            return
        info = _SESSION_STORE.get(token)
        if info is None:
            return
        st.session_state["user_id"] = info["user_id"]
        st.session_state["user_type"] = info["user_type"]
        st.session_state["store_id"] = info["store_id"]
        st.session_state["login_id"] = info["login_id"]
        st.session_state["_session_token"] = token
        return

    my_token = st.session_state.get("_session_token")
    if my_token and st.query_params.get(_SESSION_TOKEN_PARAM) != my_token:
        st.query_params[_SESSION_TOKEN_PARAM] = my_token


def logout() -> None:
    token = st.session_state.get("_session_token")
    if token:
        _SESSION_STORE.pop(token, None)
    for key in _SESSION_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("_session_token", None)
    if _SESSION_TOKEN_PARAM in st.query_params:
        del st.query_params[_SESSION_TOKEN_PARAM]
    _clear_chat_session()


def _clear_chat_session() -> None:
    """로그인/로그아웃/역할 전환 시 챗봇 대화 기록·컨텍스트를 비운다 — 이전
    사용자(또는 이전 역할)의 대화가 다음 세션에 남아 보이는 것을 막기 위함."""
    for key in _CHAT_SESSION_KEYS:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------
# 조회 헬퍼
# ---------------------------------------------------------------
def find_user_by_login_id(login_id: str) -> dict | None:
    engine = get_engine()
    if engine is None:
        return None
    sql = text(
        "SELECT user_id, user_type, store_id, login_id, password_hash "
        "FROM users WHERE login_id = :login_id"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"login_id": login_id}).mappings().first()
    return dict(row) if row else None


def store_exists(store_id: str) -> bool:
    engine = get_engine()
    if engine is None:
        return False
    sql = text("SELECT 1 FROM stores WHERE store_id = :store_id")
    with engine.connect() as conn:
        return conn.execute(sql, {"store_id": store_id}).first() is not None


# ---------------------------------------------------------------
# 기존점주(owner) — store_id + 고정 비밀번호("1234") 로그인
# ---------------------------------------------------------------
def login_owner_by_store_id(store_id: str, password: str) -> tuple[bool, str]:
    """아이디=store_id, 비밀번호는 전부 "1234"로 통일. 반환: (성공여부, 에러 메시지)."""
    store_id = (store_id or "").strip()
    if not store_id or not password:
        return False, "가게 코드와 비밀번호를 입력해주세요."
    if password != OWNER_DEMO_PASSWORD:
        return False, "아이디 또는 비밀번호가 올바르지 않아요."
    if not store_exists(store_id):
        return False, "존재하지 않는 가게 코드예요. 다시 확인해주세요."

    existing = find_user_by_login_id(store_id)
    if existing is not None:
        if existing["user_type"] != "owner":
            return False, "이 코드는 기존점주 계정으로 등록돼 있지 않아요."
        _set_session(existing["user_id"], existing["user_type"], existing["store_id"], existing["login_id"])
        return True, ""

    from .write_user import create_user
    user_id = create_user(
        user_type="owner",
        login_id=store_id,
        password_hash=_hash_password(OWNER_DEMO_PASSWORD),
        store_id=store_id,
    )
    _set_session(user_id, "owner", store_id, store_id)
    return True, ""


# ---------------------------------------------------------------
# 예비창업자(founder) — 아이디/비밀번호 직접 입력
# ---------------------------------------------------------------
def signup_founder(login_id: str, password: str) -> tuple[bool, str]:
    login_id = (login_id or "").strip()
    if not login_id or not password:
        return False, "아이디와 비밀번호를 입력해주세요."
    if login_id in ADMIN_LOGIN_IDS:
        return False, "이 아이디는 사용할 수 없어요. 다른 아이디를 입력해주세요."
    if find_user_by_login_id(login_id) is not None:
        return False, "이미 사용 중인 아이디예요."
    if store_exists(login_id):
        # users.login_id는 uq_login_id로 owner/founder/admin 전체에서 유일해야 하는데,
        # 기존점주는 store_id를 그대로 로그인 아이디로 쓴다(login_owner_by_store_id).
        # 그래서 예비창업자가 아직 아무도 로그인한 적 없는 실제 가게 코드를 자기
        # 아이디로 먼저 선점해버리면, 그 가게의 진짜 사장님은 영영 그 코드로 로그인을
        # 못 하게 된다 — 애초에 가게 코드와 겹치는 아이디는 막는다.
        return False, "이 아이디는 사용할 수 없어요. 다른 아이디를 입력해주세요."

    from .write_user import create_user
    user_id = create_user(
        user_type="founder",
        login_id=login_id,
        password_hash=_hash_password(password),
        store_id=None,
    )
    _set_session(user_id, "founder", None, login_id)
    return True, ""


def login_founder(login_id: str, password: str) -> tuple[bool, str]:
    user = find_user_by_login_id((login_id or "").strip())
    if user is None or user["user_type"] != "founder":
        return False, "아이디 또는 비밀번호가 올바르지 않아요."
    if user["password_hash"] != _hash_password(password or ""):
        return False, "아이디 또는 비밀번호가 올바르지 않아요."
    _set_session(user["user_id"], user["user_type"], user["store_id"], user["login_id"])
    return True, ""


# ---------------------------------------------------------------
# 관리자(admin) — admin1~admin5, 비밀번호는 아이디와 동일
# ---------------------------------------------------------------
def login_admin(login_id: str, password: str) -> tuple[bool, str]:
    login_id = (login_id or "").strip()
    if login_id not in ADMIN_LOGIN_IDS or (password or "") != login_id:
        return False, "아이디 또는 비밀번호가 올바르지 않아요."

    user = find_user_by_login_id(login_id)
    if user is None:
        from .write_user import create_user
        user_id = create_user(
            user_type="admin",
            login_id=login_id,
            password_hash=_hash_password(login_id),
            store_id=None,
        )
        _set_session(user_id, "admin", None, login_id)
    else:
        _set_session(user["user_id"], user["user_type"], user["store_id"], user["login_id"])
    return True, ""


# ---------------------------------------------------------------
# 통합 로그인 — 화면 하나(아이디 + 비밀번호)로 owner/founder/admin 전부 처리.
# 관리자를 따로 뺄 필요 없이, 그냥 admin1~admin5 아이디로 로그인하면 자동으로
# admin으로 판별되게 한다(사용자 요청, 2026-08-27).
# ---------------------------------------------------------------
def login_unified(login_id: str, password: str) -> tuple[bool, str]:
    """
    판별 순서:
      1) 아이디가 admin1~admin5 중 하나면 admin으로 처리(비밀번호=아이디와 동일해야 함).
      2) 이미 users에 등록된 아이디면(founder/owner) 각자 방식으로 비밀번호 검증
         (founder는 본인이 정한 비밀번호, owner는 고정값 "1234").
      3) 등록된 적 없는 아이디인데 stores 테이블의 store_id와 같으면 기존점주 데모
         로그인("1234" 입력 시 최초 로그인이면 자동 가입).
    화면 쪽에서는 이 함수 하나만 부르고, 로그인 성공 후 current_user()['user_type']이
    'admin'이면 관리자 대시보드로 바로 이동시키면 된다.
    """
    login_id = (login_id or "").strip()
    if not login_id:
        return False, "아이디를 입력해주세요."

    if login_id in ADMIN_LOGIN_IDS:
        return login_admin(login_id, password)

    existing = find_user_by_login_id(login_id)
    if existing is not None:
        if existing["user_type"] == "founder":
            return login_founder(login_id, password)
        if existing["user_type"] == "owner":
            return login_owner_by_store_id(login_id, password)
        return False, "아이디 또는 비밀번호가 올바르지 않아요."

    if store_exists(login_id):
        return login_owner_by_store_id(login_id, password)

    return False, "아이디 또는 비밀번호가 올바르지 않아요."