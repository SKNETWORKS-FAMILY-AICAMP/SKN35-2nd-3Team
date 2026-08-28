"""로그인 / 회원가입 — owner/founder/admin 전부 이 화면 하나로 처리한다.

app.py는 메인 화면(지도+패널)만 보여주고, 로그인/회원가입 입력 폼은 이 페이지로
분리했다(사용자 요청, 2026-08-27: "app.py에는 메인화면만, 로그인/회원가입은
다른 데로"). 파일명 앞에 0을 붙여서 사이드바 맨 위에 오게 했다.

관리자 아이디(admin1~admin5)로 로그인하면 auth.login_unified가 자동으로 admin을
판별해서, 로그인 성공 즉시 관리자 대시보드로 바로 이동한다(별도 관리자 로그인
화면 없음).
"""

import sys
from pathlib import Path

# app/app.py와 같은 이유(파일명 충돌 회피)로 app/ 폴더 자체를 sys.path에 넣고
# "shared"를 최상위 이름으로 바로 가져온다. pages/*.py는 app/pages/ 밑에 있으니
# 조상 2단계 위가 app/ 폴더다.
_APP_DIR = str(Path(__file__).resolve().parents[1])
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

import streamlit as st

from shared import auth

st.set_page_config(page_title="로그인", layout="centered")
st.title("로그인")

user = auth.current_user()
if user is not None:
    label = {"owner": "기존점주", "founder": "예비창업자", "admin": "관리자"}.get(
        user["user_type"], user["user_type"]
    )
    st.success(f"이미 {label}로 로그인돼 있어요 ({user['login_id']}).")
    if st.button("로그아웃"):
        auth.logout()
        st.rerun()
    st.page_link("app.py", label="메인으로 이동")
    if user["user_type"] == "admin":
        st.page_link("pages/admin_dashboard.py", label="관리자 대시보드로 이동")
    st.stop()

st.subheader("로그인")
login_id = st.text_input("아이디 (기존점주는 가게 코드)", key="login_id_input")
password = st.text_input("비밀번호 (기존점주는 1234)", type="password", key="login_pw_input")
if st.button("로그인", key="btn_unified_login"):
    ok, msg = auth.login_unified(login_id, password)
    if ok:
        if auth.current_user()["user_type"] == "admin":
            st.switch_page("pages/admin_dashboard.py")
        else:
            st.switch_page("app.py")
    else:
        st.error(msg)

st.divider()
st.subheader("예비창업자 회원가입")
sid = st.text_input("아이디", key="founder_signup_id")
spw = st.text_input("비밀번호", type="password", key="founder_signup_pw")
if st.button("가입 후 로그인", key="btn_founder_signup"):
    ok, msg = auth.signup_founder(sid, spw)
    if ok:
        st.switch_page("app.py")
    else:
        st.error(msg)