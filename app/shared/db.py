"""
app/shared/db.py — 프로젝트 공용 DB 커넥션

db/etl/03_load_to_tidb.py 가 `from app.shared.db import get_engine`로 가져다 씀.
나중에 Streamlit 앱(화면)에서도 이 모듈을 그대로 가져다 쓰면 됨 — 커넥션 로직은
여기 한 곳에만 두는 구조.

.env 키 이름은 전부 DB_ 접두사를 붙임 (USERNAME 등 OS가 이미 쓰는 이름과
겹치지 않게 하기 위함 — Windows는 USERNAME을 로그인 계정명으로 이미 갖고 있음):
    DB_HOST=...
    DB_PORT=4000
    DB_USERNAME=...
    DB_PASSWORD=...
    DB_DATABASE=...
    DB_SSL_CA=...(선택, 안 주면 certifi 기본 CA 번들 사용 — TiDB Cloud Serverless는
                   보통 공인 CA로 서명돼 있어서 이걸로 충분함)
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except Exception:
    # db/etl/*.py 처럼 streamlit 없이 순수 스크립트로 실행될 때도 동작하도록.
    def _cache_resource(fn):
        return fn

load_dotenv(override=True)
# DB_ 접두사를 쓰긴 하지만, override=True는 계속 안전장치로 남겨둠 — 다른 이름도
# OS/다른 라이브러리가 이미 쓰고 있을 가능성은 항상 있으니 .env가 항상 이기게.

try:
    import certifi
    _DEFAULT_CA = certifi.where()
except ImportError:
    _DEFAULT_CA = None


@_cache_resource
def get_engine() -> Engine | None:
    """.env(DB_HOST/DB_PORT/DB_USERNAME/DB_PASSWORD/DB_DATABASE)에서
    TiDB 접속 정보를 읽어 엔진 생성. DB_HOST/DB_USERNAME이 없으면 None 반환
    (호출부가 미연결 상태를 감지할 수 있게)."""
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USERNAME")
    if not host or not user:
        return None

    port = int(os.getenv("DB_PORT", "4000"))
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_DATABASE", "")

    # TiDB Cloud Serverless는 TLS 필수인 경우가 대부분.
    ca_path = os.getenv("DB_SSL_CA", _DEFAULT_CA)
    ssl_args = {"ssl": {"ca": ca_path}} if ca_path else {"ssl": {}}

    # f-string으로 직접 문자열을 이어붙이면 비밀번호에 @, :, / 같은 URL 특수문자가
    # 하나라도 들어있을 때 접속 문자열 자체가 깨져서(username/host 경계가 밀림)
    # "Missing user name prefix" 같은 엉뚱한 에러로 나타남. URL.create()는 각 값을
    # 안전하게 percent-encoding 해주므로 비밀번호에 어떤 문자가 있어도 안전함.
    url = URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )
    return create_engine(url, connect_args=ssl_args, pool_pre_ping=True)


def db_available() -> bool:
    return get_engine() is not None


if __name__ == "__main__":
    # 접속 확인용: python -m app.shared.db
    engine = get_engine()
    if engine is None:
        print("연결 정보 없음 (.env의 DB_HOST/DB_USERNAME 확인)")
    else:
        with engine.connect() as conn:
            print("연결 성공:", conn.engine.url.render_as_string(hide_password=True))