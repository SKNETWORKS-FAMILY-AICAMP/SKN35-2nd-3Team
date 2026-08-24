"""
app/shared/db.py

TiDB Cloud(Serverless) 연결 유틸.

TiDB Cloud는 SSL 연결을 요구한다. 최신 TiDB Cloud Serverless는 별도 CA 인증서
파일 없이도 시스템 기본 CA로 검증 가능한 경우가 많지만, 연결이 안 되면
TiDB Cloud 콘솔의 Connect 다이얼로그에서 CA 인증서 다운로드 안내를 따를 것.

필요 패키지: pymysql, sqlalchemy, python-dotenv, cryptography
    pip install pymysql sqlalchemy python-dotenv cryptography
"""
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

TIDB_HOST = os.environ["TIDB_HOST"]
TIDB_PORT = os.environ.get("TIDB_PORT", "4000")
TIDB_USER = os.environ["TIDB_USER"]
TIDB_PASSWORD = os.environ["TIDB_PASSWORD"]
TIDB_DB_NAME = os.environ["TIDB_DB_NAME"]


def get_engine(echo: bool = False):
    """SQLAlchemy 엔진 생성. pandas.to_sql / pd.read_sql 에 그대로 넘겨 쓸 수 있다."""
    url = (
        f"mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}"
        f"@{TIDB_HOST}:{TIDB_PORT}/{TIDB_DB_NAME}"
    )
    # TiDB Cloud는 SSL 필수. verify_cert=False로 완화할 수도 있지만 데모 단계에서만 권장.
    connect_args = {"ssl": {"ssl_verify_cert": True, "ssl_verify_identity": True}}
    return create_engine(url, connect_args=connect_args, echo=echo, pool_pre_ping=True)


def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.exec_driver_sql("SELECT VERSION()")
        print("연결 성공:", result.fetchone()[0])


if __name__ == "__main__":
    test_connection()
