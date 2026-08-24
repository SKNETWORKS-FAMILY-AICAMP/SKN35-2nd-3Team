"""
app/shared/write_user.py

users 테이블에 로그인/회원가입 결과를 써넣는다.
로그인/회원가입 로직(app/shared/auth.py)에서 값이 다 모이면 이 함수 하나만 호출하면 된다.
"""
import uuid
from datetime import datetime
from sqlalchemy import text
from app.shared.db import get_engine


def create_user(user_type: str, login_id: str, password_hash: str, store_id: str | None = None) -> str:
    """
    user_type: 'owner' | 'founder' | 'admin'
    store_id : user_type='owner'일 때만 값 전달, 나머지는 None
    반환값: 생성된 user_id
    """
    engine = get_engine()
    values = {
        'user_id': str(uuid.uuid4()),
        'user_type': user_type,
        'store_id': store_id,
        'login_id': login_id,
        'password_hash': password_hash,
        'created_at': datetime.now(),
    }
    sql = text(
        "INSERT INTO users (user_id, user_type, store_id, login_id, password_hash, created_at) "
        "VALUES (:user_id, :user_type, :store_id, :login_id, :password_hash, :created_at)"
    )
    with engine.begin() as conn:
        conn.execute(sql, values)
    return values['user_id']


# 사용 예 (기존점주 로그인 화면에서):
#   from app.shared.write_user import create_user
#   user_id = create_user(user_type='owner', login_id=store_id,
#                          password_hash=make_demo_password(store_id), store_id=store_id)
