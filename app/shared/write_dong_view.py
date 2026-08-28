# -*- coding: utf-8 -*-
"""
dong_view_stats 테이블 write 헬퍼.

app/shared/write_user.py, write_prediction.py, write_support_action.py와
동일한 컨벤션(from .db import get_engine, 상대 import)으로 맞춤 —
app/shared/ 밑에 이 파일을 두고 다른 write_*.py와 나란히 쓰면 됨.

클릭 구현 쪽에서는 이 파일의 increment_dong_view() 딱 하나만 부르면 됨:

    from shared.write_dong_view import increment_dong_view
    increment_dong_view(dong_code, user_type)

user_type은 "카운트할지 말지"만 판단하는 데 쓰고 저장은 안 함 — 관리자
화면에서 owner/founder를 구분해서 보여줄 계획이 없어서, dong_view_stats
테이블 자체에 user_type 컬럼이 없음(동 하나당 딱 1행). "owner"/"founder"가
아닌 값(게스트 포함, 로그인 안 한 상태 등)이 들어오면 에러 없이 조용히
아무것도 안 하고 리턴함. 그래서 클릭 구현 쪽은 로그인 여부를 따로 체크할
필요 없이, 클릭이 일어날 때마다 이 함수를 그냥 무조건 호출하면 됨.
"""
from sqlalchemy import text

from .db import get_engine

_COUNTED_USER_TYPES = ("owner", "founder")

_UPSERT_SQL = text("""
    INSERT INTO dong_view_stats (dong_code, view_count, last_viewed_at)
    VALUES (:dong_code, 1, NOW())
    ON DUPLICATE KEY UPDATE
        view_count = view_count + 1,
        last_viewed_at = NOW()
""")


def increment_dong_view(dong_code: str, user_type: str) -> None:
    """지도에서 dong_code가 클릭될 때마다 호출.

    - user_type이 "owner" 또는 "founder"면 해당 동의 카운트를 1 올림
      (owner/founder 구분 없이 하나의 총 카운트로 합산됨).
    - 그 외(게스트, None, 빈 문자열 등)면 아무 것도 안 하고 조용히 리턴
      (에러를 던지지 않음 — 호출부에서 로그인 상태를 미리 체크 안 해도 안전하게
      그냥 호출만 하면 되도록 하기 위함).
    - dong_code가 비어있는 경우도 마찬가지로 조용히 무시.

    Args:
        dong_code: 클릭으로 판정된 행정동 코드 (_nearest_dong() 결과)
        user_type: "owner" / "founder" / 그 외(게스트 등, 전부 무시됨) —
                   카운트 여부 판단에만 쓰이고 DB엔 저장 안 됨.
    """
    if not dong_code or user_type not in _COUNTED_USER_TYPES:
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, {"dong_code": dong_code})