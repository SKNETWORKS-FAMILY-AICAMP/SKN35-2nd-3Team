# -*- coding: utf-8 -*-
"""
user_view_history 테이블 write/query 헬퍼.

app/shared/write_dong_view.py와 짝을 이루는 개인용 버전 — 클릭 구현 쪽에서는
increment_dong_view()(관리자용 집계)와 이 파일의 increment_user_view()(개인
기록)를 로그인 상태일 때 같이 호출하면 됨:

    from shared.write_dong_view import increment_dong_view
    from shared.write_user_view import increment_user_view

    # _nearest_dong()으로 dong_code 판정한 직후
    increment_dong_view(dong_code, user_type)          # 관리자용 (게스트 자동 무시)
    increment_user_view(user_id, user_type)             # 개인용 (게스트/user_id 없으면 자동 무시)

마이페이지에서는 get_my_recent_views(user_id)만 부르면 됨:

    from shared.write_user_view import get_my_recent_views
    rows = get_my_recent_views(user_id, limit=5)
    for r in rows:
        st.write(f"{r['dong_code']} - {r['last_viewed_at']}")   # 동 이름 변환은 app.py의 _dong_name_map() 재사용
"""
from typing import Any, Optional

from sqlalchemy import text

from .db import get_engine

_UPSERT_SQL = text("""
    INSERT INTO user_view_history (user_id, dong_code, view_count, last_viewed_at)
    VALUES (:user_id, :dong_code, 1, NOW())
    ON DUPLICATE KEY UPDATE
        view_count = view_count + 1,
        last_viewed_at = NOW()
""")

_RECENT_VIEWS_SQL = text("""
    SELECT dong_code, view_count, last_viewed_at
    FROM user_view_history
    WHERE user_id = :user_id
    ORDER BY last_viewed_at DESC
    LIMIT :limit
""")


def increment_user_view(user_id: Optional[str], dong_code: Optional[str]) -> None:
    """지도에서 dong_code가 클릭될 때마다 호출.

    user_id가 없으면(게스트, 로그인 안 함) 에러 없이 조용히 아무것도 안 하고
    리턴함 — increment_dong_view()와 동일한 원칙(호출부가 로그인 여부를
    미리 체크할 필요 없이 그냥 무조건 호출하면 되도록).
    """
    if not user_id or not dong_code:
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, {"user_id": user_id, "dong_code": dong_code})


def get_my_recent_views(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """마이페이지 "내가 본 지역" 목록. 로그인 안 했으면(user_id 없음) 빈 리스트."""
    if not user_id:
        return []

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_RECENT_VIEWS_SQL, {"user_id": user_id, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]