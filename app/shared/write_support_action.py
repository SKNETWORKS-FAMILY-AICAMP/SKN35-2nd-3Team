"""
app/shared/write_support_action.py

support_actions 테이블에 관리자의 개입 조치를 기록한다.
관리자 대시보드의 "조치 등록" 폼 제출 시 이 함수 하나만 호출하면 된다.
"""
from datetime import date
from sqlalchemy import text
from app.shared.db import get_engine


def log_support_action(store_id: str, admin_user_id: str, action_type: str,
                        action_date: date | None = None, notes: str | None = None):
    engine = get_engine()
    values = {
        'store_id': store_id,
        'admin_user_id': admin_user_id,
        'action_type': action_type,
        'action_date': action_date or date.today(),
        'follow_up_closure_status': None,  # 나중에 실제 폐업 여부 확인되면 별도 UPDATE로 채운다.
        'notes': notes,
    }
    sql = text(
        "INSERT INTO support_actions (store_id, admin_user_id, action_type, action_date, "
        "follow_up_closure_status, notes) "
        "VALUES (:store_id, :admin_user_id, :action_type, :action_date, "
        ":follow_up_closure_status, :notes)"
    )
    with engine.begin() as conn:
        conn.execute(sql, values)


# 사용 예 (관리자 대시보드의 "조치 등록" 폼 제출 시):
#   from app.shared.write_support_action import log_support_action
#   log_support_action(store_id=selected_store_id, admin_user_id=session_admin_id,
#                       action_type='전화상담', notes='임대료 지원 프로그램 안내함')
