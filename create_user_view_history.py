# -*- coding: utf-8 -*-
"""
user_view_history 테이블 하나만 새로 만드는 1회성 스크립트.
create_dong_view_table.py와 동일한 패턴 — 기존 테이블은 전혀 안 건드림.

실행 위치: 프로젝트 루트에서
    python create_user_view_history_table.py
"""
import sys
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parent / "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from shared.db import get_engine  # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS user_view_history (
    user_id         VARCHAR(30)                  NOT NULL,
    dong_code       VARCHAR(20)                  NOT NULL,
    view_count      INT                           NOT NULL DEFAULT 0,
    last_viewed_at  DATETIME                      NOT NULL DEFAULT CURRENT_TIMESTAMP
                                                    ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, dong_code),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (dong_code) REFERENCES administrative_dongs(dong_code)
);
"""


def main():
    engine = get_engine()

    with engine.connect() as conn:
        existing = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'user_view_history'"
        ).scalar()

    if existing:
        print("user_view_history 테이블이 이미 존재합니다 — 아무것도 하지 않고 종료합니다.")
        return

    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)

    print("user_view_history 테이블 생성 완료. (기존 테이블은 전혀 건드리지 않았습니다)")


if __name__ == "__main__":
    main()