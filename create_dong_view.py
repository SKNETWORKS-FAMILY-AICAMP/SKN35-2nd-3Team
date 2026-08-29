# -*- coding: utf-8 -*-
"""
dong_view_stats 테이블 하나만 새로 만드는 1회성 스크립트.

기존 13개 테이블은 절대 건드리지 않음 — CREATE TABLE IF NOT EXISTS 문 하나만
실행함(DROP/ALTER 없음, 다른 테이블 참조는 FK로 읽기만 함).

실행 위치: 프로젝트 루트(C:\\SKN35-2nd-3Team)에서
    python create_dong_view_table.py

이미 존재하면(예: 실수로 두 번 실행) "already exists" 에러 없이 조용히
넘어가고 확인 메시지만 출력됨 (IF NOT EXISTS 덕분).
"""
import sys
from pathlib import Path

# app/shared/db.py의 get_engine()을 그대로 재사용
# (load_to_tidb.py 등 기존 스크립트들과 동일한 "app 폴더를 sys.path에 넣는" 방식)
_APP_DIR = str(Path(__file__).resolve().parent / "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from shared.db import get_engine  # noqa: E402

DDL = """
CREATE TABLE IF NOT EXISTS dong_view_stats (
    dong_code       VARCHAR(20)                  NOT NULL,
    view_count      INT                           NOT NULL DEFAULT 0,
    last_viewed_at  DATETIME                      NOT NULL DEFAULT CURRENT_TIMESTAMP
                                                    ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (dong_code),
    FOREIGN KEY (dong_code) REFERENCES administrative_dongs(dong_code)
);
"""


def main():
    engine = get_engine()

    with engine.connect() as conn:
        existing = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = 'dong_view_stats'"
        ).scalar()

    if existing:
        print("dong_view_stats 테이블이 이미 존재합니다 — 아무것도 하지 않고 종료합니다.")
        return

    with engine.begin() as conn:
        # schema.sql 만들 때 겪었던 것과 같은 이유(SQLAlchemy text()의 콜론 오인식 이슈 회피)로
        # exec_driver_sql 사용 — 이 DDL엔 ':' 플레이스홀더가 없어서 사실 text()도 되지만 통일함.
        conn.exec_driver_sql(DDL)

    print("dong_view_stats 테이블 생성 완료. (기존 13개 테이블은 전혀 건드리지 않았습니다)")


if __name__ == "__main__":
    main()