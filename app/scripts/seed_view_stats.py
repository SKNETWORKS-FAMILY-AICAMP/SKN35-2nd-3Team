# -*- coding: utf-8 -*-
"""
dong_view_stats / user_view_history 더미 데이터 시드 스크립트 (발표용).

실제 지도를 계속 클릭해서 데이터를 쌓는 건 발표 중엔 번거로우니, 그럴듯한
분포로 미리 채워둔다.
  - dong_view_stats: 모든 동에 조회수를 랜덤 배정하되, "지금 뜨는 동네" 상위
    후보(_hot_dong_ranking과 무관하게 그냥 총 매장수 기준 상위 동)에 조회수를
    더 몰아줘서 "인기 지역"이 그럴듯하게 보이게 함.
  - user_view_history: users 테이블의 owner/founder 계정마다 몇 개 동을
    "내가 본 지역"으로 기록해서 마이페이지가 비어있지 않게 함.

실행:
    uv run python app/scripts/seed_view_stats.py
    uv run python app/scripts/seed_view_stats.py --reset   # 기존 더미 지우고 다시 시드

주의: 이미 실제 클릭으로 쌓인 데이터가 있다면 --reset 없이 실행 시 그 위에
조회수가 더해진다(증분 UPSERT). 완전히 새로 채우고 싶으면 --reset 사용.
"""
import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parents[1])
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

from sqlalchemy import text

from shared.db import get_engine

random.seed(42)  # 발표 때마다 같은 분포가 나오도록 고정


def _all_dong_codes(conn) -> list[str]:
    rows = conn.execute(text("SELECT dong_code FROM administrative_dongs")).mappings().all()
    return [r["dong_code"] for r in rows]


def _store_counts_by_dong(conn) -> dict[str, int]:
    """총 매장수가 많은 동일수록 조회수도 그럴듯하게 많이 몰아주기 위한 가중치."""
    rows = conn.execute(
        text("SELECT dong_code, COUNT(*) AS n FROM stores GROUP BY dong_code")
    ).mappings().all()
    return {r["dong_code"]: r["n"] for r in rows}


def _owner_founder_users(conn) -> list[dict]:
    rows = conn.execute(
        text("SELECT user_id, user_type FROM users WHERE user_type IN ('owner', 'founder')")
    ).mappings().all()
    return [dict(r) for r in rows]


def _random_timestamp_within_days(days: int) -> datetime:
    delta_seconds = random.randint(0, days * 24 * 3600)
    return datetime.now() - timedelta(seconds=delta_seconds)


def seed_dong_view_stats(conn, dong_codes: list[str], store_counts: dict[str, int]) -> int:
    max_count = max(store_counts.values()) if store_counts else 1
    rows_inserted = 0
    for dong_code in dong_codes:
        weight = store_counts.get(dong_code, 0) / max_count if max_count else 0
        # 매장 많은 동(가중치 높음)일수록 조회수를 30~300, 적은 동은 1~40 사이로.
        base = int(30 + weight * 270)
        view_count = random.randint(max(1, base - 20), base + 20)
        last_viewed = _random_timestamp_within_days(14)
        conn.execute(
            text(
                """
                INSERT INTO dong_view_stats (dong_code, view_count, last_viewed_at)
                VALUES (:dong_code, :view_count, :last_viewed_at)
                ON DUPLICATE KEY UPDATE
                    view_count = :view_count,
                    last_viewed_at = :last_viewed_at
                """
            ),
            {"dong_code": dong_code, "view_count": view_count, "last_viewed_at": last_viewed},
        )
        rows_inserted += 1
    return rows_inserted


def seed_user_view_history(conn, users: list[dict], dong_codes: list[str]) -> int:
    rows_inserted = 0
    for u in users:
        # 유저 한 명당 3~7개 동을 "내가 본 지역"으로 기록.
        n_dongs = random.randint(3, min(7, len(dong_codes)))
        picked = random.sample(dong_codes, n_dongs)
        for dong_code in picked:
            view_count = random.randint(1, 12)
            last_viewed = _random_timestamp_within_days(21)
            conn.execute(
                text(
                    """
                    INSERT INTO user_view_history (user_id, dong_code, view_count, last_viewed_at)
                    VALUES (:user_id, :dong_code, :view_count, :last_viewed_at)
                    ON DUPLICATE KEY UPDATE
                        view_count = :view_count,
                        last_viewed_at = :last_viewed_at
                    """
                ),
                {
                    "user_id": u["user_id"],
                    "dong_code": dong_code,
                    "view_count": view_count,
                    "last_viewed_at": last_viewed,
                },
            )
            rows_inserted += 1
    return rows_inserted


def reset_tables(conn) -> None:
    conn.execute(text("DELETE FROM user_view_history"))
    conn.execute(text("DELETE FROM dong_view_stats"))


def main():
    parser = argparse.ArgumentParser(description="dong_view_stats/user_view_history 더미 시드")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 다시 시드")
    args = parser.parse_args()

    engine = get_engine()
    if engine is None:
        print("DB 연결 실패 — .env/DB 설정을 확인하세요.")
        return

    with engine.begin() as conn:
        if args.reset:
            reset_tables(conn)
            print("기존 dong_view_stats / user_view_history 삭제 완료.")

        dong_codes = _all_dong_codes(conn)
        store_counts = _store_counts_by_dong(conn)
        users = _owner_founder_users(conn)

        n_dong = seed_dong_view_stats(conn, dong_codes, store_counts)
        n_user = seed_user_view_history(conn, users, dong_codes)

    print(f"dong_view_stats: {n_dong}개 동 시드 완료")
    print(f"user_view_history: owner/founder {len(users)}명 대상, {n_user}건 시드 완료")


if __name__ == "__main__":
    main()