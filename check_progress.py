"""
프로젝트 루트에서: python check_progress.py
지금 적재 중인 테이블에 몇 행 들어갔는지 확인용 (원래 돌아가는 load_to_tidb.py는
건드리지 않음 — 조회만 하는 별도 커넥션).
"""
from app.shared.db import get_engine

TABLE = 'spatial_density_features'  # 확인하고 싶은 테이블 이름으로 바꿔서 쓰면 됨

engine = get_engine()
if engine is None:
    print("연결 정보 없음 (.env 확인)")
else:
    with engine.connect() as conn:
        n = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {TABLE}").scalar()
        print(f"{TABLE} 현재 행 수: {n:,}")