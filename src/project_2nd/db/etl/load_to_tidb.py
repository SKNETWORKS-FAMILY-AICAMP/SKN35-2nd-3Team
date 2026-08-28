"""TiDB 안전 적재/재개 도구.

기본 실행은 기존 테이블과 데이터를 절대 삭제하지 않는다. 13개 테이블이 모두
존재하는지 확인한 뒤, data/features/*.csv와 DB 행 수를 비교해 미완료 구간만
이어 적재한다. 전체 초기화는 사용자가 명시적으로 ``--reset``을 지정했을 때만
허용한다.

실행 전:
  1. .env.example을 .env로 복사하고 TiDB Cloud 연결 정보 채우기
  2. pip install pymysql sqlalchemy python-dotenv cryptography
  3. data/features/ 에 CSV들이 이미 만들어져 있어야 함 (run_pipeline.sh 먼저 실행)

실행:
  python src/project_2nd/db/etl/load_to_tidb.py          # 안전 재개(기본)
  python src/project_2nd/db/etl/load_to_tidb.py --reset  # 전체 삭제 후 재적재(명시적)
"""
import argparse
import hashlib
import re
import sys
import time
import pathlib
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd

# 이 파일 실제 경로: <project_root>/src/project_2nd/db/etl/load_to_tidb.py
_ETL_DIR = pathlib.Path(__file__).resolve().parent   # .../src/project_2nd/db/etl
_DB_DIR = _ETL_DIR.parent                              # .../src/project_2nd/db
_PROJECT_ROOT = _DB_DIR.parent.parent.parent            # .../  (project_2nd -> src -> root)

# app/shared/db.py는 <project_root>/app/shared/db.py에 있음 (db/, src/ 와는 다른 자리).
sys.path.insert(0, str(_PROJECT_ROOT))
from app.shared.db import get_engine

# schema.sql은 항상 이 파일과 같은 db/ 폴더 밑에 있음 (etl/의 형제) — 위치 추측 필요 없음.
SCHEMA_PATH = str(_DB_DIR / 'schema.sql')

# data/는 프로젝트 루트 바로 밑에 있음(확인됨. src/project_2nd/ 밑이 아님).
FEATURES_DIR = str(_PROJECT_ROOT / 'data' / 'features')


# 스키마에 반드시 있어야 하는 13개 테이블.
_ALL_TABLES = [
    'administrative_dongs', 'industries', 'stores', 'store_snapshots',
    'population_features', 'spatial_density_features', 'trend_keywords',
    'industry_transitions', 'industry_survival_stats',
    'users', 'models', 'predictions', 'support_actions',
]

_OPERATING_TABLES = {'users', 'models', 'predictions', 'support_actions'}
_RESUMABLE_LARGE_TABLES = {'store_snapshots', 'spatial_density_features'}

# DB에서 AUTO_INCREMENT 컬럼을 제외하고 비교할 실제 적재 컬럼. 이 순서는
# 내용 시그니처의 일부이므로 변경하지 않는다.
_TABLE_COLUMNS = {
    'administrative_dongs': ['dong_code', 'dong_name', 'gu_name'],
    'industries': [
        'industry_code', 'industry_name', 'industry_jung_code',
        'industry_jung_name', 'industry_dae_code', 'custom_group',
    ],
    'stores': [
        'store_id', 'current_industry_code', 'dong_code',
        'first_seen_snapshot', 'last_seen_snapshot', 'n_snapshots_observed',
        'is_closed', 'had_temporary_gap',
    ],
    'store_snapshots': [
        'store_id', 'snapshot_date', 'industry_code', 'dong_code', 'store_name',
        'floor_category', 'lng', 'lat', 'is_closed_next',
        'transitioned_next', 'label_available',
    ],
    'population_features': [
        'dong_code', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
        'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate',
    ],
    'spatial_density_features': [
        'store_id', 'snapshot_date', 'same_industry_count_300m',
        'total_count_300m', 'nearest_same_industry_distance_m',
        'dong_industry_count', 'coord_cluster_size',
    ],
    'trend_keywords': ['keyword', 'snapshot_date', 'store_count', 'growth_rate'],
    'industry_transitions': [
        'store_id', 'from_snapshot', 'to_snapshot',
        'from_industry_code', 'to_industry_code',
    ],
    'industry_survival_stats': [
        'from_industry_code', 'to_industry_code', 'sample_size', 'survival_rate',
    ],
}

_BOOLEAN_COLUMNS = {
    'is_closed', 'had_temporary_gap', 'is_closed_next', 'transitioned_next',
    'label_available', 'tourist_zone_candidate',
}
_INTEGER_COLUMNS = {
    'n_snapshots_observed', 'same_industry_count_300m', 'total_count_300m',
    'dong_industry_count', 'coord_cluster_size', 'store_count', 'sample_size',
}
_DECIMAL_SCALES = {
    'lng': 7,
    'lat': 7,
    'korean_pop': 4,
    'foreign_long_pop': 4,
    'foreign_short_pop': 4,
    'total_pop_avg': 4,
    'foreign_short_ratio': 5,
    'nearest_same_industry_distance_m': 3,
    'growth_rate': 4,
    'survival_rate': 5,
}
_SIGNATURE_MODULUS = 1 << 256


def _strip_sql_line_comments(sql):
    """각 줄에서 '--' 뒤쪽을 제거. (기존 로직은 세미콜론으로 나눈 '조각 전체'가
    '--'로 시작하면 그 조각을 통째로 버렸는데, administrative_dongs/population_features/
    users/models/support_actions처럼 CREATE TABLE 바로 앞에 섹션 구분용 주석 블록이
    붙어있는 경우 CREATE TABLE까지 같이 버려져서 13개 중 8개만 만들어지는 버그가 있었음.
    이 스키마엔 '--'를 포함하는 문자열 리터럴이 없어서 줄 단위 제거가 안전함.)"""
    return '\n'.join(line[:line.find('--')] if '--' in line else line for line in sql.splitlines())


def _existing_tables(engine):
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = DATABASE()"
        ).fetchall()
    return {row[0] for row in rows}


def _verify_required_tables(engine):
    """기본 실행에서 DDL을 수행하지 않고 13개 테이블 존재만 검증한다."""
    missing = set(_ALL_TABLES) - _existing_tables(engine)
    if missing:
        raise RuntimeError(
            "필수 TiDB 테이블이 없어 안전 재개를 중단합니다: "
            f"{sorted(missing)}. 빈 DB를 처음 구성하려는 경우에만 --reset을 "
            "명시적으로 사용하세요."
        )
    print("스키마 확인 완료: 필수 13개 테이블이 모두 존재합니다 (변경 없음).")


def _table_row_count(engine, table_name):
    if table_name not in _ALL_TABLES:
        raise ValueError(f"허용되지 않은 테이블 이름: {table_name}")
    with engine.connect() as conn:
        return int(conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table_name}").scalar())


def _canonical_value(column, value):
    """CSV와 DB 타입 차이를 없애 내용 시그니처를 같은 형태로 만든다."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return r'\N'
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if column in _BOOLEAN_COLUMNS:
        normalized = str(value).strip().lower()
        if normalized in {'1', '1.0', 'true', 't', 'yes'}:
            return '1'
        if normalized in {'0', '0.0', 'false', 'f', 'no'}:
            return '0'
        raise ValueError(f"{column}의 boolean 값을 해석할 수 없습니다: {value!r}")
    if column in _INTEGER_COLUMNS:
        try:
            return str(int(Decimal(str(value))))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{column}의 정수 값을 해석할 수 없습니다: {value!r}") from exc
    if column in _DECIMAL_SCALES:
        scale = _DECIMAL_SCALES[column]
        try:
            quantized = Decimal(str(value)).quantize(
                Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP
            )
        except InvalidOperation as exc:
            raise ValueError(f"{column}의 소수 값을 해석할 수 없습니다: {value!r}") from exc
        return f"{quantized:.{scale}f}"
    return str(value)


def _new_signature():
    return [0, 0, 0]  # 행 수, SHA256 정수 합, SHA256 정수 XOR


def _update_signature(signature, rows, columns):
    for row in rows:
        digest = hashlib.sha256()
        for column, value in zip(columns, row):
            encoded = _canonical_value(column, value).encode('utf-8')
            digest.update(len(encoded).to_bytes(4, 'big'))
            digest.update(encoded)
        number = int.from_bytes(digest.digest(), 'big')
        signature[0] += 1
        signature[1] = (signature[1] + number) % _SIGNATURE_MODULUS
        signature[2] ^= number
    return signature


def _frame_signature(df, columns):
    return tuple(_update_signature(
        _new_signature(), df.loc[:, columns].itertuples(index=False, name=None), columns
    ))


def _chunks_signature(chunks, columns, limit=None):
    signature = _new_signature()
    remaining = limit
    for chunk in chunks:
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        _update_signature(
            signature,
            chunk.loc[:, columns].itertuples(index=False, name=None),
            columns,
        )
    if limit is not None and signature[0] != limit:
        raise RuntimeError(
            f"CSV prefix 요청 {limit:,}행보다 실제 읽은 행이 적습니다: {signature[0]:,}행"
        )
    return tuple(signature)


def _db_signature(engine, table_name, columns):
    if table_name not in _TABLE_COLUMNS or list(columns) != _TABLE_COLUMNS[table_name]:
        raise ValueError(f"{table_name} 시그니처 컬럼 정의가 올바르지 않습니다.")
    signature = _new_signature()
    select_columns = ', '.join(f"`{column}`" for column in columns)
    with engine.connect() as raw_conn:
        conn = raw_conn.execution_options(stream_results=True)
        result = conn.exec_driver_sql(f"SELECT {select_columns} FROM `{table_name}`")
        while True:
            rows = result.fetchmany(10_000)
            if not rows:
                break
            _update_signature(signature, rows, columns)
    return tuple(signature)


def _assert_same_signature(table_name, source_signature, db_signature, scope='전체'):
    if source_signature != db_signature:
        raise RuntimeError(
            f"{table_name} {scope} 내용이 현재 CSV와 다릅니다. 자동 적재/재개를 "
            "중단합니다. 기존 데이터를 삭제하거나 덮어쓰지 않았습니다."
        )


def _load_or_skip_dataframe(engine, table_name, df, *, dry_run=False):
    """작은 테이블: 0행이면 적재, 완전 일치면 skip, 부분/불일치면 중단."""
    columns = _TABLE_COLUMNS[table_name]
    df = df.loc[:, columns]
    expected = len(df)
    actual = _table_row_count(engine, table_name)
    print(f"{table_name}: CSV {expected:,}행 / DB {actual:,}행")

    if actual == expected:
        source_signature = _frame_signature(df, columns)
        db_signature = _db_signature(engine, table_name, columns)
        _assert_same_signature(table_name, source_signature, db_signature)
        print(f"  완료 내용 일치 — 건너뜀")
        return
    if actual != 0:
        extra = (
            "industry_transitions에는 자연키 UNIQUE가 없어 부분 재개가 불가능합니다."
            if table_name == 'industry_transitions'
            else "부분 상태는 자동 보정하지 않습니다."
        )
        raise RuntimeError(
            f"{table_name}: DB {actual:,}행, CSV {expected:,}행. {extra} "
            "기존 데이터를 변경하지 않고 중단합니다."
        )
    if dry_run:
        print(f"  [DRY-RUN] {expected:,}행을 새로 적재할 예정")
        return

    _to_sql_with_retry(df, table_name, engine, if_exists='append', index=False)
    _assert_same_signature(
        table_name, _frame_signature(df, columns),
        _db_signature(engine, table_name, columns),
    )
    print(f"  {expected:,}행 적재 및 내용 검증 완료")


def create_tables(engine, *, reset=False):
    """``reset=True``일 때만 모든 테이블을 삭제하고 다시 만든다.

    기본값은 삭제 금지다. 기존 호출부가 인자를 빠뜨려도 안전하도록, reset=False면
    DDL 대신 필수 테이블 존재 검증만 수행한다.
    """
    if not reset:
        _verify_required_tables(engine)
        return

    print("⚠️ --reset 지정됨: 기존 13개 테이블 삭제 후 스키마를 다시 생성합니다.")
    with open(SCHEMA_PATH, encoding='utf-8') as f:
        sql = f.read()
    sql = _strip_sql_line_comments(sql)
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    with engine.begin() as conn:
        # TiDB Cloud Serverless는 분산 환경이라, 방금 만든 참조 테이블이 클러스터
        # 전체에 전파되기 전에 바로 다음 FK 있는 CREATE TABLE이 실행되면
        # "Failed to open the referenced table" 에러가 남(TiDB 쪽 타이밍 이슈).
        # 스키마 생성 구간만 FK 체크를 꺼서 우회 — FK 관계 자체는 그대로 생성됨.
        # exec_driver_sql을 씀 — text()는 SQL 안의 ":아무개" 패턴을 전부 바인드
        # 파라미터로 해석해버려서, schema.sql 주석에 있는 예시 JSON
        # ("feature_value":6 처럼 콜론 뒤에 오는 부분)까지 파라미터로 착각해
        # "A value is required for bind parameter '6'" 에러가 났었음.
        # exec_driver_sql은 그런 파싱 없이 SQL 문자열을 그대로 드라이버에 넘김.
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=0")

        # DROP은 위의 reset=True 분기를 통과한 경우에만 도달한다.
        for table in reversed(_ALL_TABLES):
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table}")

        for stmt in statements:
            conn.exec_driver_sql(stmt)
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=1")
    print(f"  {len(statements)}개 테이블 생성 완료")


def _wait_for_schema_visible(engine, tables, timeout=20, interval=1):
    """TiDB Cloud Serverless는 스키마 변경(DDL)이 클러스터 전체 노드에 퍼지는 데
    약간의 지연이 있음. create_tables() 끝나자마자 pandas.to_sql()이 새 커넥션으로
    "이 테이블 있나?" 확인했다가, 그 커넥션이 아직 스키마 전파 안 된 노드에 붙으면
    "없다"고 오판해서 자기 나름대로(TEXT 컬럼 등) 새로 만들려다 기존 FK 제약과
    충돌내는 걸 실제로 겪었음(administrative_dongs에서 발생).
    모든 테이블이 information_schema에 실제로 보일 때까지 여기서 기다림."""
    deadline = time.time() + timeout
    remaining = set(tables)
    while remaining and time.time() < deadline:
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
            ).fetchall()
        existing = {r[0] for r in rows}
        remaining = set(tables) - existing
        if remaining:
            time.sleep(interval)
    if remaining:
        print(f"  ⚠️ 스키마 전파 대기 시간({timeout}초) 초과, 아직 안 보이는 테이블: {sorted(remaining)}")
        print("     (그래도 계속 진행 — 여기서 또 에러 나면 timeout을 늘려서 재시도)")
    else:
        print("  스키마 전파 확인 완료 (모든 테이블 조회 가능)")


def _collect_dong_codes(csv_path, chunksize=200_000):
    """CSV 하나에서 dong_code 컬럼만 뽑아 유니크 집합으로 반환. dong_code 종류는
    많아야 수백 개라 파일이 아무리 커도 결과 집합은 작음 — chunksize로 메모리만 아낌."""
    codes = set()
    for chunk in pd.read_csv(csv_path, usecols=['dong_code'], dtype={'dong_code': str}, chunksize=chunksize):
        codes.update(chunk['dong_code'].dropna().unique())
    return codes


def _to_sql_with_retry(df, table_name, engine, max_retries=3, retry_wait=3, **kwargs):
    """store_snapshots 첫 청크(10만행)에서 실제로 겪은 문제: INSERT 자체는 끝났는데
    commit() 시점에 `sqlalchemy.exc.PendingRollbackError: Can't reconnect until invalid
    transaction is rolled back`로 실패함. 이 에러 메시지 자체는 진짜 원인이 아니라 —
    커밋 도중 커넥션이 끊기면(TiDB Cloud Serverless 쪽 유휴/실행 타임아웃, 네트워크 순단 등)
    SQLAlchemy가 "커넥션이 invalid 상태인데 그걸 롤백하기 전엔 재연결 못 한다"는 걸
    알려주는 것뿐이라, 진짜 원인(끊긴 이유)은 감춰짐.
    풀에 남아있는 커넥션이 이미 깨진 상태일 수 있어서, 실패하면 engine.dispose()로
    풀을 통째로 비우고(다음 사용 시 완전히 새 커넥션으로 다시 연결됨) 재시도함 —
    네트워크성 순단이면 이걸로 대부분 복구됨. method='multi'+chunksize=2000은
    한 행씩 insert하던 걸 묶어서 보내 왕복 횟수를 줄이는 것 — 속도 개선 겸,
    커밋까지 걸리는 시간을 줄여서 이런 타임아웃성 문제 자체가 덜 나게 하는 효과도 있음."""
    kwargs.setdefault('method', 'multi')
    kwargs.setdefault('chunksize', 2000)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            df.to_sql(table_name, engine, **kwargs)
            return
        except Exception as e:
            last_exc = e
            orig = getattr(e, 'orig', None)
            print(f"    ⚠️ {table_name} 적재 중 오류(시도 {attempt}/{max_retries}): "
                  f"{type(e).__name__}: {e}" + (f"  [원인: {orig}]" if orig else ""))
            engine.dispose()  # 풀에 남은 (깨졌을 수도 있는) 커넥션 전부 버림
            if attempt < max_retries:
                time.sleep(retry_wait)
    raise last_exc


_GU_CODE_MAP = {
    '11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구',
    '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구',
    '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구',
    '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구',
    '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구',
    '11740': '강동구',
}


def load_administrative_dongs(engine):
    # administrative_dongs.csv는 features/spatial/build_population_features.py에서
    # 생활인구 dong_code ∪ 상권(6개 스냅샷) dong_code를 합쳐서 만든 완전한 마스터임
    # (예전엔 population_features.csv만 보고 만들어서, 상권 데이터에만 있는 dong_code
    # 12개가 실제 이름이 있는데도 여기서 '(미상)' placeholder로 채워지고 있었음 —
    # 그 문제를 소스 단계에서 고쳐서 이제 이 파일 하나가 이미 정답임).
    dongs = pd.read_csv(f'{FEATURES_DIR}/administrative_dongs.csv', dtype=str)

    # 그래도 안전망: stores.csv/store_snapshots.csv가 이 마스터에마저 없는 dong_code를
    # 참조하면(원래는 없어야 함 — administrative_dongs.csv가 상권 6개 스냅샷 전체를
    # 이미 합친 결과라) 최소한 gu_name은 코드로 채워서 등록해두고 크게 경고한다.
    extra_codes = _collect_dong_codes(f'{FEATURES_DIR}/stores.csv')
    extra_codes |= _collect_dong_codes(f'{FEATURES_DIR}/store_snapshots.csv')
    missing_codes = extra_codes - set(dongs['dong_code'])
    if missing_codes:
        print(f"  ⚠️ administrative_dongs.csv에도 없는데 stores/store_snapshots가 참조하는 "
              f"dong_code {len(missing_codes)}건 발견(예상 밖 — build_population_features.py를 "
              f"다시 돌렸는지, data/raw 원본이 최신인지 확인 권장): {sorted(missing_codes)}")
        extra_rows = pd.DataFrame({'dong_code': sorted(missing_codes)})
        extra_rows['dong_name'] = '(미상)'
        extra_rows['gu_name'] = extra_rows['dong_code'].str[:5].map(_GU_CODE_MAP).fillna('(미상)')
        dongs = pd.concat([dongs, extra_rows], ignore_index=True)

    _to_sql_with_retry(dongs, 'administrative_dongs', engine, if_exists='append', index=False)
    print(f"administrative_dongs: {len(dongs):,}행 적재")


def load_industries(engine):
    df = pd.read_csv(f'{FEATURES_DIR}/industries.csv', dtype=str)
    _to_sql_with_retry(df, 'industries', engine, if_exists='append', index=False)
    print(f"industries: {len(df):,}행 적재")


def load_stores(engine):
    # dong_code/current_industry_code를 문자열로 명시 안 하면 pandas가 숫자로만
    # 보이는 컬럼을 int로 읽어버려서(예: '11140550' -> 11140550) administrative_dongs/
    # industries 쪽 문자열 dong_code/industry_code와 타입이 어긋날 수 있음.
    df = pd.read_csv(f'{FEATURES_DIR}/stores.csv',
                      dtype={'store_id': str, 'dong_code': str, 'current_industry_code': str})
    _to_sql_with_retry(df, 'stores', engine, if_exists='append', index=False)
    print(f"stores: {len(df):,}행 적재")


def _fetch_id_set(engine, table, col):
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"SELECT {col} FROM {table}").fetchall()
    return {r[0] for r in rows}


_STORE_SNAPSHOT_NOT_NULL_COLS = [
    'store_id', 'snapshot_date', 'industry_code', 'dong_code', 'store_name',
    'floor_category', 'lng', 'lat', 'is_closed_next', 'transitioned_next', 'label_available',
]


def _validate_store_snapshots_chunk(chunk, valid_dong_codes, valid_industry_codes, valid_store_ids):
    """같은 청크를 재시도해도 매번 똑같이 PendingRollbackError만 뜨고 끝나는 건
    (에러 메시지에 [원인: ...]이 안 붙는 것도 확인됨 — DBAPI 레벨 원인이 없다는 뜻)
    네트워크 순단이 아니라 이 청크 안에 진짜 제약 위반(NOT NULL/FK/UNIQUE) 데이터가
    있어서인데, pandas.to_sql(method='multi')가 그 원인을 그대로 넘겨주지 않고
    커밋 시점의 "invalid transaction"이라는 뭉뚱그려진 메시지로만 보여준 것으로 보임.
    그래서 insert 시도 전에 파이썬에서 직접 스키마 제약을 검사해 진짜 원인을 밝혀낸다."""
    problems = []

    for col in _STORE_SNAPSHOT_NOT_NULL_COLS:
        n_null = chunk[col].isna().sum()
        if n_null:
            sample = chunk.loc[chunk[col].isna(), 'store_id'].head(3).tolist()
            problems.append(f"{col} 컬럼 NULL {n_null}행 (store_id 예: {sample})")

    bad_industry = chunk.loc[~chunk['industry_code'].isin(valid_industry_codes), 'industry_code']
    if len(bad_industry):
        problems.append(f"industries 테이블에 없는 industry_code {bad_industry.nunique()}종류, "
                         f"{len(bad_industry)}행 (예: {sorted(bad_industry.dropna().unique())[:10]})")

    bad_dong = chunk.loc[~chunk['dong_code'].isin(valid_dong_codes), 'dong_code']
    if len(bad_dong):
        problems.append(f"administrative_dongs 테이블에 없는 dong_code {bad_dong.nunique()}종류, "
                         f"{len(bad_dong)}행 (예: {sorted(bad_dong.dropna().unique())[:10]})")

    bad_store = chunk.loc[~chunk['store_id'].isin(valid_store_ids), 'store_id']
    if len(bad_store):
        problems.append(f"stores 테이블에 없는 store_id {bad_store.nunique()}종류, "
                         f"{len(bad_store)}행 (예: {sorted(bad_store.dropna().unique())[:5]})")

    dup_mask = chunk.duplicated(subset=['store_id', 'snapshot_date'], keep=False)
    if dup_mask.any():
        sample = chunk.loc[dup_mask, ['store_id', 'snapshot_date']].head(5).to_dict('records')
        problems.append(f"(store_id, snapshot_date) 중복 {dup_mask.sum()}행 (예: {sample})")

    return problems


def load_store_snapshots(engine):
    # snapshot_id는 AUTO_INCREMENT라 CSV에서 빼고 적재
    cols = ['store_id', 'snapshot_date', 'industry_code', 'dong_code', 'store_name',
            'floor_category', 'lng', 'lat', 'is_closed_next', 'transitioned_next', 'label_available']

    # store_snapshots는 dong_code/industry_code/store_id 세 개나 FK로 물고 있어서,
    # administrative_dongs 때(Bug 8)처럼 원본 CSV들 사이에 코드 불일치가 또 있을 수 있음.
    # insert 시도 전에 이미 적재된 값들과 미리 대조해서 실제 원인을 알아냄.
    valid_dong_codes = _fetch_id_set(engine, 'administrative_dongs', 'dong_code')
    valid_industry_codes = _fetch_id_set(engine, 'industries', 'industry_code')
    valid_store_ids = _fetch_id_set(engine, 'stores', 'store_id')

    for i, chunk in enumerate(pd.read_csv(f'{FEATURES_DIR}/store_snapshots.csv', usecols=cols,
                                           dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                                                  'snapshot_date': str},
                                           chunksize=100_000)):
        problems = _validate_store_snapshots_chunk(chunk, valid_dong_codes, valid_industry_codes, valid_store_ids)
        if problems:
            start_row = i * 100_000
            print(f"  ⚠️ store_snapshots.csv {start_row:,}~{start_row + len(chunk):,}행 구간에서 "
                  f"제약 위반 발견 (PendingRollbackError로 감춰졌던 진짜 원인):")
            for p in problems:
                print(f"     - {p}")
            raise ValueError(
                "store_snapshots 데이터 정합성 문제로 적재 중단 — 위 목록 참고해서 "
                "CSV를 보완하거나(권장) 해당 행을 제외하는 방식으로 대응 필요."
            )
        _to_sql_with_retry(chunk, 'store_snapshots', engine, if_exists='append', index=False)
    print("store_snapshots 적재 완료")


def load_population_features(engine):
    cols = ['dong_code', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
            'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
    df = pd.read_csv(f'{FEATURES_DIR}/population_features.csv', usecols=cols, dtype={'dong_code': str})
    _to_sql_with_retry(df, 'population_features', engine, if_exists='append', index=False)
    print(f"population_features: {len(df):,}행 적재")


def load_spatial_density_features(engine):
    for chunk in pd.read_csv(f'{FEATURES_DIR}/spatial_density_features.csv',
                              dtype={'store_id': str, 'snapshot_date': str},
                              chunksize=100_000):
        _to_sql_with_retry(chunk, 'spatial_density_features', engine, if_exists='append', index=False)
    print("spatial_density_features 적재 완료")


def load_trend_keywords(engine):
    # 원본은 keyword x snapshot_date 와이드 포맷 -> 롱 포맷으로 pivot
    df = pd.read_csv(f'{FEATURES_DIR}/trend_keywords.csv')
    snapshot_cols = [c for c in df.columns if c not in ('keyword', 'growth_rate')]
    long_rows = []
    for _, row in df.iterrows():
        for snap in snapshot_cols:
            long_rows.append({
                'keyword': row['keyword'], 'snapshot_date': snap,
                'store_count': row[snap],
                'growth_rate': row['growth_rate'] if snap == snapshot_cols[-1] else None,
            })
    long_df = pd.DataFrame(long_rows)
    _to_sql_with_retry(long_df, 'trend_keywords', engine, if_exists='append', index=False)
    print(f"trend_keywords: {len(long_df):,}행 적재 (롱 포맷 변환)")


def load_industry_transitions(engine):
    # store_id/from_industry_code/to_industry_code 셋 다 raw 상권 데이터에서 그대로
    # 뽑혀나온 코드라(features/industry_grouping/build_industries.py는 6개 스냅샷이
    # "전부 동일한 소분류 체계"라는 팀 검증 결과를 전제로 최신 스냅샷 하나에서만
    # industries.csv를 만듦), administrative_dongs 때(Bug 8)와 같은 유형의 코드
    # 불일치가 여기서도 생길 수 있음. 통째로 실패하기 전에 미리 걸러서, 안 맞는
    # 행만 로그로 남기고 제외 — 이 테이블을 FK로 참조하는 다른 테이블이 없어서
    # (참조 무결성 연쇄가 없음) 일부 행을 빼고 넘어가도 안전함.
    df = pd.read_csv(f'{FEATURES_DIR}/industry_transitions.csv', dtype=str)
    df = df.drop(columns=['transition_id'])  # AUTO_INCREMENT

    valid_store_ids = _fetch_id_set(engine, 'stores', 'store_id')
    valid_industry_codes = _fetch_id_set(engine, 'industries', 'industry_code')
    bad_mask = (~df['store_id'].isin(valid_store_ids)
                | ~df['from_industry_code'].isin(valid_industry_codes)
                | ~df['to_industry_code'].isin(valid_industry_codes))
    if bad_mask.any():
        print(f"  ⚠️ industry_transitions.csv {bad_mask.sum()}행이 존재하지 않는 "
              f"store_id/industry_code를 참조해서 제외함(원본 데이터 확인 권장): "
              f"{df.loc[bad_mask].head(10).to_dict('records')}")
        df = df.loc[~bad_mask].reset_index(drop=True)

    _to_sql_with_retry(df, 'industry_transitions', engine, if_exists='append', index=False)
    print(f"industry_transitions: {len(df):,}행 적재")


def load_industry_survival_stats(engine):
    # 위와 같은 이유로 from/to_industry_code가 industries에 실제로 있는지 미리 확인.
    df = pd.read_csv(f'{FEATURES_DIR}/industry_survival_stats.csv', dtype=str)

    valid_industry_codes = _fetch_id_set(engine, 'industries', 'industry_code')
    bad_mask = ~df['from_industry_code'].isin(valid_industry_codes) | ~df['to_industry_code'].isin(valid_industry_codes)
    if bad_mask.any():
        print(f"  ⚠️ industry_survival_stats.csv {bad_mask.sum()}행이 존재하지 않는 "
              f"industry_code를 참조해서 제외함(원본 데이터 확인 권장): "
              f"{df.loc[bad_mask, ['from_industry_code', 'to_industry_code']].to_dict('records')}")
        df = df.loc[~bad_mask].reset_index(drop=True)

    _to_sql_with_retry(df, 'industry_survival_stats', engine, if_exists='append', index=False)
    print(f"industry_survival_stats: {len(df):,}행 적재")


if __name__ == '__main__':
    # 안전장치: 이 레거시 전체 적재기는 테이블 DROP 및 append를 수행하므로,
    # 현재 일부/전체 적재된 TiDB에 다시 실행하면 운영 데이터 삭제 또는 중복 적재가
    # 발생할 수 있다. 따라서 어떤 인수로 실행해도 DB에 연결하거나 쓰지 않는다.
    # 안전한 현황 점검/재개는 전용 스크립트만 사용한다.
    print(
        "안전 중단: load_to_tidb.py의 전체 초기화/append 실행은 비활성화되어 있습니다.\n"
        "읽기 전용 점검: python src/project_2nd/db/etl/resume_tidb_current.py\n"
        "검증 후 안전 재개: python src/project_2nd/db/etl/resume_tidb_current.py --execute\n"
        "이 스크립트에서는 --execute, --reset 등 어떤 인수도 DB 쓰기를 허용하지 않습니다."
    )
    raise SystemExit(2)
