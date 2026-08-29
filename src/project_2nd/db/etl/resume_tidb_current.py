"""Safely audit/resume the *current* TiDB load without recreating tables.

Default mode is read-only. Pass ``--execute`` explicitly to write only the
three idempotent feature tables and, under strict conditions, transitions.
The four application/serving tables are never written by this program.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sqlalchemy import text


ETL_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = ETL_DIR.parents[3]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
sys.path.insert(0, str(PROJECT_ROOT))
from app.shared.db import get_engine  # noqa: E402

READ_CHUNK = 100_000
SQL_BATCH = 2_000
PROTECTED_TABLES = frozenset({"users", "models", "predictions", "support_actions"})


@dataclass(frozen=True)
class CsvSpec:
    table: str
    filename: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    nullable: tuple[str, ...] = ()

    @property
    def path(self) -> pathlib.Path:
        return FEATURES_DIR / self.filename


MASTER_SPECS = (
    CsvSpec("administrative_dongs", "administrative_dongs.csv",
            ("dong_code", "dong_name", "gu_name"), ("dong_code",)),
    CsvSpec("industries", "industries.csv",
            ("industry_code", "industry_name", "industry_jung_code", "industry_jung_name",
             "industry_dae_code", "custom_group"), ("industry_code",)),
    CsvSpec("stores", "stores.csv",
            ("store_id", "current_industry_code", "dong_code", "first_seen_snapshot",
             "last_seen_snapshot", "n_snapshots_observed", "is_closed", "had_temporary_gap"),
            ("store_id",)),
    CsvSpec("store_snapshots", "store_snapshots.csv",
            ("store_id", "snapshot_date", "industry_code", "dong_code", "store_name",
             "floor_category", "lng", "lat", "is_closed_next", "transitioned_next",
             "label_available"), ("store_id", "snapshot_date")),
    CsvSpec("population_features", "population_features.csv",
            ("dong_code", "korean_pop", "foreign_long_pop", "foreign_short_pop",
             "total_pop_avg", "foreign_short_ratio", "tourist_zone_candidate"),
            ("dong_code",)),
)

SPATIAL = CsvSpec("spatial_density_features", "spatial_density_features.csv",
                  ("store_id", "snapshot_date", "same_industry_count_300m", "total_count_300m",
                   "nearest_same_industry_distance_m", "dong_industry_count", "coord_cluster_size"),
                  ("store_id", "snapshot_date"), ("nearest_same_industry_distance_m",))
SURVIVAL = CsvSpec("industry_survival_stats", "industry_survival_stats.csv",
                   ("from_industry_code", "to_industry_code", "sample_size", "survival_rate"),
                   ("from_industry_code", "to_industry_code"))
TRANSITIONS = CsvSpec("industry_transitions", "industry_transitions.csv",
                      ("store_id", "from_snapshot", "to_snapshot", "from_industry_code",
                       "to_industry_code"),
                      ("store_id", "from_snapshot", "to_snapshot", "from_industry_code",
                       "to_industry_code"))


def die(message: str) -> None:
    raise RuntimeError(message)


def header(spec: CsvSpec) -> list[str]:
    if not spec.path.is_file():
        die(f"필수 소스 없음: {spec.path}")
    cols = list(pd.read_csv(spec.path, nrows=0).columns)
    missing = [c for c in spec.columns if c not in cols]
    if missing:
        die(f"{spec.filename} 필수 컬럼 누락: {missing}")
    return cols


def key_strings(frame: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    # Length-prefix encoding prevents delimiter collisions.
    result: list[str] = []
    for row in frame.loc[:, list(cols)].itertuples(index=False, name=None):
        result.append("".join(f"{len(str(v))}:{v}" for v in row))
    return result


def scan_source(spec: CsvSpec, seen: sqlite3.Connection) -> int:
    header(spec)
    seen.execute("DROP TABLE IF EXISTS seen_keys")
    seen.execute("CREATE TABLE seen_keys (k TEXT PRIMARY KEY) WITHOUT ROWID")
    total = 0
    nonnull = [c for c in spec.columns if c not in spec.nullable]
    dtypes = {c: str for c in spec.key}
    for chunk in pd.read_csv(spec.path, usecols=list(spec.columns), dtype=dtypes,
                             chunksize=READ_CHUNK):
        bad = chunk[nonnull].isna().any(axis=1)
        if bad.any():
            die(f"{spec.filename}: NOT NULL 컬럼 결측 {int(bad.sum()):,}행")
        keys = key_strings(chunk, spec.key)
        before = seen.total_changes
        seen.executemany("INSERT OR IGNORE INTO seen_keys(k) VALUES (?)", ((k,) for k in keys))
        inserted = seen.total_changes - before
        if inserted != len(keys):
            die(f"{spec.filename}: 자연키 {spec.key} 중복 {len(keys)-inserted:,}건")
        total += len(chunk)
    if total == 0:
        die(f"{spec.filename}: 빈 소스")
    print(f"[SOURCE] {spec.table}: {total:,} rows, key unique")
    return total


def db_count(engine, table: str) -> int:
    if table in PROTECTED_TABLES:
        die(f"보호 테이블 접근 거부: {table}")
    with engine.connect() as conn:
        return int(conn.exec_driver_sql(f"SELECT COUNT(*) FROM `{table}`").scalar_one())


def require_tables(engine, tables: Iterable[str]) -> None:
    wanted = set(tables)
    with engine.connect() as conn:
        found = {r[0] for r in conn.exec_driver_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema=DATABASE()"
        )}
    missing = sorted(wanted - found)
    if missing:
        die(f"필수 DB 테이블 없음(DDL 실행 금지, 작업 중단): {missing}")


def audit_masters(engine, expected: dict[str, int]) -> None:
    for spec in MASTER_SPECS:
        actual = db_count(engine, spec.table)
        want = expected[spec.table]
        print(f"[AUDIT] {spec.table}: DB {actual:,} / source {want:,}")
        if actual != want:
            die(f"기준 테이블 불일치: {spec.table} DB={actual:,}, source={want:,}")


def normalize_records(frame: pd.DataFrame, cols: tuple[str, ...]) -> list[dict]:
    clean = frame.loc[:, list(cols)].astype(object).where(pd.notna(frame.loc[:, list(cols)]), None)
    return clean.to_dict("records")


def upsert_sql(spec: CsvSpec) -> str:
    cols = list(spec.columns)
    updates = [c for c in cols if c not in spec.key]
    if not updates:
        die(f"upsert 갱신 컬럼 없음: {spec.table}")
    quoted = ", ".join(f"`{c}`" for c in cols)
    values = ", ".join(f":{c}" for c in cols)
    update = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in updates)
    return f"INSERT INTO `{spec.table}` ({quoted}) VALUES ({values}) ON DUPLICATE KEY UPDATE {update}"


def execute_spatial(engine, expected: int) -> None:
    sql = text(upsert_sql(SPATIAL))
    processed = 0
    previous = db_count(engine, SPATIAL.table)
    if previous > expected:
        die(f"{SPATIAL.table}: DB count가 source보다 큼")
    for chunk in pd.read_csv(SPATIAL.path, usecols=list(SPATIAL.columns),
                             dtype={"store_id": str, "snapshot_date": str},
                             chunksize=READ_CHUNK):
        records = normalize_records(chunk, SPATIAL.columns)
        with engine.begin() as conn:
            for start in range(0, len(records), SQL_BATCH):
                conn.execute(sql, records[start:start + SQL_BATCH])
        processed += len(chunk)
        actual = db_count(engine, SPATIAL.table)
        if actual < previous or actual > expected:
            die(f"{SPATIAL.table}: 비정상 count 변화 {previous:,}->{actual:,}")
        previous = actual
        print(f"[WRITE] spatial source {processed:,}/{expected:,}; DB {actual:,} "
              f"({actual/expected*100:.2f}%)")
    if previous != expected:
        die(f"{SPATIAL.table}: 최종 count 불일치 DB={previous:,}, source={expected:,}")


def trend_layout() -> tuple[list[str], int]:
    path = FEATURES_DIR / "trend_keywords.csv"
    if not path.is_file():
        die(f"필수 소스 없음: {path}")
    cols = list(pd.read_csv(path, nrows=0).columns)
    for c in ("keyword", "growth_rate"):
        if c not in cols:
            die(f"trend_keywords.csv 필수 컬럼 누락: {c}")
    snapshots = [c for c in cols if c not in ("keyword", "growth_rate")]
    if not snapshots:
        die("trend_keywords.csv 스냅샷 컬럼 없음")
    return snapshots, sum(1 for _ in open(path, encoding="utf-8-sig")) - 1


def iter_trend(snapshots: list[str]):
    path = FEATURES_DIR / "trend_keywords.csv"
    for chunk in pd.read_csv(path, chunksize=READ_CHUNK):
        if chunk["keyword"].isna().any() or chunk[snapshots].isna().any(axis=None):
            die("trend_keywords.csv keyword/store_count 결측")
        long = chunk.melt(id_vars=["keyword", "growth_rate"], value_vars=snapshots,
                          var_name="snapshot_date", value_name="store_count")
        long["growth_rate"] = long["growth_rate"].where(
            long["snapshot_date"].eq(snapshots[-1]), None)
        yield long[["keyword", "snapshot_date", "store_count", "growth_rate"]]


def audit_trend_source(seen: sqlite3.Connection) -> tuple[list[str], int]:
    snapshots, wide_count = trend_layout()
    seen.execute("DROP TABLE IF EXISTS seen_keys")
    seen.execute("CREATE TABLE seen_keys (k TEXT PRIMARY KEY) WITHOUT ROWID")
    total = 0
    for chunk in iter_trend(snapshots):
        keys = key_strings(chunk, ("keyword", "snapshot_date"))
        before = seen.total_changes
        seen.executemany("INSERT OR IGNORE INTO seen_keys(k) VALUES (?)", ((k,) for k in keys))
        inserted = seen.total_changes - before
        if inserted != len(keys):
            die(f"trend_keywords.csv 자연키 중복 {len(keys)-inserted:,}건")
        total += len(chunk)
    expected = wide_count * len(snapshots)
    if total != expected or total == 0:
        die(f"trend_keywords 변환 행 수 오류: {total:,}!={expected:,}")
    print(f"[SOURCE] trend_keywords: {total:,} long rows, key unique")
    return snapshots, total


def execute_trend(engine, snapshots: list[str], expected: int) -> None:
    spec = CsvSpec("trend_keywords", "trend_keywords.csv",
                   ("keyword", "snapshot_date", "store_count", "growth_rate"),
                   ("keyword", "snapshot_date"), ("growth_rate",))
    sql = text(upsert_sql(spec))
    with engine.begin() as conn:
        for frame in iter_trend(snapshots):
            records = normalize_records(frame, spec.columns)
            for start in range(0, len(records), SQL_BATCH):
                conn.execute(sql, records[start:start + SQL_BATCH])
    actual = db_count(engine, spec.table)
    print(f"[WRITE] trend_keywords: DB {actual:,}/{expected:,}")
    if actual != expected:
        die("trend_keywords 최종 count 불일치")


def execute_simple_upsert(engine, spec: CsvSpec, expected: int) -> None:
    frame = pd.read_csv(spec.path, usecols=list(spec.columns), dtype={c: str for c in spec.key})
    records = normalize_records(frame, spec.columns)
    sql = text(upsert_sql(spec))
    with engine.begin() as conn:
        for start in range(0, len(records), SQL_BATCH):
            conn.execute(sql, records[start:start + SQL_BATCH])
    actual = db_count(engine, spec.table)
    print(f"[WRITE] {spec.table}: DB {actual:,}/{expected:,}")
    if actual != expected:
        die(f"{spec.table} 최종 count 불일치")


def canonical_rows(frame: pd.DataFrame, cols: tuple[str, ...]) -> list[tuple[str, ...]]:
    return sorted(tuple("<NULL>" if pd.isna(v) else str(v) for v in row)
                  for row in frame.loc[:, list(cols)].itertuples(index=False, name=None))


def handle_transitions(engine, expected: int, execute: bool) -> None:
    actual = db_count(engine, TRANSITIONS.table)
    print(f"[AUDIT] industry_transitions: DB {actual:,} / source {expected:,}")
    source = pd.read_csv(TRANSITIONS.path, usecols=list(TRANSITIONS.columns), dtype=str)
    if actual == expected:
        with engine.connect() as conn:
            db = pd.read_sql(text(
                "SELECT store_id,from_snapshot,to_snapshot,from_industry_code,to_industry_code "
                "FROM industry_transitions"), conn, dtype=str)
        if canonical_rows(db, TRANSITIONS.columns) != canonical_rows(source, TRANSITIONS.columns):
            die("industry_transitions count는 같지만 전체 내용이 source와 다름")
        print("[SKIP] industry_transitions: 전체 내용 일치")
        return
    if actual != 0:
        die(f"industry_transitions 부분 적재 상태({actual:,}/{expected:,}) — 자동 수정 금지")
    if not execute:
        print(f"[PLAN] industry_transitions: 단일 transaction으로 {expected:,}행 insert")
        return
    records = normalize_records(source, TRANSITIONS.columns)
    cols = ",".join(f"`{c}`" for c in TRANSITIONS.columns)
    vals = ",".join(f":{c}" for c in TRANSITIONS.columns)
    stmt = text(f"INSERT INTO industry_transitions ({cols}) VALUES ({vals})")
    with engine.begin() as conn:  # all-or-nothing, one transaction
        for start in range(0, len(records), SQL_BATCH):
            conn.execute(stmt, records[start:start + SQL_BATCH])
    final = db_count(engine, TRANSITIONS.table)
    if final != expected:
        die(f"industry_transitions 최종 count 불일치 {final:,}!={expected:,}")
    print(f"[WRITE] industry_transitions: {final:,}행")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="DB를 변경하지 않고 소스와 현재 적재 상태만 검증(기본값)")
    mode.add_argument("--execute", action="store_true",
                      help="명시했을 때만 비운영 피처 테이블을 안전 재개")
    args = parser.parse_args()
    print("MODE:", "EXECUTE" if args.execute else "DRY-RUN (read-only)")
    print("Protected tables (never written):", ", ".join(sorted(PROTECTED_TABLES)))

    engine = get_engine()
    if engine is None:
        die("DB 연결 정보 없음(.env 확인)")
    all_tables = [s.table for s in MASTER_SPECS] + [
        SPATIAL.table, "trend_keywords", SURVIVAL.table, TRANSITIONS.table]
    require_tables(engine, all_tables)

    with tempfile.TemporaryDirectory(prefix="tidb_resume_audit_") as td:
        seen = sqlite3.connect(str(pathlib.Path(td) / "keys.sqlite3"))
        expected = {s.table: scan_source(s, seen) for s in MASTER_SPECS}
        expected[SPATIAL.table] = scan_source(SPATIAL, seen)
        expected[SURVIVAL.table] = scan_source(SURVIVAL, seen)
        expected[TRANSITIONS.table] = scan_source(TRANSITIONS, seen)
        snapshots, expected["trend_keywords"] = audit_trend_source(seen)
        seen.close()

    # Fail closed before any possible write.
    audit_masters(engine, expected)
    for table in (SPATIAL.table, "trend_keywords", SURVIVAL.table):
        actual = db_count(engine, table)
        want = expected[table]
        print(f"[AUDIT] {table}: DB {actual:,} / source {want:,} ({actual/want*100:.2f}%)")
        if actual > want:
            die(f"{table}: DB count가 source보다 큼")
    handle_transitions(engine, expected[TRANSITIONS.table], execute=False)

    if not args.execute:
        print("DRY-RUN PASS: 쓰기 없음. 검증 통과 후 --execute를 별도로 명시하세요.")
        return 0

    execute_spatial(engine, expected[SPATIAL.table])
    execute_trend(engine, snapshots, expected["trend_keywords"])
    execute_simple_upsert(engine, SURVIVAL, expected[SURVIVAL.table])
    handle_transitions(engine, expected[TRANSITIONS.table], execute=True)
    print("EXECUTE COMPLETE: 보호된 운영 4개 테이블은 쓰지 않았습니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
