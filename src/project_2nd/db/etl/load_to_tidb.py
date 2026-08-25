"""
db/etl/03_load_to_tidb.py

db/schema.sql로 테이블을 생성하고, data/features/*.csv를 순서대로 적재한다.
순서는 FK 의존관계를 지켜야 한다 (참조되는 테이블이 먼저 채워져야 함).

실행 전:
  1. .env.example을 .env로 복사하고 TiDB Cloud 연결 정보 채우기
  2. pip install pymysql sqlalchemy python-dotenv cryptography
  3. data/features/ 에 CSV들이 이미 만들어져 있어야 함 (run_pipeline.sh 먼저 실행)

실행: python db/etl/03_load_to_tidb.py
"""
import sys
import os
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from app.shared.db import get_engine

FEATURES_DIR = 'data/features'
SCHEMA_PATH = 'db/schema.sql'


def create_tables(engine):
    print("스키마 생성 중...")
    with open(SCHEMA_PATH, encoding='utf-8') as f:
        sql = f.read()
    statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"  {len(statements)}개 테이블 생성 완료")


def load_administrative_dongs(engine):
    # population_features.csv에서 dong_code/dong_name/gu_name 추출 (원본 소스)
    df = pd.read_csv(f'{FEATURES_DIR}/population_features.csv', dtype={'dong_code': str})
    dongs = df[['dong_code', 'dong_name', 'gu_name']].drop_duplicates()
    dongs.to_sql('administrative_dongs', engine, if_exists='append', index=False)
    print(f"administrative_dongs: {len(dongs):,}행 적재")


def load_industries(engine):
    df = pd.read_csv(f'{FEATURES_DIR}/industries.csv', dtype=str)
    df.to_sql('industries', engine, if_exists='append', index=False)
    print(f"industries: {len(df):,}행 적재")


def load_stores(engine):
    df = pd.read_csv(f'{FEATURES_DIR}/stores.csv', dtype={'store_id': str})
    df.to_sql('stores', engine, if_exists='append', index=False)
    print(f"stores: {len(df):,}행 적재")


def load_store_snapshots(engine):
    # snapshot_id는 AUTO_INCREMENT라 CSV에서 빼고 적재
    cols = ['store_id', 'snapshot_date', 'industry_code', 'dong_code', 'store_name',
            'floor_category', 'lng', 'lat', 'is_closed_next', 'transitioned_next', 'label_available']
    for chunk in pd.read_csv(f'{FEATURES_DIR}/store_snapshots.csv', usecols=cols,
                              dtype={'store_id': str, 'dong_code': str, 'industry_code': str,
                                     'snapshot_date': str},
                              chunksize=100_000):
        chunk.to_sql('store_snapshots', engine, if_exists='append', index=False)
    print("store_snapshots 적재 완료")


def load_population_features(engine):
    cols = ['dong_code', 'korean_pop', 'foreign_long_pop', 'foreign_short_pop',
            'total_pop_avg', 'foreign_short_ratio', 'tourist_zone_candidate']
    df = pd.read_csv(f'{FEATURES_DIR}/population_features.csv', usecols=cols, dtype={'dong_code': str})
    df.to_sql('population_features', engine, if_exists='append', index=False)
    print(f"population_features: {len(df):,}행 적재")


def load_spatial_density_features(engine):
    for chunk in pd.read_csv(f'{FEATURES_DIR}/spatial_density_features.csv',
                              dtype={'store_id': str, 'snapshot_date': str},
                              chunksize=100_000):
        chunk.to_sql('spatial_density_features', engine, if_exists='append', index=False)
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
    long_df.to_sql('trend_keywords', engine, if_exists='append', index=False)
    print(f"trend_keywords: {len(long_df):,}행 적재 (롱 포맷 변환)")


def load_industry_transitions(engine):
    df = pd.read_csv(f'{FEATURES_DIR}/industry_transitions.csv', dtype=str)
    df = df.drop(columns=['transition_id'])  # AUTO_INCREMENT
    df.to_sql('industry_transitions', engine, if_exists='append', index=False)
    print(f"industry_transitions: {len(df):,}행 적재")


def load_industry_survival_stats(engine):
    df = pd.read_csv(f'{FEATURES_DIR}/industry_survival_stats.csv', dtype=str)
    df.to_sql('industry_survival_stats', engine, if_exists='append', index=False)
    print(f"industry_survival_stats: {len(df):,}행 적재")


if __name__ == '__main__':
    engine = get_engine()

    create_tables(engine)

    # FK 의존관계 순서: 참조되는 테이블 먼저
    load_administrative_dongs(engine)
    load_industries(engine)
    load_stores(engine)
    load_store_snapshots(engine)
    load_population_features(engine)
    load_spatial_density_features(engine)
    load_trend_keywords(engine)
    load_industry_transitions(engine)
    load_industry_survival_stats(engine)

    # users, models, predictions, support_actions는 앱 실행 중 생성되는 운영 데이터라
    # 여기서는 적재하지 않는다 (테이블만 미리 생성해둠).

    print("\n전체 적재 완료.")