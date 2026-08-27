# DB 적재 가이드

서울 상권 폐업예측 프로젝트 — `data/raw/*.csv` 원본부터 TiDB Cloud까지 전체 파이프라인을 처음부터 돌리는 방법. (2026-08-27 기준, 전체 파이프라인 실제로 끝까지 성공 확인됨.)

## 0. 사전 준비

**환경 변수** — 프로젝트 루트에 `.env` 파일 (git에 커밋 금지):
```dotenv
DB_HOST=gateway01.ap-northeast-1.prod.aws.tidbcloud.com
DB_PORT=4000
DB_USERNAME=<클러스터prefix>.root
DB_PASSWORD=<비밀번호>
DB_DATABASE=seoul_market
# 선택: TiDB Cloud Serverless는 보통 공인 CA라 안 넣어도 certifi 기본 CA로 붙음
# DB_SSL_CA=/path/to/ca.pem
```
`DB_` 접두사를 꼭 붙일 것 — Windows는 `USERNAME`을 로그인 계정명으로 이미 갖고 있어서, 접두사 없이 쓰면 `.env` 값이 OS 값한테 밀리는 문제가 있었음.

**패키지 설치** (`uv` 사용 시):
```powershell
uv sync 꼭 한번더
```

**연결 확인**:
```powershell
python -m app.shared.db
```
`연결 성공: ...`이 뜨면 OK.

## 1. 전체 실행 순서

전부 **프로젝트 루트**(`C:\SKN35-2nd-3Team`)에서 실행 — 모든 스크립트가 `data/raw`, `data/features` 상대경로를 씀.

```powershell
.\run_pipeline.ps1

**의존관계**: `build_closure_transitions`(pkl 2개 + `industry_transitions.csv` 생성) → `build_store_snapshots`(그 pkl 사용, `store_snapshots.csv`/`stores.csv` 생성) → `build_industries`/`build_population_features`는 raw만 있으면 독립적으로 실행 가능 → `build_spatial_features`/`build_survival_stats`/`build_trend_keywords`는 `store_snapshots.csv`가 나온 뒤에 → 마지막에 `load_to_tidb.py`.

각 스크립트가 만드는 산출물 (전부 `data/features/`):

| 스크립트 | 산출물 |
|---|---|
| `build_closure_transitions.py` | `closed_ids_by_snap.pkl`, `transition_by_snap.pkl`, `industry_transitions.csv` |
| `build_store_snapshots.py` | `store_snapshots.csv`, `stores.csv` |
| `build_industries.py` | `industries.csv` |
| `build_population_features.py` | `population_features.csv`, **`administrative_dongs.csv`** |
| `build_spatial_features.py` | `spatial_density_features.csv` |
| `build_survival_stats.py` | `industry_survival_stats.csv` |
| `build_trend_keywords.py` | `trend_keywords.csv` |

## 2. `load_to_tidb.py` 실행 방식

```powershell
python src/project_2nd/db/etl/load_to_tidb.py            # 기본: 이어서 하기
python src/project_2nd/db/etl/load_to_tidb.py --reset     # 완전 초기화 후 처음부터
```

- **기본 동작(이어서 하기)**: 이미 데이터 있는 테이블은 건너뜀. `store_snapshots`/`spatial_density_features`(수백만 행이라 제일 오래 걸림)는 이미 들어간 만큼 청크를 건너뛰고 그 뒤부터 이어서 적재함. 뒤쪽 작은 테이블에서 에러 나서 재실행해야 할 때 앞의 느린 단계를 또 하지 않아도 됨.
- **`--reset`**: 13개 테이블 전부 지우고 완전히 새로 시작. 데이터 자체를 다시 만들었거나(원본 CSV 갱신 등) 뭔가 꼬여서 처음부터 확실히 다시 하고 싶을 때만.
- `users`/`models`/`predictions`/`support_actions` 4개는 스키마만 생성되고 데이터는 안 들어감 — 앱/모델 서빙 시작하면 그때 채워지는 운영 테이블이라 의도적으로 비워둠.

## 3. 오래 걸릴 때 진행 상황 확인

`store_snapshots`/`spatial_density_features`는 수백만 행이라 몇십 분 걸릴 수 있음(정상). 원래 돌아가는 터미널은 그대로 두고, **새 터미널**에서:

```powershell
python -c "from app.shared.db import get_engine; e=get_engine(); print(e.connect().exec_driver_sql('SELECT COUNT(*) FROM store_snapshots').scalar())"
```
(테이블 이름만 바꿔가며 다른 테이블도 같은 방식으로 확인 가능)

1~2분 간격으로 두 번 실행해서 숫자가 늘어나면 정상 진행 중, 안 늘어나면 멈춘 것.

## 4. 자주 나왔던 문제들 (참고용)

- **`ModuleNotFoundError: No module named 'app'`**: `app/`은 프로젝트 루트에, `db/`는 `src/project_2nd/` 밑에 있어서 경로 계산이 까다로움 — `load_to_tidb.py`가 이미 올바르게 처리하고 있음.
- **TiDB FK 타이밍 이슈** (`Failed to open the referenced table`): TiDB Cloud Serverless가 분산 환경이라 스키마 전파에 지연이 있음 — 스키마 생성 구간에서 `FOREIGN_KEY_CHECKS=0`으로 우회 처리됨.
- **`PendingRollbackError`**: 겉보기엔 커넥션 문제 같지만 실제로는 삽입하려는 데이터가 스키마 제약(NOT NULL/FK/UNIQUE)을 위반해서 나는 경우가 많음 — `load_to_tidb.py`가 insert 전에 미리 검증해서 진짜 원인을 바로 보여주도록 되어 있음.
- **`administrative_dongs`의 dong_name/gu_name 결측**: `population_features.csv`와 상권(store) 데이터의 행정동 목록이 완벽히 일치하지 않아서 생기던 문제 — `build_population_features.py`가 두 소스를 합집합으로 모아서 해결함(`administrative_dongs.csv`).
- **적재가 너무 느림**: `app/shared/db.py`에 커넥션 타임아웃(10s/60s/60s)이 걸려 있어서, 응답 없이 무한정 멈추는 대신 명확한 에러로 실패함. CPU/네트워크 사용량이 계속 움직이면 정상 진행 중인 것.

## 5. 다음 단계

DB 적재 끝나면 화면(UI) 작업 — 이미 있는 `app/pages/*.py`, `app/shared/auth.py` 등 실제 코드 기준으로 이어서 진행.