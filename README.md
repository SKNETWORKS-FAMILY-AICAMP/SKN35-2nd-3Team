# 서울 상권 폐업예측 프로젝트 — 폴더 구조 안내

5인 팀 병렬 작업을 전제로 짠 구조입니다. `db/`, `features/`, `models/`는 `src/project_2nd/`
패키지 밑으로 통합돼 있고(`pyproject.toml`이 `uv_build` 기반이라 이 위치가 정식 패키지
경로), `app/`, `data/`, `docs/`, `notebooks/`는 패키지 코드가 아니라서 최상위에 그대로 둡니다.

---

## 최상위 구조

```
SKN35-2nd-3Team/
├── src/
│   └── project_2nd/       db·features·models를 통합한 파이썬 패키지 (아래 1~4번)
├── data/                   데이터 원본부터 최종 피처까지 단계별 저장
├── app/                    Streamlit 앱 (사용자 그룹별 화면)
├── docs/                   설계/로드맵/제안서/방법론 문서
├── notebooks/               EDA·실험용 노트북
├── pyproject.toml           패키지 정의 (uv_build, src/project_2nd 자동 인식)
├── requirements.txt         데이터/모델 파이프라인 의존성
├── uv.lock
├── run_pipeline.sh          파이프라인 전체 자동 실행 (Mac/Linux/Git Bash용)
├── run_pipeline.ps1         파이프라인 전체 자동 실행 (Windows PowerShell용)
├── .env / .env.example      TiDB 연결 정보 (.env는 git에 안 올라감)
└── .gitignore
```

---

## 0. `src/project_2nd/` — 패키지 루트

| 파일 | 내용 |
|---|---|
| `src/project_2nd/__init__.py` | 패키지 진입점 |
| `src/project_2nd/db/` | 1번 |
| `src/project_2nd/features/` | 2번 |
| `src/project_2nd/models/` | 3번 |

각 하위 폴더에는 빈 `__init__.py`를 하나씩 둬서 정식 임포트가 가능하게 합니다.

---

## 1. `src/project_2nd/db/` — DB 설계 및 적재

| 파일 | 내용 |
|---|---|
| `db/schema.sql` | 테이블 DDL (13개 테이블) |
| `db/erd.dot`, `db/erd.png` | ERD 다이어그램 |
| `db/테이블_설명.md` | 테이블별 상세 설명 |
| `db/etl/build_closure_transitions.py` | 폐업 라벨·업종전환 이력 생성 |
| `db/etl/build_store_snapshots.py` | 스냅샷별 매장 테이블(`store_snapshots.csv`) + 매장 마스터(`stores.csv`) 생성 |
| `db/etl/load_to_tidb.py` | `schema.sql`로 13개 테이블 생성 + `data/features/*.csv`를 FK 순서에 맞춰 TiDB에 적재 |

**적재 실행**: `.env`에 TiDB Cloud 접속정보 채운 뒤 `python src/project_2nd/db/etl/load_to_tidb.py` — `administrative_dongs`, `industries`, `stores`, `store_snapshots`, `population_features`, `spatial_density_features`, `trend_keywords`, `industry_transitions`, `industry_survival_stats` 9개 테이블이 채워진다. `users`/`models`/`predictions`/`support_actions` 4개는 앱이 실제로 돌아가면서 채워지는 운영 데이터라 테이블만 생성되고 비어있다(4번 참고).

---

## 2. `src/project_2nd/features/` — 피처 엔지니어링

| 폴더 | 내용 |
|---|---|
| `features/spatial/` | `build_spatial_features.py`(BallTree/Haversine 기반 밀도 피처 4종), `build_population_features.py`(생활인구 피처) |
| `features/industry_grouping/` | `build_industries.py` — 업종 소분류(247개)를 대분류(10개) 커스텀 그룹으로 매핑 |
| `features/trend_keywords/` | `build_trend_keywords.py` — 상호명 키워드 기반 '지금 뜨는 사업' 탐지, 6개 스냅샷 매장수 증가율 계산 (실데이터 기반, 더미 없음) |
| `features/survival_transition/` | `build_survival_stats.py` — 업종 전환 이력 기반 생존율 피처 |

---

## 3. `src/project_2nd/models/` — 모델링

| 폴더 | 내용 |
|---|---|
| `models/ml/build_modeling_dataset.py` | 위 피처들을 전부 합쳐 학습용 최종 데이터셋(`modeling_dataset.csv`) 조립. GroupKFold 배정, fold-safe 과거폐업률 인코딩, 스코프 제외(과학·기술/부동산/시설관리·임대) 포함 |
| `models/ml/` (베이스라인 학습, 임계값 튜닝) | 팀원이 직접 작성 중 |
| `models/dl/` | 딥러닝(MLP 등) — 아직 코드 없음, 폴더만 존재 |
| `models/shap/` | SHAP 기반 설명가능성 — 아직 코드 없음, 폴더만 존재 |

---

## 4. `app/` — 화면 (Streamlit, 사용자 그룹별) — 최상위 유지

| 파일/폴더 | 내용 |
|---|---|
| `app/app.py` | 진입점. 사용자 유형 선택 → 로그인 라우팅 (현재 스텁) |
| `app/requirements.txt` | 앱 실행 전용 의존성 |
| `app/build_features_and_model.py` | 서빙용 피처+모델 준비 스크립트 (현재 스텁) |
| `app/pages/` | 실제 화면 6개, 파일명 숫자 prefix로 사이드바 순서 제어 (현재 스텁) |
| `app/shared/db.py` | TiDB 연결 유틸(`get_engine`) — **구현 완료** |
| `app/shared/write_user.py` | `users` 테이블에 로그인/회원가입 기록 — 함수 구현 완료, 호출부(`auth.py`)는 아직 |
| `app/shared/write_model.py` | `models` 테이블에 학습된 모델 등록 — 함수 구현 완료, 학습 스크립트에서 호출 필요 |
| `app/shared/write_prediction.py` | `predictions` 테이블에 예측+SHAP 결과 기록 — 함수 구현 완료, 예측 화면에서 호출 필요 |
| `app/shared/write_support_action.py` | `support_actions` 테이블에 관리자 조치 기록 — 함수 구현 완료, 관리자 화면에서 호출 필요 |
| `app/shared/auth.py`, `components.py` | 로그인 로직, 공용 UI — 현재 스텁 |
| `app/founder/`, `app/owner/`, `app/admin/` | 유형별 로직 모듈 — 현재 빈 폴더 |

**write_*.py 4개는 "완성된 부품"이고, 아직 어디서도 호출되고 있지 않다.** 로그인 화면이 완성되면 `write_user.create_user()`를, 모델 학습이 끝나면 `write_model.register_model()`을, 예측 화면이 완성되면 `write_prediction.log_prediction()`을, 관리자 조치 폼이 완성되면 `write_support_action.log_support_action()`을 그 자리에서 호출하기만 하면 된다.

**실행 방법(예정)**
```
pip install -r requirements.txt          # 데이터/모델 파이프라인
pip install -r app/requirements.txt      # 앱 전용
python app/build_features_and_model.py   # 서빙용 피처+모델 준비 (최초 1회)
streamlit run app/app.py
```

---

## 5. `data/` — 데이터 파이프라인 단계 (최상위 유지)

| 폴더 | 내용 |
|---|---|
| `data/raw/` | 원본 CSV 전부. 소상공인은 `seoul_YYYYMM.csv`, 생활인구는 `local_pop.csv`/`longf_pop.csv`/`tempf_pop.csv` |
| `data/features/` | 최종 피처 테이블. 파이프라인의 실질적인 출력 폴더 |
| `data/processed/`, `data/labeled/` | **현재 미사용.** 원래 `raw → processed(정리) → labeled(라벨링) → features` 4단계로 설계했지만, 실제 원본 데이터가 인코딩 문제 없이 깨끗했고 폐업 라벨링도 `store_snapshots` 생성 단계에 바로 통합돼서 `raw → features` 2단계로 단순화됐다. 두 폴더 다 어떤 스크립트에서도 참조하지 않으므로 삭제해도 무방하다. |

(`data/` 하위 산출물은 `.gitignore`에서 제외되지만, `data/features/`의 작은 참고용 CSV 5개(`industries.csv`, `trend_keywords.csv`, `industry_survival_stats.csv`, `industry_transitions.csv`, `population_features.csv`)는 예외적으로 git에 포함된다.)

---

## 6. `docs/` — 문서 (최상위 유지)

## 7. `notebooks/` — 실험 (최상위 유지)

---

## 참고: 진행 상태

**완료됨**
- `src/project_2nd/db/etl/` — 원본 CSV → 폐업라벨/업종전환/store_snapshots/stores 생성
- `src/project_2nd/db/etl/load_to_tidb.py` — 9개 테이블 TiDB 적재 스크립트, 각 스크립트 출력 컬럼과 1:1 대조 검증 완료
- `src/project_2nd/features/` — 공간밀도, 업종 마스터, 트렌드키워드, 생존율 통계, 생활인구
- `src/project_2nd/models/ml/build_modeling_dataset.py` — 학습용 데이터셋 조립 (스코프 제외 포함)
- `src/project_2nd/db/erd.dot`, `db/테이블_설명.md`, `db/schema.sql` — 13개 테이블 ERD + DDL, `dong_code`/`shap_top_features` 반영
- `app/shared/db.py`, `write_user.py`, `write_model.py`, `write_prediction.py`, `write_support_action.py` — DB 연결 및 4개 테이블 쓰기 함수
- `run_pipeline.sh` / `run_pipeline.ps1` — Mac·Linux·Windows 전부 지원, 문법 검증 완료

**아직 안 된 것**
- 모델 베이스라인 학습·임계값 튜닝 (팀원이 직접 진행 중)
- `app/`의 각 화면 실제 로직 (진입점, 로그인, 6개 페이지 전부 스텁) — 완성되면 위 write_*.py 함수들을 호출하기만 하면 됨
- `src/project_2nd/models/dl/`, `src/project_2nd/models/shap/` — 아직 코드 없음
- Random Forest, MLP와의 성능 비교
- 카카오맵 API 등 외부 데이터 연동
- `users`/`models`/`predictions`/`support_actions` 4개 테이블 — 위 앱 로직이 완성돼야 채워짐