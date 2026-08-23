# 서울 상권 폐업예측 프로젝트 — 폴더 구조 안내

5인 팀 병렬 작업을 전제로 짠 구조입니다. 코드는 최소한의 진입점/스텁만 채워뒀고,
실제 로직은 담당자가 채워 넣으면 됩니다. 완전히 비어 있는 폴더에는 `.gitkeep`만 있습니다.

---

## 최상위 구조

```
seoul-closure-prediction/
├── data/              데이터 원본부터 최종 피처까지 단계별 저장
├── db/                DB 스키마·ERD·적재(ETL) 스크립트
├── features/           피처 엔지니어링 로직
├── models/             ML/DL 모델 학습·설명가능성
├── app/                Streamlit 앱 (사용자 그룹별 화면)
├── docs/                설계/로드맵/제안서/방법론 문서
├── notebooks/           EDA·실험용 노트북
├── requirements.txt     데이터/모델 파이프라인 의존성
└── .gitignore
```

---

## 1. `data/` — 데이터 파이프라인 단계

| 폴더 | 내용 |
|---|---|
| `data/raw/` | 원본 CSV 전부(소상공인 6개 스냅샷 + 생활인구 3개) 한 폴더에 그대로 둔다. 파일명으로 구분: 소상공인은 `seoul_YYYYMM.csv`, 생활인구는 `local_pop.csv`(내국인)/`longf_pop.csv`(외국인 장기)/`tempf_pop.csv`(외국인 단기) |
| `data/processed/` | 인코딩(UTF-8)·BOM 보정 등 정리된 중간 산출물 |
| `data/labeled/` | 상가업소번호 기준 폐업 라벨링이 완료된 데이터셋 |
| `data/features/` | 모델 입력 직전 단계의 최종 피처 테이블 |

**원칙**: raw는 절대 직접 수정하지 않고, processed부터 파생시킨다. 이렇게 해야 "전처리 결과서"에서 각 단계의 변환 근거를 명확히 설명할 수 있다. (`data/` 하위 산출물은 `.gitignore`에 의해 git에는 올라가지 않는다 — 용량 문제.)

---

## 2. `db/` — DB 설계 및 적재

| 폴더/파일 | 내용 |
|---|---|
| `db/schema.sql` | 테이블 DDL |
| `db/erd.png` / `db/erd.drawio` | ERD 다이어그램 |
| `db/etl/` | CSV → DB 적재 스크립트 (스냅샷별 적재, 생활인구 적재, 피처 뷰 생성 등) |

---

## 3. `features/` — 피처 엔지니어링

| 폴더 | 내용 |
|---|---|
| `features/spatial/` | BallTree/Haversine 기반 반경 내 경쟁업소 밀도 피처 |
| `features/industry_grouping/` | 업종명 기반 그룹핑 (837→247 코드체계 변경 대응, 텍스트 라벨을 안정적 브릿지로 사용) |
| `features/trend_keywords/` | 상호명 키워드(예: 탕후루, 버터떡) 기반 '지금 뜨는 사업' 탐지 — 6개 스냅샷 매장수 증가율 계산 (실데이터 기반, 더미 없음) |
| `features/survival_transition/` | 업종 전환 이력 기반 생존율 피처 (업종전환 추천 근거) |

---

## 4. `models/` — 모델링

| 폴더 | 내용 |
|---|---|
| `models/ml/` | 트리 기반 등 머신러닝 모델 학습·튜닝 코드, `saved/`에 학습 완료 모델 저장 |
| `models/dl/` | 딥러닝(MLP 등) 모델 학습·튜닝 코드, `saved/`에 학습 완료 모델 저장 |
| `models/shap/` | SHAP 기반 설명가능성 — 점수 산출 근거를 사용자에게 보여주기 위함 |

---

## 5. `app/` — 화면 (Streamlit, 사용자 그룹별)

Bank Churn 트랙 때처럼 Streamlit으로 데모를 만든다. Streamlit은 `pages/` 폴더 안의 파일을
자동으로 사이드바 메뉴로 인식하므로, 진입점(`app.py`)과 화면(`pages/`)을 분리했다.

| 파일/폴더 | 내용 |
|---|---|
| `app/app.py` | **진입점.** 사용자 유형(예비창업자/기존점주/관리자) 선택 → 로그인 라우팅 |
| `app/requirements.txt` | 앱 실행 전용 의존성 (streamlit, plotly 등) |
| `app/build_features_and_model.py` | 서빙용 피처 테이블 + 최종 모델 준비 스크립트 (최초 1회 실행) |
| `app/pages/` | 실제 화면. 파일명 숫자 prefix로 사이드바 순서 제어 (`1_founder_score.py` ~ `6_admin_trend.py`) |
| `app/shared/` | 공통 모듈 — `auth.py`(로그인/세션), `db.py`(DB 조회), `components.py`(위험도 배지 등 공용 UI) |
| `app/founder/` | 예비창업자용 로직 모듈 (pages에서 import) |
| `app/owner/` | 기존점주용 로직 모듈 — 건강검진 점수, 업종전환 추천 |
| `app/admin/` | 관리자용 로직 모듈 — 모니터링, 지원 액션 로그, 뜨는 사업 탐지 |

**실행 방법(예정)**
```
pip install -r requirements.txt          # 데이터/모델 파이프라인
pip install -r app/requirements.txt      # 앱 전용
python app/build_features_and_model.py   # 서빙용 피처+모델 준비 (최초 1회)
streamlit run app/app.py
```

---

## 6. `docs/` — 문서

기존에 작성한 설계 문서들을 이곳에 모은다:
- 서울상권_폐업예측_프로젝트설계.txt
- 서울상권_폐업예측_종합로드맵.txt
- 서울상권_폐업예측_제안서.txt
- 업종그룹핑_방법론_정리.txt

---

## 7. `notebooks/` — 실험

EDA, 피처 검증, 모델 실험 등 정식 스크립트로 옮기기 전 단계의 탐색적 작업용.

---

## 참고: 진행 상태

**완료됨** (실행 가능한 코드 + 실제 데이터로 검증 완료)
- `db/etl/` — 원본 CSV → 폐업라벨/업종전환/store_snapshots/stores 생성
- `features/` — 공간밀도, 업종 마스터, 트렌드키워드, 생존율 통계, 생활인구
- `models/ml/` — 학습용 데이터셋 조립 + LightGBM 베이스라인(ROC-AUC 0.725) + 임계값 튜닝
- `db/erd.png`, `db/erd.dot`, `db/테이블_설명.md`, `db/schema.sql` — 13개 테이블 ERD + DDL
- `data/features/`에 작은 산출물(업종마스터, 트렌드키워드, 생존율, 전환이력, 생활인구) 포함
  — 대용량 파일(store_snapshots, spatial_density_features, modeling_dataset)은 용량 문제로 zip에는 미포함, 스크립트를 실행하면 로컬에서 재생성됨

**아직 안 된 것**
- TiDB 실제 적재 스크립트 (`db/schema.sql`은 완료, 실행/적재만 남음)
- `app/`의 각 화면 로직 (현재는 스텁 파일만 존재)
- Random Forest, MLP와의 성능 비교
- 카카오맵 API 등 외부 데이터 연동 (팀 로컬 환경에서 API 키로 실행 필요, 이번 세션에서는 스크립트 작성 전 단계)
