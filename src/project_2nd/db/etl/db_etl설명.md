## DB 연결 & 적재 진행 상황

작성일: 2026-08-27
**상태: DB 적재 완료 (2026-08-27).** 아래는 완료까지의 과정 기록.

## 실제 프로젝트 구조 (전체 zip으로 확정됨)

```
C:\SKN35-2nd-3Team\                  <- 프로젝트 루트
  app/
    app.py                           <- Streamlit 메인 (이미 존재!)
    build_features_and_model.py
    requirements.txt                 <- DB 관련 패키지 추가함(완료)
    pages/ 1_founder_score.py ~ 6_admin_trend.py
    shared/ db.py(완성, 타임아웃 추가됨), auth.py, components.py, write_*.py
    admin/, founder/, owner/ (현재 .gitkeep만, 비어있음)
  data/                               <- 루트 바로 밑 (확정됨)
    raw/ (원본 6개 스냅샷 CSV), features/ (전처리 산출물 CSV — administrative_dongs.csv 새로 추가됨)
  src/project_2nd/
    db/ schema.sql, etl/{build_closure_transitions,build_store_snapshots,load_to_tidb}.py
    features/spatial/{build_population_features,build_spatial_features}.py,
    features/industry_grouping/build_industries.py, features/survival_transition/build_survival_stats.py,
    features/trend_keywords/build_trend_keywords.py, models/ml/build_modeling_dataset.py, preprocessing_dataset_pjw/
```

**중요**: `app/`에 이미 팀원이 만든 실제 화면(pages 6개, auth, write_* 헬퍼)이 있음. 다음 작업은 이 기존 코드 기준으로 화면 작업 재개.

## 최종 적재 결과 (2026-08-27, 에러 없이 전체 완료)

| 테이블 | 행 수 |
|---|---|
| administrative_dongs | 436 |
| industries | 247 |
| stores | 778,895 |
| store_snapshots | 수백만(정확한 수는 DB에서 `SELECT COUNT(*)`로 확인 가능) |
| population_features | 424 |
| spatial_density_features | store_snapshots와 동일 규모 |
| trend_keywords | 84 (롱 포맷 변환됨) |
| industry_transitions | 21,767 |
| industry_survival_stats | 3,529 |
| users / models / predictions / support_actions | 0 (의도된 설계 — 앱/모델 서빙 시작하면 채워짐) |

`industry_transitions`/`industry_survival_stats` 적재 시 FK 불일치 경고가 전혀 없었음 — "6개 스냅샷이 동일한 업종분류 체계를 쓴다"는 팀의 검증 전제가 실제로도 맞았음이 확인됨.

## 스키마/적재 단계에서 발견하고 고친 것들 (요약 — 세부는 아래 계속)

**스키마 생성**: `app` 모듈 경로, `schema.sql` 위치, TiDB FK 타이밍(FK_CHECKS=0 우회), SQLAlchemy `text()`의 `:숫자` 오인식(→`exec_driver_sql`), 재실행 안전성, 스키마 전파 대기, **schema.sql 주석 파싱 버그**(섹션 헤더 주석이 바로 다음 CREATE TABLE까지 통째로 삼켜서 13개 중 8개만 생성되던 근본 원인) — 전부 해결.

**데이터 품질 — 소스 단계로 이관**: `administrative_dongs`(dong_name/gu_name) 결측을 DB 적재 시점 placeholder로 땜질하지 않고, `features/spatial/build_population_features.py`를 고쳐서 생활인구 dong_code ∪ 상권 6개 스냅샷 dong_code 합집합으로 완전한 마스터(`administrative_dongs.csv`)를 직접 생성하도록 근본 수정. gu_name은 dong_code 앞 5자리로 100% 결정(GU_CODE_MAP, `models/ml/build_modeling_dataset.py`와 동일 원칙). dong_name은 정말 알 방법 없는 경우만 '(미상)'.

**정합성 검증**: `load_store_snapshots()`/`load_industry_transitions()`/`load_industry_survival_stats()`에 insert 전 dong_code/industry_code 존재, NOT NULL, 중복 검증 추가 — PendingRollbackError처럼 뭉뚱그려진 에러 대신 정확한 원인이 바로 드러나게 함.

**성능**: 매번 stores 778,895행 전체를 SELECT해서 하던 불필요한 store_id 대조 제거(구조상 항상 유효), `to_sql` 배치 크기 2000→5000.

**재개(resume) 기능**: `create_tables()` 기본 동작을 "이미 있는 테이블 유지"로 변경(`--reset`으로만 전체 초기화), store_snapshots/spatial_density_features는 이미 적재된 행 수만큼 청크를 건너뛰고 이어서 적재. 이번 실행은 결국 끝까지 한 번에 완료돼서 이 기능은 실사용되진 않았지만, 다음에 비슷한 대량 작업(데이터 갱신, 재적재 등) 필요할 때 계속 유효함.

**DB 커넥션**: `app/shared/db.py`에 pymysql connect/read/write timeout 추가(10s/60s/60s) — 대량 적재 중 응답 없이 멈춘 것처럼 보이는 상황에서 무한 대기 대신 명확한 에러로 실패하게 함.

## 다음 단계: 화면(UI) 작업 재개

DB 적재가 끝났으니 이제 기존 `app/pages/*.py`, `app/shared/auth.py` 등 실제 코드를 기준으로 화면 작업을 이어가면 됨 — [[seoul-biz-ui-logic]] 문서와 대조해서 반영. (이 세션 초반에 만든 `/home/claude/seoul-biz-ui` 스캐폴드는 실제 `app/` 코드가 이미 있는 게 확인된 이상 참고용일 뿐, 그걸 기반으로 새로 시작하지 않음.)