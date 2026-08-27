# 데이터 사전 (Feature Dictionary) — `modeling_dataset_refined_pjw.csv`

`data/processed/modeling_dataset_refined_pjw.csv` 기준, 37개 컬럼. 결측률/고유값 수는
`preprocess_modeling_dataset_pjw.py` 실행 결과에서 직접 뽑은 실측치(2026-08-26 최신 파이프라인 기준).

원본(팀원 `data/features/modeling_dataset.csv`, 34컬럼)과 차이:
- 제거: `total_pop_avg`(단순 합, 중복), `transitioned_next`(타깃 누수 컬럼), `dong_code`(ablation 검증 후 제거해도 무방)
- 추가: `is_mass_reclass_window`, `is_left_censored_age`, `is_trend_keyword_match`, `industry_specialization_300m`, `competition_per_capita_300m`, `dong_industry_count_growth`

---

## 1. 식별자 / 메타

| 컬럼 | 타입 | 결측 | 의미 |
|---|---|---|---|
| `store_id` | str | 0% | 매장 고유 ID (538,147개 유니크) |
| `snapshot_date` | int | 0% | 관측 시점. `202312`/`202406`/`202412`/`202506`/`202512`/`202606` 6개 중 5개만 피처로 존재(마지막 스냅샷은 `is_closed_next` 계산 불가라 라벨 없음) |
| `fold` | int(0~4) | 0% | GroupKFold 배정 결과. `store_id`를 md5 해시해 5등분 — 같은 매장은 항상 같은 fold (누수 방지) |
| `is_closed_next` | int(0/1) | 0% | **타깃**. 다음 스냅샷에서 이 매장이 사라졌는지(폐업 추정). 전체 평균 약 10.6% |

## 2. 업종 정보

| 컬럼 | 타입 | 결측 | 의미 |
|---|---|---|---|
| `industry_code` | str | 0% | 세분류 업종 코드 (192개). **feature importance 1위(39%)** — 가장 강력한 신호 |
| `industry_name` | str | 0% | 세분류 업종명 (예: "백반/한정식", "카페") |
| `industry_jung_code` / `industry_jung_name` | str | 0% | 중분류 업종 코드/명 (53개) |
| `industry_dae_code` | str | 0% | 대분류 코드 (7개) |
| `industry_group` (`custom_group`) | str | 0% | 팀원분이 재그룹핑한 업종 대분류 10개 중 7개가 최종 데이터셋에 남음(음식/소매/수리·개인/교육/예술·스포츠 등). 과학·기술/부동산/시설관리·임대 3개 그룹은 폐업 패턴이 이질적이라 스코프에서 제외됨 |

## 3. 위치 / 공간 피처

| 컬럼 | 타입 | 결측 | 의미 |
|---|---|---|---|
| `gu_name` | str | 0% | 자치구명(25개). dong_code 앞 5자리로 100% 결정되는 값 — 우리(v4)와 팀원분(v5) 양쪽에서 독립적으로 채움 검증 |
| `lng` / `lat` | float | 0% | 매장 좌표(경도/위도) |
| `floor_category` | str | 0% | 층 정보 5종(1층/2층이상/지하/기타/결측). 팀원분이 검증한 유일하게 효과 있던 외부 피처(ROC-AUC 0.721→0.728) |
| `coord_cluster_size` | int | 0% | 같은 좌표를 공유하는 매장 수(DBSCAN, 스냅샷별 계산 — 시점 누수 없음). 대형 복합상가/아파트 상가 입점 여부의 대리 신호 (최대 883개 사례: 문정동 현대시티몰) |
| `same_industry_count_300m` | int | 0% | 반경 300m 내 같은 업종 매장 수 |
| `total_count_300m` | int | 0% | 반경 300m 내 전체 매장 수 |
| `nearest_same_industry_distance_m` | float | 0% | 가장 가까운 동일 업종 매장까지 거리(m) |
| `dong_industry_count` | int | 0% | 같은 행정동+업종 조합의 매장 수 |
| `industry_specialization_300m` (우리 추가) | float | 0.01% | `same_industry_count_300m / total_count_300m` — 반경 내 업종 특화도. `total_count_300m==0`이면 NaN |
| `competition_per_capita_300m` (우리 추가) | float | 0% | `same_industry_count_300m / (생활인구 합)` — 인구 대비 경쟁 밀도 |
| `dong_industry_count_growth` (우리 추가) | float | 2.91% | `(dong_code, industry_code)` 그룹 내 `dong_industry_count`의 전기 대비 증감률. 각 그룹의 첫 스냅샷은 이전 값이 없어 NaN |

## 4. 매장 특성 / 이력

| 컬럼 | 타입 | 결측 | 의미 |
|---|---|---|---|
| `store_age_months` | int | 0% | 매장 나이(개월). `(snapshot_date - first_seen_snapshot) * 6`. **좌측절단 주의**: 첫 스냅샷(202312)에 이미 있던 매장은 실제 개업일 불명이라 과소추정됨 |
| `is_left_censored_age` (우리 추가) | int(0/1) | 0% | 위 좌측절단 문제를 명시하는 플래그. `first_seen_snapshot=='202312'`인 행이 1 (전체의 78.85%) |
| `previously_transitioned` | int(0/1) | 0% | 과거에 업종을 전환한 이력이 있는지 |
| `is_mass_reclass_window` (우리 추가) | int(0/1) | 0% | `snapshot_date==202406`이면 1 — 소상공인시장진흥공단 대규모 업종 재정비 구간으로 추정되는 시점(2024/06 스냅샷에서 21,194건의 업종전환 이력이 몰림) |
| `keyword_growth_score` | float | 0% | 트렌드 키워드 상호명 매칭 시 해당 키워드의 growth_rate(연속형). 매칭 안 되거나 growth_rate≤0이면 0 |
| `is_trend_keyword_match` (우리 추가) | int(0/1) | 0% | 위와 별개로 "키워드 매칭 자체" 여부만 이진으로 분리(growth_rate 부호 무관). growth_score=0인데 실제로는 매칭된 행이 4,839건 존재해서 추가 |

## 5. 생활인구 (`data/raw` 생활인구 원본 → `population_features.csv`)

| 컬럼 | 타입 | 결측 | 의미 |
|---|---|---|---|
| `korean_pop` | float | 0% | 내국인 생활인구(평균) |
| `foreign_long_pop` | float | 0% | 장기체류 외국인 생활인구 |
| `foreign_short_pop` | float | 0% | 단기체류 외국인 생활인구 |
| `foreign_short_ratio` | float | 0% | `foreign_short_pop / total_pop_avg` |
| `tourist_zone_candidate` | bool | 0% | `foreign_short_ratio` 상위 10% 행정동이면 True (관광특구 후보 추정) |
| `population_is_proxied` (팀원분 v5 추가) | bool | 0% | 원래 `population_features.csv`에 dong_code가 없어(12개 동, 37,066행) BallTree 최근접 이웃으로 근처 동 값을 근사 대체한 행이면 True — **근사치이므로 해석 시 유의** |

## 6. Fold-safe historical rate (타깃 인코딩)

각 fold의 테스트 구간은 **해당 fold를 제외한 나머지로 계산한 값**만 사용(누수 방지). `industry_historical_rate`는 15개 조합 전수 검증 결과 정확히 일치 확인. `dong_historical_rate`는 스코프 필터(과학·기술/부동산/시설관리·임대 제외) **적용 이전**에 계산되는 값이라, 최종 데이터셋만으로 재계산하면 약 30% 높게 나옴 — 버그 아니라 계산 시점 차이(자세한 검증은 `preprocess_report_pjw.md` §5 참고).

| 컬럼 | 타입 | 결측 | 의미 |
|---|---|---|---|
| `industry_historical_rate` | float | 0% | 업종별 과거 폐업률(fold-safe) |
| `dong_historical_rate` | float | 0% | 행정동별 과거 폐업률(fold-safe, 스코프 필터 이전 기준값) |
| `dong_industry_historical_rate` | float | 0% | 행정동+업종 조합별 과거 폐업률(fold-safe, 표본 30건 미만이면 업종 전체 비율로 대체) |

---

## 참고
- 결측이 남아있는 컬럼은 `industry_specialization_300m`(0.01%, 164행)과 `dong_industry_count_growth`(2.91%, 54,940행) 둘뿐 — 둘 다 정의상 계산 불가능한 경우(분모 0 또는 그룹의 첫 관측치)라 자연스러운 결측이며, LightGBM 등 트리 모델은 결측을 자체 처리하므로 별도 대체 불필요
- 원본 데이터/생성 스크립트: `data/features/modeling_dataset.csv` (팀원 파이프라인, `run_pipeline.sh`) → `preprocess_modeling_dataset_pjw.py` → `data/processed/modeling_dataset_refined_pjw.csv`
