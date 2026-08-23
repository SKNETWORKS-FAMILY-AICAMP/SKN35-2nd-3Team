# modeling_dataset_v4.csv 컬럼 설명

총 2,591,877행, 31개 컬럼. 소상공인 폐업 예측 모델 학습용 최종 데이터셋.

---

## 식별/시점

| 컬럼 | 설명 |
|---|---|
| `snapshot_date` | 202312~202512 중 하나 (202606은 다음 시점이 없어 라벨을 매길 수 없어서 제외) |
| `store_id` | 상가업소번호 |
| `fold` | GroupKFold 5분할(0~4). store_id를 해시해서 배정 — 같은 매장이 train/test에 동시에 들어가지 않도록 함 |

---

## 업종

| 컬럼 | 설명 |
|---|---|
| `industry_dae_code` | 업종 대분류 코드 |
| `industry_group` | 업종 대분류명 (10개 그룹: 음식/소매/과학·기술/수리·개인/교육/부동산/시설관리·임대/예술·스포츠/보건의료/숙박) |
| `industry_jung_code` | 업종 중분류 코드 |
| `industry_jung_name` | 업종 중분류명 (75개) |
| `industry_code` | 업종 소분류 코드 (247개) — "동일 업종" 판정 기준이 되는 가장 세부 단위 |
| `industry_name` | 업종 소분류명 |

---

## 위치

| 컬럼 | 설명 |
|---|---|
| `gu_name` | 구 이름 |
| `dong_code` | 행정동코드 (생활인구 조인 키) |
| `lng` | 경도 |
| `lat` | 위도 |

---

## 공간 밀도

| 컬럼 | 설명 |
|---|---|
| `same_industry_count_300m` | 반경 300m 내 동일업종(소분류 기준) 매장 수. BallTree(haversine)로 계산 |
| `total_count_300m` | 반경 300m 내 업종 무관 전체 매장 수. 상권 자체가 얼마나 붐비는지를 나타냄 |
| `nearest_same_industry_distance_m` | 가장 가까운 동일업종 매장까지의 실제 거리(미터). 해당 스냅샷에서 동일업종이 자기 자신뿐이면 결측(NaN) |
| `dong_industry_count` | 반경이 아니라 행정동 전체 범위 기준 동일업종 매장 수 |

---

## 매장 이력

| 컬럼 | 설명 |
|---|---|
| `store_age_months` | 처음 관측된 스냅샷 대비 몇 개월째인지 (스냅샷이 6개월 간격이라 6의 배수) |
| `previously_transitioned` | 이 매장이 과거에 업종을 바꾼 적이 있는지 (0/1) |
| `keyword_growth_score` | 상호명에 트렌드 키워드가 포함되면 그 키워드의 실제 스냅샷간 증가율 값, 포함되지 않으면 0 |

---

## 생활인구 (행정동 단위, dong_code로 조인, 2026년 7월 한 달 평균)

| 컬럼 | 설명 |
|---|---|
| `korean_pop` | 내국인 생활인구 |
| `foreign_long_pop` | 외국인 장기체류 생활인구 |
| `foreign_short_pop` | 외국인 단기체류 생활인구 |
| `total_pop_avg` | 위 세 값의 합 |
| `foreign_short_ratio` | 단기외국인 비율 (foreign_short_pop / total_pop_avg) |
| `tourist_zone_candidate` | 위 비율이 상위 10%에 들면 1, 아니면 0 |

---

## 과거 폐업률 (fold-safe 타겟 인코딩)

각 fold의 값은 해당 fold를 제외한 나머지 데이터로만 계산 — 데이터 누수 방지.

| 컬럼 | 설명 |
|---|---|
| `industry_historical_rate` | 이 업종(소분류)의 과거 평균 폐업률 |
| `dong_historical_rate` | 이 행정동의 과거 평균 폐업률 |
| `dong_industry_historical_rate` | 행정동+업종 조합의 과거 평균 폐업률. 해당 조합의 표본이 30건 미만이면 `industry_historical_rate`로 대체 |

---

## 기타

| 컬럼 | 설명 |
|---|---|
| `transitioned_next` | 다음 스냅샷에서 업종이 바뀌었는지 (0/1) |

---

## 타겟

| 컬럼 | 설명 |
|---|---|
| `is_closed_next` | 다음 스냅샷에서 이 매장이 더 이상 관측되지 않는지 (0/1). 모델이 예측해야 할 값 |

---

## 결측치

생활인구 관련 6개 컬럼(`korean_pop`, `foreign_long_pop`, `foreign_short_pop`, `total_pop_avg`, `foreign_short_ratio`, `tourist_zone_candidate`)에서 약 37,066행(전체의 1.4%) 결측. 소상공인 데이터의 행정동코드와 생활인구 데이터의 행정동코드가 매칭되지 않는 행정동 9개 때문.

`nearest_same_industry_distance_m`도 동일업종이 해당 스냅샷에 1개뿐인 경우 결측 발생.

## 클래스 비율

`is_closed_next` = 1(폐업)인 비율 약 9.2%. 불균형 데이터.
