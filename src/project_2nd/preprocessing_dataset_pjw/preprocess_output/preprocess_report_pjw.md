# modeling_dataset_refined_pjw.csv — 전처리 작업 기록

원본(`data/features/modeling_dataset.csv`)은 건드리지 않고, 별도 스크립트(`preprocess_modeling_dataset_pjw.py`)로 정제본(`data/processed/modeling_dataset_refined_pjw.csv`)을 생성한다. 팀원 파이프라인 코드(`db/`, `features/`, `models/`, `preprocessing_dataset/`)는 미변경.

---

## 🆕 v4 (2026-08-26): 팀원분 파이프라인 재실행 + `gu_name` 버그 발견/수정

팀원분이 `floor_category`(층정보)와 `coord_cluster_size`(DBSCAN 기반, 저희 exact-match 버전보다 정교함)를 공식 파이프라인에 반영. 공유 zip엔 대용량 산출물(`modeling_dataset.csv`)이 안 들어있어서, **`run_pipeline.sh`로 직접 로컬 재실행**해서 새 원본을 만들었다(uv 환경, 8단계, 총 555초 소요, 무사 완주).

**부수 확인**: 예전에 "파일-문서 버전 불일치"라고 여겼던 것(1,889,582행 vs 문서 2,591,877행)의 진짜 원인이 파이프라인 로그로 확인됨 — **버전 차이가 아니라 파이프라인 단계 차이**였음. `build_modeling_dataset.py`가 스코프 제외(과학·기술/부동산/시설관리·임대) 적용 전엔 2,591,877행, 적용 후 1,889,582행. 문서는 필터링 전 숫자를 적어놓은 것.

### 🐛 새로 발견한 버그: `gu_name` 결측 30,181행

기존 `data/features/modeling_dataset.csv`를 감사하다가, `gu_name`(및 그로 인해 연쇄적으로 `korean_pop` 등 생활인구 5개 컬럼)이 **12개 `dong_code`에서 결측**인 걸 발견. `population_features.csv`(행정동 424개)에 이 12개 동이 아예 없어서(팀원분 문서의 "9개 동 매칭 실패"와 같은 종류 이슈, 새 파이프라인에서 12개로 확인됨) 조인 시 `gu_name`부터 비어버리는 구조였음 — 즉 원래 있던 "생활인구 결측 1.6%"의 근본 원인이 인구 데이터가 아니라 `gu_name` 자체였다는 걸 이번에 알게 됨.

**수정**: 행정동코드 앞 5자리(구 코드)가 같은 다른 행의 `gu_name`으로 매핑해보니 12개 전부 모호함 없이 하나의 구로 정확히 복원됨(강북구 6 / 강동구 2 / 동대문구 2 / 구로구 1 / 강남구 1). 이 매핑으로 `gu_name`을 먼저 채운 뒤, 인구 결측 대체(구 평균)도 정상 작동하게 함 — 정제본 기준 남은 결측치 0건.

### 실험 A: 생활인구 결측 처리 방식 비교 — gu 평균 대체 vs 결측 유지

`gu_name` 버그를 고치고 나니 생활인구 대체(`population_imputed` 플래그 포함)가 실제로 효과가 있는지 5-fold로 직접 비교해봤다(`exp_a_pop_impute.py`).

| | gu 평균 대체 | 결측 유지(native NaN) |
|---|---|---|
| ROC-AUC | 0.747610 ± 0.0007 | 0.747853 ± 0.0009 |
| F1 | 0.281365 | 0.281517 |

통계적으로 동률(표준편차 범위 안), 오히려 결측 유지 쪽이 근소 우세. LightGBM이 결측을 자체적으로 잘 처리하므로 **대체 로직과 `population_imputed` 플래그를 최종본에서 제거**하고 결측은 그대로 둠 — 불필요한 로직을 줄이는 방향으로 단순화.

### 실험 B: store_age_months 좌측절단 플래그 + 트렌드 키워드 이진 플래그

팀원분 한계점 문서에 명시된 두 가지 이슈를 플래그로 보완할 수 있는지 5-fold로 검증(`exp_b_age_kw.py`).

- **`is_left_censored_age`**: `store_age_months`는 첫 스냅샷(202312) 기준으로 계산되는데, 그 시점에
  이미 존재하던 매장(전체의 78.85%)은 실제 개업일을 몰라 나이가 과소추정됨. `first_seen_snapshot`
  이 202312인지를 플래그로 노출.
- **`is_trend_keyword_match`**: 기존 `keyword_growth_score`는 growth_rate가 0 이하인 키워드가
  매칭돼도 값이 0으로 남아 "매칭 안 됨"과 구분이 안 됨(실제 4,839행이 이 케이스). 매칭 자체를
  이진 플래그로 분리.

| | base | +censor_flag | +kw_flag | +both |
|---|---|---|---|---|
| ROC-AUC | 0.747853 | 0.748151 (+0.0003) | 0.748002 (+0.00015) | 0.748261 (+0.0004) |

절대 개선폭은 작지만(표준편차 ~0.0008 대비 절반 수준), `coord_cluster_size`를 채택했을 때와 같은
기준으로 판단 — **5개 fold 전부 일관되게 같은 방향으로 개선**돼서 노이즈가 아니라 실제 신호로
보고 두 플래그 모두 채택(`+both`가 가장 좋음).

### 최종 검증 (새 데이터, LightGBM 5-fold)

| | 새 원본(33컬럼) | 저희 정제본(36컬럼, dong_code 제거 + 생활인구 결측 유지 + censor/kw 플래그 추가) |
|---|---|---|
| ROC-AUC | 0.747896 ± 0.0009 | **0.748261 ± 0.0007 (+0.00037)** |
| F1 | 0.280995 | 0.282078 |
| Accuracy | 0.906171 | 0.906311 |

`gu_name` 결측 버그 수정(데이터 완전성 개선, 30,181행)에 더해, censor/kw 플래그로 원본 대비
5-fold 전체 일관된 소폭 개선을 확인 — 두 결과 모두 이번 라운드의 실질적 성과.

`dong_code`는 이미 정제 단계에서 제거하고 있어(`DROP_COLS`) 정제본 자체에 컬럼이 없음 — 있음/없음 ablation은 더 이상 의미가 없고, 예전 재확인(있음 0.747755 vs 없음 0.747610, 동률)으로 "제거해도 무방" 결론은 이미 유효하게 확정돼 있음.

**v4에서 저희 `coord_cluster_size` 자체 구현은 제거**했음(팀원분의 DBSCAN 버전으로 완전 대체됨, 아래 v2/v3 이력은 참고용으로만 남겨둠).

---

## v2/v3 이력 (구 `modeling_dataset.csv`, 31컬럼 — floor_category/공식 coord_cluster_size 반영 전) — 참고용

아래 내용은 팀원분이 `floor_category`를 추가하고 `coord_cluster_size`를 공식화하기 전, 구버전 데이터 기준으로 작업했던 기록이다. 결론(어떤 컬럼을 왜 남기고 뺐는지, `coord_cluster_size` 누수를 어떻게 찾고 고쳤는지)은 v4에도 그대로 이어지므로 남겨둔다.

✅ **원본 파일 버전 확인 완료**: 공유받은 `modeling_dataset.csv`(1,889,582행)가 맞는 최신 버전임을 팀원분께 확인함. `modeling_설명.md` 문서에 적힌 2,591,877행 쪽이 오래된 수치. (→ v4에서 진짜 원인 규명됨: 스코프 필터링 전/후 차이였음)

---

## 적용한 전처리 (v2)

1. `total_pop_avg` 제거 — `korean_pop`+`foreign_long_pop`+`foreign_short_pop`의 단순 합이라 순수 중복
2. `is_mass_reclass_window` 플래그 추가 — `snapshot_date==202406`이면 1
3. 생활인구 결측(1.6%, 30,181행)을 같은 `gu_name` 평균/최빈값으로 대체
4. `population_imputed` 플래그 추가 — 3번에서 대체된 행 표시
5. 파생 피처 3종 추가 (기존 컬럼 조합만 사용):
   - `industry_specialization_300m` = `same_industry_count_300m` / `total_count_300m`
   - `competition_per_capita_300m` = `same_industry_count_300m` / (korean_pop+foreign_long_pop+foreign_short_pop)
   - `dong_industry_count_growth` = `(dong_code, industry_code)` 그룹 내 `dong_industry_count`의 전기 대비 증감률
6. `coord_cluster_size` 추가 — 정확히 같은 (lng, lat)을 공유하는 유니크 store_id 수. 아래 참고
7. `dong_code` 제거 — 5-fold ablation에서 있음/없음 성능 사실상 동일(뒤쪽 표 참고), `dong_historical_rate`/`dong_industry_historical_rate`/`gu_name`/`coord_cluster_size`가 지역 정보를 충분히 대체

### 시도했다가 되돌린 것

- **`industry_code`류(`industry_dae_code`/`industry_code`/`industry_name`/`industry_jung_code`) 제거** — `industry_historical_rate`와 그룹평균 상관 0.9999라 중복이라고 판단했었으나, LightGBM 5-fold 검증에서 ROC-AUC가 전 fold 일관되게 하락(-0.002)해서 되돌림. 그룹평균 상관은 높아도, 트리 모델이 `industry_code`를 다른 피처와 조합해 만드는 세밀한 분기 정보는 스무딩된 rate 하나로 대체가 안 됐던 것으로 보임.

---

## 성능 검증 결과

### LightGBM, 5-fold 평균 (fold 컬럼 기준 GroupKFold, 최종본 — `coord_cluster_size` 시점 누수 수정 반영)

| | original | refined |
|---|---|---|
| ROC-AUC | 0.745082 ± 0.0009 | **0.745632 ± 0.0009 (+0.00055)** |
| F1 | 0.262997 | 0.265138 |
| Accuracy | 0.905215 | 0.905354 |

⚠️ **아래 `coord_cluster_size` 최초 버전엔 시점 누수가 있었고, 이 표는 수정 후 재검증한 최종 수치.** 처음엔 +0.0033이었는데 수정 후 +0.00055로 줄어듦 — 개선폭의 약 5/6이 누수 때문이었음. 그래도 5개 fold 전부 아주 작게나마 일관되게 개선돼서(노이즈는 아님) 진짜 신호가 조금은 남아있음. `industry_specialization_300m`/`dong_industry_count_growth`는 top-15에 들지만 트리가 이미 비슷한 분기를 만들 수 있어 이쪽은 추가 이득이 거의 없었음.

### `dong_code` ablation (정제본 기준, 있음 vs 없음)

| | with_dong_code | without_dong_code |
|---|---|---|
| ROC-AUC | 0.748361 | 0.748395 |
| F1 | 0.264521 | 0.265132 |

⚠️ 이 ablation은 `coord_cluster_size` 누수 수정 **전**에 돌린 결과라 절대 수치는 위 최종 표와 안 맞음. 다만 두 갈래(with/without) 모두 같은 버전의 `coord_cluster_size`를 공유한 채 비교한 거라 "dong_code는 있으나 없으나 차이 없다"는 상대적 결론 자체는 유효할 것으로 판단 — `dong_code`(428개 카테고리)를 빼도 손실 없어서 최종본에서 제거함.

## ✅ 처리 완료: `coord_cluster_size` 시점 누수 수정

최초 구현은 `groupby(["lng", "lat"])`만 써서 **전체 5개 스냅샷을 합쳐서** 좌표당 유니크 store_id 수를 계산했다. 이러면 2023-12 시점 행인데도 그 좌표에 2025-12까지 나중에 새로 생긴 매장까지 카운트에 섞여 들어가서, 과거 시점 행에 미래 정보가 새어들어가는 구조였다(팀원분이 "데이터가 하나로 합쳐진 형태니 피처 조심하라"고 짚어준 지점).

**확인**: 전체기간 합산 버전과 `snapshot_date`까지 넣어 스냅샷별로 계산한 버전을 비교하니 **전체 행의 41.9%에서 값이 달라짐**(상관계수는 0.995로 높아서 완전히 다른 신호는 아니었음).

**수정**: `groupby(["snapshot_date", "lng", "lat"])`로 변경. 재검증 결과 ROC-AUC 개선폭이 +0.0033 → +0.00055로 줄어듦(위 표 참고) — 원래 개선의 대부분이 누수 때문이었던 것으로 확인됨.

## ✅ 원인 규명 완료: 좌표 중복 클러스터 — `coord_cluster_size`가 실제 성능을 올린 핵심 피처(수정 후에도 소폭 유효)

서울 범위 밖 좌표는 0건으로 깨끗했지만, 정확히 같은 (lng, lat)을 공유하는 서로 다른 `store_id`가 최대 883개까지 나옴(전체 행의 83.5%가 2개 이상과 좌표 공유, 18.7%는 20개 이상, 5.9%는 100개 이상). 어떤 행정동은 전체 매장의 62.5%가 좌표 하나에 몰려있음.

**원인 확정 — 지오코딩 오류가 아니라 실제 대형 복합상가**: 원본 스냅샷 CSV(`2nd_raw/seoul_YYYYMM.csv`)의 지번주소/도로명주소로 최대 두 클러스터를 직접 조회해서 확인.
- 가장 큰 클러스터(883개, 송파구): `서울특별시 송파구 충민로 66` — 상호명에 "현대시티몰"이 그대로 포함됨(문정동 현대시티몰). 신발/의류/가전/학원/음식점 등 완전히 다른 업종 883개가 이 건물 하나에 입점
- 행정동 최다 집중 클러스터(135개, 62.5%): `서울특별시 송파구 올림픽로 435` — 상호명에 "파크리오"가 포함됨(잠실 파크리오 아파트 단지 상가)

두 사례 다 실제 대형 복합상가/아파트단지 상가였음 — 지오코딩 fallback 가설은 기각. `coord_cluster_size`는 **"이 매장이 대형 복합상가에 입점해 있는지"를 나타내는 신뢰할 수 있는 신호**로 확인됨. 다만 `same_industry_count_300m`/`nearest_same_industry_distance_m` 같은 기존 공간 피처는 이런 좌표에서 여전히 왜곡돼 있을 수 있다는 점은 유의할 것(같은 좌표에 있는 매장끼리는 거리 0으로 계산됨).

### LogisticRegression, fold 4만 (saga, class_weight=balanced, 원-핫 인코딩)

| | original | refined |
|---|---|---|
| ROC-AUC | 0.6967 | **0.7002** (+0.0035) |
| F1 | 0.2808 | 0.2835 |

LightGBM 대비 훨씬 뚜렷한 개선. 트리와 달리 로지스틱회귀는 상호작용을 스스로 못 만들기 때문에, 미리 만들어준 비율 피처가 더 유효한 것으로 보임. **fold 4 하나만 확인한 결과라 5-fold 전체 검증은 아직 안 함** — 실제 모델 비교(로지스틱회귀/RF/MLP)는 팀원분 담당이라 여기서는 "가능성 확인"까지만 하고 넘김.

---

## ✅ 처리 완료: `transitioned_next` 데이터 누수 제거

`modeling_dataset.csv`에 피처로 포함된 `transitioned_next`("다음 스냅샷에서 업종이 바뀌었는지")는 타깃 `is_closed_next`("다음 스냅샷에서 사라졌는지")와 **같은 미래 시점(다음 스냅샷) 정보**다. 실제로 둘은 상호배타적으로 나온다(교차표 확인 결과 둘 다 1인 행 0건 — `transitioned_next=1`이면 `is_closed_next=0`이 자동 확정됨).

**문제**: 실제 예측 시점(오늘)엔 "다음 스냅샷에 업종이 바뀔지"를 알 수 없다 — 이건 타깃과 정확히 같은 종류의 미래 정보라 look-ahead/타깃 누수에 해당한다.

**영향 규모**: `transitioned_next=1`인 행은 전체의 0.94%뿐이라 지금까지의 성능 비교(원본 vs 정제본 둘 다 이 컬럼을 동일하게 포함한 채 비교했음)에 미치는 영향은 작았을 것으로 추정되지만, 실제 모델 제출 전엔 **`transitioned_next`를 피처 목록에서 제외**해야 한다.

**처리 결과**: `preprocess_modeling_dataset_pjw.py`에서 `transitioned_next`를 제거한 뒤 5-fold로 재검증. ROC-AUC 0.745082 → 0.744991(-0.00009, 노이즈 수준), F1은 오히려 소폭 상승(0.262997 → 0.263493). 예상대로 영향 규모가 작아서(전체의 0.94%) 성능 손실 없이 누수만 제거됨.

---

## fold-safe `*_historical_rate` 신뢰 검증 결과

표본 많은 업종 3개 × fold 5개로 "해당 fold 제외하고 재계산한 값"과 저장된 `industry_historical_rate`를 대조.

- **`industry_historical_rate`**: 15개 조합 전부 소수점까지 정확히 일치 — fold-safe하게 정확히 구현돼 있음 확인
- **`dong_historical_rate`**: 샘플 동(11680640) 5개 fold 전부 저장된 값(~0.083)이 "modeling_dataset.csv 기준 fold 제외 재계산"(~0.118)과 일관되게 약 30% 차이. ✅ **원인 규명 완료(버그 아님)**: `build_modeling_dataset.py`에서 `dong_historical_rate`는 스코프 필터링(과학·기술/부동산/시설관리·임대 제외) **적용 이전**에 계산됨. 이 3개 제외 업종군은 폐업률이 훨씬 낮은데(이 동 기준 제외 업종군 31,988행 폐업률 4.61% vs 최종 포함 업종군 11.86%), 그 낮은 폐업률의 행들이 계산 당시 분모에 섞여 있어서 저장값이 낮게 나옴. 우리가 가진 `modeling_dataset.csv`는 이미 그 행들이 빠진 상태라 재계산하면 자연히 높게 나오는 것 — 파이프라인 코드를 그대로 재현해서 계산해보니 fold별 값(0.0833/0.0830/0.0835/0.0838/0.0825)이 저장값과 소수점까지 정확히 일치함을 확인. 데이터 누수도 계산 버그도 아니고 fold-safe하게 의도대로 동작 중.

## 추가로 검토하면 좋을 것 (미착수)

- `gu_name` 결측 12개 동 문제를 우리 쪽에서만 patch하지 말고, 근본적으로는 `population_features.csv`(팀원분 파이프라인)에 이 12개 동을 채워넣는 게 맞는 방향 — 팀원분과 공유해서 상류에서 고칠지 논의 필요
- (선택) `coord_cluster_size`를 대형상가 "규모 구간"(예: 소/중/대형)으로 나눠서 범주형으로도 실험해볼 수 있음
