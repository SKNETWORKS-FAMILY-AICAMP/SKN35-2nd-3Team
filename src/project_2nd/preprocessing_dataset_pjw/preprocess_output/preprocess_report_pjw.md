# modeling_dataset_refined.csv — 전처리 v2 작업 기록

원본(`modeling_dataset.csv`, 팀원 공유분)은 건드리지 않고, 별도 스크립트(`preprocess_modeling_dataset_pjw.py`)로 정제본(`data/processed/modeling_dataset_refined_pjw.csv`)을 생성한다. 팀원 파이프라인 코드(`db/`, `features/`, `models/`, `preprocessing_dataset/`)는 미변경.

⚠️ **원본 파일 버전 주의**: 공유받은 `modeling_dataset.csv`가 실제로는 1,889,582행인데, `modeling_설명.md` 문서엔 2,591,877행이라고 적혀있음(약 27% 차이, 폐업 비율도 문서 9.2% vs 실제 10.6%). 파일-문서 버전이 안 맞는 것으로 보임 — 계속 작업하기 전에 팀원분께 최신 버전 확인 필요.

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

## 🔑 발견: 좌표 중복 클러스터 — `coord_cluster_size`가 실제 성능을 올린 핵심 피처(수정 후에도 소폭 유효)

서울 범위 밖 좌표는 0건으로 깨끗했지만, 정확히 같은 (lng, lat)을 공유하는 서로 다른 `store_id`가 최대 883개까지 나옴(전체 행의 83.5%가 2개 이상과 좌표 공유, 18.7%는 20개 이상, 5.9%는 100개 이상). 어떤 행정동은 전체 매장의 62.5%가 좌표 하나에 몰려있음.

**원인 미확정**: (1) 가락시장류 대형 상가건물이 실제로 있어서 건물 단위 지오코딩이 그렇게 잡혔을 수도 있고, (2) 개별 주소 지오코딩이 실패해 행정동 중심좌표로 fallback됐을 수도 있음 — 확정할 방법 없음.

**중요한 건**: 원인이 뭐든 이 클러스터 크기 자체가 유의미한 예측 신호였다는 것. 좌표를 고칠 순 없지만 "이 좌표를 공유하는 store_id 수"를 그대로 피처로 남겼더니 위 표처럼 성능이 뚜렷하게 개선됨. `same_industry_count_300m`/`nearest_same_industry_distance_m` 같은 기존 공간 피처는 이런 좌표에서 여전히 왜곡돼 있을 수 있다는 점은 유의할 것(같은 좌표에 있는 매장끼리는 거리 0으로 계산됨).

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
- **`dong_historical_rate`**: 샘플 동(11680640) 5개 fold 전부 저장된 값(~0.083)이 "fold 제외 재계산"(~0.118)과 일관되게 약 30% 차이. 랜덤 오차가 아니라 매 fold 같은 방향·비슷한 폭이라 **다른 정의(다른 데이터 범위 또는 스무딩)로 계산됐을 가능성**. 원인 미확정 — 팀원분께 정확한 계산 로직 문의 필요

## 추가로 검토하면 좋을 것 (미착수)

- `coord_cluster_size`가 왜 성능을 올렸는지 원인 규명 (대형상가 vs 지오코딩 fallback) — 가능하면 `store_panel_master.csv`의 주소 텍스트로 대형 상가/시장 여부 역추적
- `dong_historical_rate` 계산 로직 불일치 원인 확인 (팀원 문의 필요)
