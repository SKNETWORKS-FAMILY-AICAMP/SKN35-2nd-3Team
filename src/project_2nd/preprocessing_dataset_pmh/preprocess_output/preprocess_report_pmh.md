
# modeling_dataset.csv 전처리 리포트 (pmh)

- 입력 행 수: 1,889,582
- data/raw는 읽지 않음 (원본에 gu_name/생활인구 결측이 이미 없어 조회 불필요)

## 1. 원본 결측치 현황

- 전체 결측치: 0건
- population_is_proxied=True 비율: 1.60% (생활인구가 대체값으로 채워진 행, 원본 파이프라인 단계에서 이미 처리됨)

## 2. nearest_same_industry_distance_m 결측 처리

- 결측 0행을 9999.0(동일업종 없음을 의미하는 상수)로 대체

## 3. bool 컬럼 정수 변환

- transitioned_next, tourist_zone_candidate, population_is_proxied -> 0/1 정수로 변환

## 4. 범주형 라벨 인코딩

- industry_dae_code -> industry_dae_code_enc (카디널리티=7)
- industry_group -> industry_group_enc (카디널리티=7)
- industry_jung_code -> industry_jung_code_enc (카디널리티=53)
- industry_jung_name -> industry_jung_name_enc (카디널리티=53)
- industry_code -> industry_code_enc (카디널리티=192)
- industry_name -> industry_name_enc (카디널리티=192)
- gu_name -> gu_name_enc (카디널리티=25)
- dong_code -> dong_code_enc (카디널리티=428)
- floor_category -> floor_category_enc (카디널리티=5)

인코딩 매핑 저장: src/project_2nd/preprocessing_dataset_pmh/preprocess_output/encoders_pmh.json (서빙 시 동일 매핑 재사용 가능, 사전에 없는 값은 -1)

## 결과

- 출력 행 수: 1,889,582 (입력과 동일, 행 제거 없음)
- 출력 컬럼 수: 43 (원본 34 + 인코딩 9)
- 최종 결측치: 0건
- 저장 위치: data/processed/modeling_dataset_preprocessed_pmh.csv