
# modeling_dataset.csv 전처리 리포트 (pmh)

- 입력 행 수: 1,889,582

## 1. gu_name 결측 복구

- 복구 전 gu_name 결측: 30,181행
- 복구 후 gu_name 결측: 0행 (data/raw 6개 스냅샷 전체 조회로 복구, data/raw는 읽기만 함)

## 2. 생활인구 결측 대체

- 생활인구 5개 수치 컬럼 결측(대체 전): 30,181행 -> gu_name 그룹 중앙값으로 대체
- tourist_zone_candidate 결측(대체 전): 30,181행 -> False(최빈값)로 대체

## 3. nearest_same_industry_distance_m 결측 처리

- 결측 0행을 9999.0(동일업종 없음을 의미하는 상수)로 대체

## 4. bool 컬럼 정수 변환

- transitioned_next, tourist_zone_candidate -> 0/1 정수로 변환

## 5. 범주형 라벨 인코딩

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
- 출력 컬럼 수: 42 (원본 33 + 인코딩 9)
- 최종 결측치: 0건
- 저장 위치: data/processed/modeling_dataset_preprocessed_pmh.csv