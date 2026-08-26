
# modeling_dataset_preprocessed_pmh.csv 검증 리포트

- 원본(modeling_dataset.csv): 1,889,582행 x 33열
- 전처리 결과(modeling_dataset_preprocessed_pmh.csv): 1,889,582행 x 42열

## 1. 기본 형태

- [PASS] 행 수 원본과 일치 (필터링 없이 결측치 대체만 수행됐는지): 원본 1,889,582행 vs 전처리 1,889,582행
- [PASS] 결측치 0건: 0건

## 2. 인코딩(_enc) 컬럼 무결성

- [PASS] industry_dae_code_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] industry_group_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] industry_jung_code_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] industry_jung_name_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] industry_code_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] industry_name_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] gu_name_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] dong_code_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건
- [PASS] floor_category_enc: 원본값 1개 -> 코드 1개로만 매핑: 코드가 2개 이상으로 흩어진 원본값 0건

## 3. encoders_pmh.json 매핑과 실제 _enc 값 일치

- [PASS] industry_dae_code_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 7개 중)
- [PASS] industry_group_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 7개 중)
- [PASS] industry_jung_code_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 53개 중)
- [PASS] industry_jung_name_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 53개 중)
- [PASS] industry_code_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 192개 중)
- [PASS] industry_name_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 192개 중)
- [PASS] gu_name_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 25개 중)
- [PASS] dong_code_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 428개 중)
- [PASS] floor_category_enc: encoders_pmh.json <-> 실제 값: 불일치 0건 (전체 카테고리 5개 중)

## 4. 타겟(is_closed_next) 비율 (전처리 전후 비교)

- [PASS] 폐업 비율이 전처리 전후로 동일: 원본 0.106485 vs 전처리 0.106485

## 5. nearest_same_industry_distance_m 결측 대체값(9999.0) 개수

- [PASS] 대체값 개수가 원본 결측 건수와 일치: 원본 결측 0건 vs 대체값 0건

## 6. gu_name 복구 결과

- [PASS] UNKNOWN으로 남은 gu_name 없음: 0건

## 7. 수치형 컬럼 이상값 (inf / 음수)

- [PASS] korean_pop: inf/음수 없음: inf 0건, 음수 0건
- [PASS] foreign_long_pop: inf/음수 없음: inf 0건, 음수 0건
- [PASS] foreign_short_pop: inf/음수 없음: inf 0건, 음수 0건
- [PASS] total_pop_avg: inf/음수 없음: inf 0건, 음수 0건
- [PASS] foreign_short_ratio: inf/음수 없음: inf 0건, 음수 0건
- [PASS] same_industry_count_300m: inf/음수 없음: inf 0건, 음수 0건
- [PASS] total_count_300m: inf/음수 없음: inf 0건, 음수 0건
- [PASS] nearest_same_industry_distance_m: inf/음수 없음: inf 0건, 음수 0건
- [PASS] dong_industry_count: inf/음수 없음: inf 0건, 음수 0건
- [PASS] coord_cluster_size: inf/음수 없음: inf 0건, 음수 0건
- [PASS] store_age_months: inf/음수 없음: inf 0건, 음수 0건
- [PASS] keyword_growth_score: inf/음수 없음: inf 0건, 음수 0건

## 결과

- 전체 판정: PASS