# models·predictions SHAP 적재 결과

## 처리 순서

1. 최종 DNN의 기존 SHAP 1,000건을 위험점수 순으로 정렬
2. 상위 예측 2건에서 절대 SHAP 값 기준 상위 5개 피처 추출
3. `models` 1건과 `predictions` 2건을 `schema.sql` 컬럼 형식으로 변환
4. 로컬 `stores.csv`, `industries.csv`로 외래키 존재 여부 검증
5. 실제 TiDB 스키마를 읽기 전용으로 확인하고 컬럼명·자료형 검증
6. DB 적재 없이 실행 가능한 SQL·CSV 생성

외래키 때문에 실제 SQL 실행 순서는 `models → predictions`이지만, predictions에 넣을 SHAP 데이터는 먼저 준비했다.

## models 결과 1건

| model_id | 유형 | 버전 | Accuracy | Precision | Recall | F1 | ROC-AUC | 운영 모델 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `DNN_20260828_094006_7e84a4a_pjw_official` | DL | `2026.08.28-e100p5` | 0.88525 | 0.44918 | 0.35856 | 0.39878 | 0.74577 | FALSE |

## predictions 결과 2건

`score`는 보정된 폐업 확률이 아니라 DNN의 미보정 폐업 위험점수다.

| 번호 | store_id | industry_code | score | 실제 타깃 | SHAP 피처 |
|---:|---|---|---:|---:|---:|
| 1 | `MA0106202201A0025935` | `G21202` | 0.99386 | 1 | 5개 |
| 2 | `MA0106202509A1423510` | `I21001` | 0.99171 | 1 | 5개 |

## SHAP 상세

### 예측 1: `MA0106202201A0025935`

| 순위 | 피처 | 피처값 | SHAP 값 |
|---:|---|---:|---:|
| 1 | `dong_industry_historical_rate` | 0.67188 | +2.269207 |
| 2 | `store_age_months` | 6.00000 | +0.655202 |
| 3 | `is_left_censored_age` | 0.00000 | +0.638668 |
| 4 | `snapshot_month_index` | 18.00000 | +0.593110 |
| 5 | `nearest_same_industry_distance_m` | 0.00000 | +0.490359 |
### 예측 2: `MA0106202509A1423510`

| 순위 | 피처 | 피처값 | SHAP 값 |
|---:|---|---:|---:|
| 1 | `store_age_months` | 0.00000 | +2.302361 |
| 2 | `snapshot_month_index` | 24.00000 | +1.669854 |
| 3 | `is_left_censored_age` | 0.00000 | -1.024327 |
| 4 | `floor_category_enc` | 2.00000 | +0.679179 |
| 5 | `industry_group_enc` | 6.00000 | +0.400522 |

## 검증 결과

| 항목 | 결과 |
|---|---:|
| models 행 | 1 |
| predictions 행 | 2 |
| SHAP 비어 있지 않은 행 | 2 |
| 예측별 SHAP 피처 수 | [5, 5] |
| models 값·타입 일치 | True |
| predictions 값·타입 일치 | True |
| prediction_id 자동 증가로 입력 제외 | True |
| 로컬 stores FK 확인 | True |
| 로컬 industries FK 확인 | True |
| TiDB INSERT 실행 | False |

## 실제 TiDB 컬럼 타입

### models

| 컬럼 | 타입 |
|---|---|
| `model_id` | `varchar(50)` |
| `model_name` | `varchar(100)` |
| `version` | `varchar(20)` |
| `model_type` | `enum('ML','DL')` |
| `accuracy` | `decimal(6,5)` |
| `precision_score` | `decimal(6,5)` |
| `recall_score` | `decimal(6,5)` |
| `f1_score` | `decimal(6,5)` |
| `roc_auc` | `decimal(6,5)` |
| `trained_at` | `datetime` |
| `is_production` | `tinyint(1)` |

### predictions

| 컬럼 | 타입 |
|---|---|
| `prediction_id` | `bigint AUTO_INCREMENT` |
| `model_id` | `varchar(50)` |
| `user_id` | `varchar(30) NULL` |
| `query_type` | `enum('existing_store','new_location')` |
| `store_id` | `varchar(30) NULL` |
| `query_lat` | `decimal(10,7) NULL` |
| `query_lng` | `decimal(10,7) NULL` |
| `industry_code` | `varchar(20)` |
| `score` | `decimal(6,5)` |
| `shap_top_features` | `json NULL` |
| `created_at` | `datetime` |

`prediction_id`는 `BIGINT AUTO_INCREMENT`이므로 SQL과 CSV에서 입력하지 않는다.

## 생성 파일

- `models_result_1row.csv`
- `predictions_shap_result_2rows.csv`
- `models_predictions_shap_seed.sql`
- `schema_validation.json`
