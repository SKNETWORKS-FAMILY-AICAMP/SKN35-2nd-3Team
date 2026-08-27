# DNN GPU 학습 결과

서울 상권 폐업 예측용 Deep MLP(DNN)를 NVIDIA GPU에서 학습한 결과다.

## 학습 결과

| 항목 | 결과 |
|---|---:|
| GPU | NVIDIA GeForce RTX 3070 |
| CUDA PyTorch | 2.13.0+cu126 |
| Epoch | 7 |
| 입력 피처 | 18개, time-safe |
| Validation PR-AUC | 0.3481 |
| Test PR-AUC | 0.3455 |
| Test ROC-AUC | 0.7194 |
| Test F1 | 0.3610 |
| Precision | 0.3555 |
| Recall | 0.3667 |
| 임계값 | 0.6570 |

`transitioned_next`, 타깃, fold, 매장 식별자와 시간 누수 의심 피처는 학습에서 제외했다. 출력값은 보정된 폐업확률이 아니라 폐업 위험점수다.

## 데이터 분할

- Train: fold 0~2, 1,137,077행
- Validation: fold 3, 375,496행
- Test: fold 4, 377,009행
- 폐업 비율: 10.6485%
- `pos_weight`: 8.404714
- StandardScaler는 train 데이터에만 fit
- validation PR-AUC로 최적 모델 선택
- validation F1로 임계값 선택 후 test에 고정 적용

## 모델 구조

```text
Input(18)
  -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(1 logit)
```

- 손실함수: `BCEWithLogitsLoss(pos_weight=negative/positive)`
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 4096
- Seed: 42
- 최대 epoch: 7
- Early stopping patience: 2
- 학습 파라미터 수: 46,977개

## Test 상세 결과

- Accuracy: 0.8622
- Brier score(미보정): 0.2003
- ECE 10-bin(미보정): 0.3314
- Precision@Top 5%: 0.5039
- Recall@Top 5%: 0.2374
- Precision@Top 10%: 0.3720
- Recall@Top 10%: 0.3504
- Confusion matrix: `[[310389, 26604], [25343, 14673]]`

Accuracy는 전체를 정상으로 예측해도 높게 나오는 불균형 데이터 특성 때문에 단독 성능 판단에 사용하지 않는다. 주 지표는 PR-AUC다.

## SHAP 분석 결과

- Explainer: DeepExplainer
- Background: train 계층표본 256건
- 설명 대상: test 계층표본 1,000건
- 가산성 평균 절대오차: 0.00000012
- 가산성 최대 절대오차: 0.00000119
- 가산성 검증: PASS

전역 중요도 상위 피처:

1. `store_age_months`
2. `industry_group_enc`
3. `snapshot_month_index`
4. `industry_code_enc`
5. `floor_category_enc`

SHAP 값은 모델의 raw logit에 대한 예측 기여도다. 상관계수나 폐업 원인이 아니며, SHAP 값 자체를 확률 퍼센트포인트로 해석하면 안 된다.

## 산출물

기준 RUN:

```text
saved/runs/DNN_20260828_001038_e807fa0_time_safe/
```

- `model/dnn_model.pt`: 학습된 DNN 모델
- `model/config.json`: 모델 설정
- `metrics/history.json`: epoch별 학습 기록
- `metrics/metrics.json`: validation/test 평가 결과
- `predictions/test_predictions.parquet`: 실제 매장과 연결된 test 예측 377,009건
- `plots/`: 학습곡선, PR/ROC, 혼동행렬, calibration 그래프
- `shap/`: 전역 중요도, beeswarm, 매장별 기여도, waterfall, SHAP metadata
- `run_manifest.json`: Git·데이터·Python·CUDA·GPU·피처·분할 실행 정보

## 검증 결과

- DNN 단위 테스트 7개 PASS
- X/y/split/row metadata 행 수 일치
- DNN 입력 NaN·Inf 없음
- 금지 컬럼 입력 제외 확인
- CUDA 실제 텐서 연산 확인
- 새 Python 프로세스에서 모델 재로딩 및 추론 성공
- test 예측 377,009건의 매장 ID 연결 확인
- 평가 지표 NaN·Inf 없음
- SHAP 단위 테스트 3개 PASS
- SHAP 가산성 및 필수 산출물 검증 PASS

Python 버전 파일, `pyproject.toml`, `uv.lock`, 공용 `.venv`는 변경하지 않았다. GPU 학습은 저장소 밖 별도 Python 3.12.13 환경에서 실행했다.
