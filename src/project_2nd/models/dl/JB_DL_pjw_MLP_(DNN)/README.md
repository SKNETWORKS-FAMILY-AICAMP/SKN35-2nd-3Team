# PJW 심층 MLP(DNN) 작업 폴더

서울 상권 폐업예측 프로젝트에서 PJW 전처리 결과를 심층 MLP로 학습하기 위한 폴더입니다.
처음 보는 사람도 실행 순서를 따라갈 수 있도록 데이터 준비와 모델 학습을 분리했습니다.

## 이 작업이 필요한 이유

- 머신러닝과 같은 폐업 예측 문제를 딥러닝으로도 풀어 성능을 비교합니다.
- 구조는 벤치마크 문서의 주력 후보인 `입력 → 256 → 128 → 64 → 출력`을 사용합니다.
- 폐업 표본이 적은 불균형 데이터이므로 `Weighted BCEWithLogitsLoss`를 사용합니다.
- 과적합과 실행 비용을 제한하기 위해 최대 epoch를 **7회**로 강제합니다.

## 전체 실행 순서

저장소 루트에서 아래 순서로 실행합니다.

```powershell
.\run_pipeline.ps1
uv run python .\src\project_2nd\preprocessing_dataset_pjw\preprocess_output\preprocess_modeling_dataset_pjw.py
& '.\src\project_2nd\models\dl\JB_DL_pjw_MLP_(DNN)\run_after_pipeline.ps1'
```

마지막 명령은 다음 두 단계를 차례로 실행합니다.

1. `prepare_dnn_dataset.py`: PJW 정제 CSV를 숫자 배열과 전처리 JSON으로 변환
2. `train_dnn.py`: 심층 MLP를 최대 7 epoch 학습하고 평가 결과 저장

## 입력과 출력

입력:

- `data/processed/modeling_dataset_refined_pjw.csv`

출력 폴더:

- `artifacts/data/X.npy`: 표준화된 입력 피처
- `artifacts/data/y.npy`: 폐업 여부 정답(0/1)
- `artifacts/data/split.npy`: train/validation/test 구분
- `artifacts/data/preprocessing_metadata.json`: 피처 순서, 범주 매핑, 결측 대체값, 스케일링 값
- `artifacts/model/dnn_model.pt`: 학습된 PyTorch 모델
- `artifacts/model/history.json`: epoch별 학습 기록
- `artifacts/model/metrics.json`: 검증·테스트 평가지표와 최종 임계값

행 단위 전체 데이터를 JSON으로 저장하면 파일이 지나치게 커지고 학습도 느려집니다. 따라서 실제 학습 데이터는 `.npy`로 저장하고, 재현과 배포에 필요한 전처리 규칙만 JSON으로 저장합니다.

## 데이터 분할

기존 파이프라인의 `fold`를 그대로 사용해 같은 매장이 여러 데이터 영역에 섞이는 것을 막습니다.

- 학습: fold 0, 1, 2
- 검증 및 임계값 결정: fold 3
- 최종 테스트: fold 4

테스트 데이터는 모델 선택이나 임계값 결정에 사용하지 않습니다.

## 평가 지표

- 주 지표: PR-AUC
- 보조 지표: ROC-AUC, F1, Precision, Recall, Accuracy
- 분류 임계값: 검증 데이터에서 F1이 가장 높은 값으로 결정

## 주의사항

- `config.json`의 `epochs`는 1~7만 허용합니다.
- 원본 CSV와 팀 공용 파이프라인 파일은 수정하지 않습니다.
- `artifacts/`는 실행 결과이므로 Git에 올리지 않습니다.
- 현재 프로젝트의 PyTorch는 팀 환경 통일을 위해 CPU 빌드로 고정되어 있습니다.
