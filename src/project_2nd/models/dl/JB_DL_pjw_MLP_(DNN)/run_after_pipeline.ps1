$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$PjwPreprocess = Join-Path $RepoRoot "src\project_2nd\preprocessing_dataset_pjw\preprocess_output\preprocess_modeling_dataset_pjw.py"
$ModelingDataset = Join-Path $RepoRoot "data\features\modeling_dataset.csv"
$RefinedDataset = Join-Path $RepoRoot "data\processed\modeling_dataset_refined_pjw.csv"

Push-Location $RepoRoot
try {
    if (-not (Test-Path $ModelingDataset)) {
        throw "modeling_dataset.csv가 없습니다. 먼저 저장소 루트에서 .\run_pipeline.ps1을 실행하세요."
    }

    Write-Host "[1/3] PJW 전처리 실행"
    uv run python $PjwPreprocess
    if ($LASTEXITCODE -ne 0) { throw "PJW 전처리 실패" }
    if (-not (Test-Path $RefinedDataset)) { throw "PJW 정제 데이터셋 생성 확인 실패" }

    Write-Host "[2/3] DNN 학습 배열 및 JSON 생성"
    uv run python (Join-Path $PSScriptRoot "prepare_dnn_dataset.py")
    if ($LASTEXITCODE -ne 0) { throw "DNN 데이터 준비 실패" }

    Write-Host "[3/3] 심층 MLP 학습 (최대 7 epoch)"
    uv run python (Join-Path $PSScriptRoot "train_dnn.py")
    if ($LASTEXITCODE -ne 0) { throw "DNN 학습 실패" }
}
finally {
    Pop-Location
}
