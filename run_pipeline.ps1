# 서울 상권 폐업예측 — 데이터 파이프라인 전체를 한 번에 실행 (Windows PowerShell용)
#
# 실행 위치: 저장소 루트 (SKN35-2nd-3Team\) 에서
#   .\run_pipeline.ps1
#
# 사전 조건: data\raw\ 안에 원본 CSV가 있어야 함
#   - 소상공인 6개 스냅샷: seoul_YYYYMM.csv
#   - 생활인구 3종: local_pop.csv / longf_pop.csv / tempf_pop.csv
#
# 중간에 하나라도 실패하면 그 자리에서 멈춥니다.

$ErrorActionPreference = "Stop"

$PKG = "src\project_2nd"
$stepNum = 0
$totalStart = Get-Date

function Run-Step {
    param(
        [string]$Desc,
        [string]$Script
    )
    $script:stepNum++
    Write-Host ""
    Write-Host "==> [$script:stepNum/8] $Desc"
    Write-Host "    $Script"
    $start = Get-Date

    uv run python $Script
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "!! [$script:stepNum] '$Desc' 단계 실패. 여기서 중단합니다."
        Write-Host "   위 에러 메시지 확인 후 원인 해결하고 다시 실행하세요."
        exit 1
    }

    $end = Get-Date
    $elapsed = [int]($end - $start).TotalSeconds
    Write-Host "    완료 ($Desc, ${elapsed}초)"
}

if (-not (Test-Path "data\raw") -or -not (Get-ChildItem "data\raw" -ErrorAction SilentlyContinue)) {
    Write-Host "data\raw\ 에 원본 CSV가 없습니다. 먼저 데이터를 넣어주세요."
    exit 1
}

Write-Host "저장소 루트에서 uv 환경 동기화 중..."
uv sync

Run-Step "폐업 라벨·업종전환 이력 생성"        "$PKG\db\etl\build_closure_transitions.py"
Run-Step "스냅샷별 매장 테이블 생성"           "$PKG\db\etl\build_store_snapshots.py"
Run-Step "공간 밀도 피처 (BallTree/Haversine)"  "$PKG\features\spatial\build_spatial_features.py"
Run-Step "생활인구 피처"                       "$PKG\features\spatial\build_population_features.py"
Run-Step "업종 그룹핑"                         "$PKG\features\industry_grouping\build_industries.py"
Run-Step "트렌드 키워드 탐지"                  "$PKG\features\trend_keywords\build_trend_keywords.py"
Run-Step "업종 전환 생존율 피처"               "$PKG\features\survival_transition\build_survival_stats.py"
Run-Step "최종 모델링 데이터셋 조립"           "$PKG\models\ml\build_modeling_dataset.py"

$totalEnd = Get-Date
$totalElapsed = [int]($totalEnd - $totalStart).TotalSeconds
Write-Host ""
Write-Host "=================================================="
Write-Host "전체 파이프라인 완료 (총 ${totalElapsed}초)"
Write-Host "  - data\features\modeling_dataset.csv 생성 확인"
Write-Host "  - models\ml\saved\ 에 학습된 모델 확인"
Write-Host "=================================================="