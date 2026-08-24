#!/usr/bin/env bash
# 서울 상권 폐업예측 — 데이터 파이프라인 전체를 한 번에 실행
#
# 실행 위치: 저장소 루트 (SKN35-2nd-3Team/) 에서
#   bash run_pipeline.sh
#
# 사전 조건: data/raw/ 안에 원본 CSV가 있어야 함
#   - 소상공인 6개 스냅샷: seoul_YYYYMM.csv
#   - 생활인구 3종: local_pop.csv / longf_pop.csv / tempf_pop.csv
#
# 실행 순서(raw -> labeled -> features -> modeling dataset -> 학습)대로
# db/etl -> features/* -> models/ml 를 차례로 돌립니다.
# 중간에 하나라도 실패하면 그 자리에서 멈춥니다(뒷 단계가 앞 단계 결과물에 의존하기 때문).

set -euo pipefail

PKG="src/project_2nd"

RUN="uv run python"
STEP_NUM=0
TOTAL_START=$(date +%s)

step() {
  STEP_NUM=$((STEP_NUM + 1))
  local desc="$1"
  local script="$2"
  echo ""
  echo "==> [${STEP_NUM}/10] ${desc}"
  echo "    ${script}"
  local start
  start=$(date +%s)
  if ! ${RUN} "${script}"; then
    echo ""
    echo "!! [${STEP_NUM}] '${desc}' 단계 실패. 여기서 중단합니다."
    echo "   위 에러 메시지 확인 후 원인 해결하고 다시 실행하세요."
    exit 1
  fi
  local end
  end=$(date +%s)
  echo "    완료 (${desc}, $((end - start))초)"
}

if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  echo "data/raw/ 에 원본 CSV가 없습니다. 먼저 데이터를 넣어주세요."
  exit 1
fi

echo "저장소 루트에서 uv 환경 동기화 중..."
uv sync

step "폐업 라벨·업종전환 이력 생성"     "${PKG:+$PKG/}db/etl/01_build_closure_transitions.py"
step "스냅샷별 매장 테이블 생성"        "${PKG:+$PKG/}db/etl/02_build_store_snapshots.py"
step "공간 밀도 피처 (BallTree/Haversine)" "${PKG:+$PKG/}features/spatial/build_spatial_features.py"
step "생활인구 피처"                    "${PKG:+$PKG/}features/spatial/build_population_features.py"
step "업종 그룹핑"                      "${PKG:+$PKG/}features/industry_grouping/build_industries.py"
step "트렌드 키워드 탐지"               "${PKG:+$PKG/}features/trend_keywords/build_trend_keywords.py"
step "업종 전환 생존율 피처"            "${PKG:+$PKG/}features/survival_transition/build_survival_stats.py"
step "최종 모델링 데이터셋 조립"        "${PKG:+$PKG/}models/ml/01_build_modeling_dataset.py"
step "LightGBM 베이스라인 학습"        "${PKG:+$PKG/}models/ml/02_train_baseline.py"
step "임계값 튜닝"                     "${PKG:+$PKG/}models/ml/03_tune_threshold.py"

TOTAL_END=$(date +%s)
echo ""
echo "=================================================="
echo "전체 파이프라인 완료 (총 $((TOTAL_END - TOTAL_START))초)"
echo "  - data/features/modeling_dataset.csv 생성 확인"
echo "  - models/ml/saved/ 에 학습된 모델 확인"
echo "=================================================="