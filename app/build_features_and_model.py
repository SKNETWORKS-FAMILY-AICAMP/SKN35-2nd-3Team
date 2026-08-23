"""
서빙용 피처 테이블 + 모델 준비 스크립트 (최초 1회 실행)

역할: models/ml, models/dl 에서 학습이 끝난 최종 모델과
      features/ 에서 만든 피처 로직을 가져와, 앱이 실시간으로
      가볍게 조회/추론할 수 있는 형태로 정리한다.

실행:
    python app/build_features_and_model.py

TODO:
- 학습된 최종 모델 로드 (models/ml/saved 또는 models/dl/saved)
- 서빙용 피처 테이블 생성/저장 (data/features 기반)
- 지금 뜨는 사업 키워드 집계 결과 캐싱
"""
