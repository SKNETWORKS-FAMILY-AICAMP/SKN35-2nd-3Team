# data/

원본·전처리 데이터 파일은 git에 커밋하지 않습니다(용량이 커서 저장소가 무거워짐). 대신 팀 공유 스토리지(구글 드라이브 등)에 올려두고, 아래 폴더 구조에 맞게 로컬에 내려받아 사용하세요.

```
data/
├── raw/        # 공공데이터 원본 CSV (다운로드 그대로)
└── processed/  # load_and_clean() 등으로 정리한 결과물
```

## 받는 법

1. 팀 공유 드라이브 링크: (링크 채워 넣기)
2. 필요한 파일을 받아 `data/raw/`에 그대로 저장 (파일명은 아래 "현재 보유 파일" 참고 — 원본은 한글명이라 영문으로 rename해서 둠)
3. `scripts/run_all.py`로 인허가 6개 업종 전처리·통합, `scripts/run_retail_pilot.py`로 소매업 스냅샷 비교, `scripts/add_features.py`로 밀집도·대중교통 접근성 피처 추가 — 순서대로 실행하면 `data/processed/`에 결과 생성

## 현재 보유 파일

원본은 사이트 캡차/지역 접근 제한 때문에 자동 다운로드가 안 돼서 브라우저로 직접 받아야 함 (자세한 내용은 `PROJECT_BRIEF.md` 참고).

`raw/` — 인허가 6개 업종(전국): `restaurant_nationwide.csv`, `lodging_nationwide.csv`, `beauty_nationwide.csv`, `laundry_nationwide.csv`, `bathhouse_nationwide.csv`, `pc_cafe_nationwide.csv` (모두 cp949)
소매업 스냅샷(상가상권정보, 지역별 CSV): `commercial_20240630/`, `commercial_20241231/`, `commercial_20250630/`, `commercial_20251231/`, `commercial_20260630/` (현재 파이프라인은 `20250630`↔`20260630` 서울 파일만 사용, 나머지는 보류)
대중교통: `bus_stops_nationwide.csv`(cp949), `subway_stations_nationwide.xlsx`

`processed/` — `all_industries_clean.csv`(인허가 6개 업종 통합), `retail_seoul_clean.csv`(소매업 4개 카테고리, 서울), `all_industries_features.csv`/`retail_seoul_features.csv`(위 두 파일 + 업종밀집도·대중교통 접근성)
