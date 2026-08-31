# 🏪 서울 상권 폐업 위험 예측 — 우리 동네, 우리 가게는 안전할까?

> "이 자리에 이 업종, 지금 시작해도 괜찮을까?" 예비창업자·기존점주·관리자 세 유형에 맞춰, 서울시 상가업소 데이터와 생활인구 데이터로 학습한 모델이 상권별·업종별 폐업 위험도를 계산해주는 Streamlit 웹 애플리케이션입니다.

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-1.60+-FF4B4B?logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/TiDB%20Cloud-MySQL%20호환-4479A1?logo=mysql&logoColor=white">
<img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9">
<img src="https://img.shields.io/badge/LightGBM%20%7C%20ExtraTrees%20%7C%20DNN-ML%2FDL-F7931E">
<img src="https://img.shields.io/badge/SHAP-설명가능성-8A2BE2">
<img src="https://img.shields.io/badge/Gemini%20API-상담%20챗봇-4285F4?logo=googlegemini&logoColor=white">
</p>

---

## 📌 서비스명

**서울 상권 폐업 위험 예측 시스템** — 상권 데이터 기반 폐업 위험도 진단 & 업종/입지 추천 서비스

## 📖 프로젝트 개요

신규 창업 3년 내 폐업률이 절반에 가까운 상황에서, "감"이 아니라 **실제 상가업소 데이터와 생활인구 데이터**로 상권과 업종의 생존 가능성을 미리 가늠해볼 수 있게 만든 서비스입니다. 로그인한 사용자 유형에 따라 화면과 기능이 달라집니다.

- 🧭 **예비창업자**: 관심 지역·업종의 위험도/추천도 점수 확인, "관심 업종으로 동네 찾기"로 반경 300m 경쟁밀도 기반 추천 동네 탐색
- 🏪 **기존점주**: 내 가게 개인화 폐업 위험 점수 + SHAP 기반 근거 설명 + Wilson score 하한 기반 업종전환 추천
- 🛡️ **관리자**: 고위험 상권 모니터링 대시보드, 지원 조치(support_actions) 이력 추적
- 🔥 **지금 뜨는 사업 탐지**: 실제 상호명 키워드(탕후루, 버터떡 등) 매장수를 6개 스냅샷에 걸쳐 집계해 증가율로 트렌드를 탐지 (더미 데이터 없이 실측 기반)
- 💬 **상담 챗봇**: 화면에 떠 있는 실제 DB 조회 결과 또는 메시지에서 직접 추출한 동/업종 정보를 근거로만 답하도록 설계해 환각(hallucination)을 최소화

## 👥 팀원

<img src="./app/assets/heichi.png" width="120"> <img src="./app/assets/w_tiger.png" width="120"> <img src="./app/assets/yolo.png" width="120"> <img src="./app/assets/jujak.png" width="120"> <img src="./app/assets/blue_d.png" width="120">

## 🛠️ 기술 스택

| 분류 | 기술 |
| --- | --- |
| Language | ![Python](https://img.shields.io/badge/Python%203.12-3776AB?logo=python&logoColor=white) |
| Frontend / App | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) |
| Database | ![TiDB](https://img.shields.io/badge/TiDB%20Cloud-4479A1?logo=mysql&logoColor=white) ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logoColor=white) ![PyMySQL](https://img.shields.io/badge/PyMySQL-4479A1?logoColor=white) |
| ML | ![LightGBM](https://img.shields.io/badge/LightGBM-baseline-9ACD32) ![ExtraTrees](https://img.shields.io/badge/ExtraTrees-최종%20채택-2E8B57) ![Optuna](https://img.shields.io/badge/Optuna-하이퍼파라미터튜닝-0078D4) |
| DL | ![PyTorch](https://img.shields.io/badge/DNN%2FMLP-5--fold%20ensemble-EE4C2C) |
| 설명가능성 | ![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer%20%2F%20KernelExplainer-8A2BE2) |
| 공간 연산 | ![scipy](https://img.shields.io/badge/BallTree%20(haversine)-scipy-8CAAE6?logo=scipy&logoColor=white) |
| 지도 시각화 | ![folium](https://img.shields.io/badge/folium-OpenStreetMap-77B829) |
| 챗봇 | ![Gemini](https://img.shields.io/badge/Gemini%20API-google--genai-4285F4?logo=googlegemini&logoColor=white) |
| 환경/패키지 관리 | ![uv](https://img.shields.io/badge/uv-DE5FE9?logoColor=white) ![python-dotenv](https://img.shields.io/badge/python--dotenv-ECD53F?logoColor=black) |

## 🧩 핵심 기술

- **생존점수 변환**: 모델이 출력하는 폐업확률(p)을 `생존점수 = (1 - p) × 100`으로 변환해 UI에 노출. 다만 calibration 검증이 안 된 원본 확률이라, 절대 점수보다 **"서울 423개 동 중 O위"** 같은 상대 순위로 표현
- **패널 데이터 누수 방지**: 동일 점포가 6개 스냅샷에 반복 등장하는 구조라, `GroupKFold`(K=5, store_id 해시)로 같은 점포가 Train/Test에 동시에 들어가지 않도록 분할
- **행정동 코드 버전 불일치 처리**: 생활인구(2016년 코드) ↔ 상가업소(현행 코드) 불일치를 `BallTree` 최근접 이웃으로 대체하고 `population_is_proxied` 플래그로 투명하게 표시
- **Wilson score 하한 기반 추천 정렬**: 업종전환 추천에서 표본 1~2건짜리가 100점으로 부풀려지는 문제를 방지
- **모델 자동 승격 파이프라인**: `load_models_and_predictions.py --auto-promote-best`가 `roc_auc` 기준으로 최신 최고 성능 모델을 자동으로 프로덕션에 반영
- **역할 분리형 데이터 흐름**: 모델 담당은 JSON으로 결과를 저장 → DB 담당이 적재 → 각자 병렬로 작업 가능

## 🗂️ Project Structure

```text
SKN35-2nd-3Team/
├── data/                      # 데이터 파이프라인 단계별 저장 (raw → processed → labeled → features)
├── db/                        # DB 스키마·ERD·적재(ETL) 스크립트
│   ├── schema.sql
│   ├── erd.png / erd.dot
│   └── etl/
├── features/                  # 피처 엔지니어링
│   ├── spatial/                #   BallTree/Haversine 기반 반경 내 경쟁업소 밀도
│   ├── industry_grouping/      #   업종명 기반 그룹핑
│   ├── trend_keywords/         #   '지금 뜨는 사업' 키워드 트렌드 탐지
│   └── survival_transition/    #   업종 전환 이력 기반 생존율
├── models/                    # ML/DL 모델링
│   ├── ml/                     #   LightGBM, ExtraTrees(Optuna 튜닝) 등
│   ├── dl/                     #   DNN/MLP (5-fold 앙상블)
│   └── shap/                   #   SHAP 기반 설명가능성
├── app/                       # Streamlit 앱
│   ├── app.py                  #   진입점 — 로그인 라우팅 (founder/owner/admin)
│   ├── pages/                  #   화면별 페이지
│   └── shared/                 #   공용 모듈 — auth.py, db.py, components.py, llm_client.py
├── docs/                       # 설계/로드맵/결과서 문서
├── notebooks/                  # EDA·실험용 노트북
├── requirements.txt
└── .gitignore
```

## 🗄️ ERD

`db/erd.png` 참고 (13개 테이블, TiDB Cloud/MySQL 호환).

주요 테이블 관계 요약:

```text
stores 1─N store_snapshots            (점포별 스냅샷 6회, 폐업 여부 is_closed_next)
administrative_dongs 1─N store_snapshots
industries 1─N store_snapshots
industries 1─N industry_survival_stats  (업종전환 생존율)
stores 1─N spatial_density_features    (반경 300m 경쟁 밀도)
administrative_dongs 1─N population_features (생활인구)
users 1─N predictions                  (예측 결과 캐시)
users 1─N support_actions              (관리자 개입 이력)
```

## 🤖 모델 성능

`GroupKFold`(K=5, store_id 해시 기준) 검증 기준, 약 188만 행(소비자 대면 9개 업종군, 폐업률 10.6%) 학습.

| 모델 ID | 모델명 | 버전 | 유형 | 정확도 | 정밀도 | 재현율 | F1 | ROC-AUC | PR-AUC | 운영 반영 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pmh_ml_extratrees_pjw_v1` | ExtraTreesClassifier (Optuna tuned) | 1.0 | ML | 0.888 | 0.4668 | 0.3578 | 0.4051 | **0.7486** | 약 0.40 | ✅ |
| `dnn_mlp_v2` | DNN (5-fold ensemble) | 2.0 | DL | 0.855 | 0.3324 | 0.3524 | 0.3417 | 0.7233 | `[TODO]` | ☐ |
| (베이스라인) LightGBM | LightGBM | - | ML | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | 약 0.721\~0.725 | 0.300 | ☐ |

**최종 채택 모델**: **ExtraTreesClassifier (`pmh_ml_extratrees_pjw_v1`)** — LightGBM 베이스라인과 DNN 둘 다보다 ROC-AUC가 높아 `--auto-promote-best` 기준으로 프로덕션에 자동 반영됨.

> ⚠️ Accuracy(0.888)는 기저 폐업률(10.6%)이 낮은 데이터 특성상 단독으로는 성능 근거가 되기 어렵습니다. **ROC-AUC·PR-AUC·F1을 함께 봐야 합니다.** 자세한 내용은 [`docs/02_AI_학습_결과서.md`](./docs/02_AI_학습_결과서.md) 참고.

## 📄 산출물

- [`docs/01_데이터_전처리_결과서.md`](./docs/01_데이터_전처리_결과서.md) — 인공지능 데이터 전처리 결과서
- [`docs/02_AI_학습_결과서.md`](./docs/02_AI_학습_결과서.md) — 인공지능 학습 결과서
- [`docs/03_학습된_모델_설명.md`](./docs/03_학습된_모델_설명.md) — 학습된 인공지능 모델 설명
- 발표 PPT — `[TODO: 링크 추가]`

## ⚙️ 실행 방법

필요한 도구: `Git`, `uv`

```bash
git clone https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN35-2nd-3Team.git
cd SKN35-2nd-3Team
uv sync
```

### DB 접속 설정

```bash
cp .env.example .env   # [TODO: 예시 파일 실제로 추가]
```

`.env`에 아래 항목을 채웁니다 (`app/shared/db.py` 참고, `DB_` 접두사 사용 — Windows의 `USERNAME` 환경변수와 충돌 방지).

```
DB_HOST=...
DB_PORT=4000
DB_USERNAME=...
DB_PASSWORD=...
DB_DATABASE=...
GEMINI_API_KEY=...   # 챗봇용, 각자 https://aistudio.google.com/ 에서 무료 발급
```

### 앱 실행

```bash
uv run python -m streamlit run app/app.py
```

> ⚠️ `streamlit run app/app.py`를 직접 실행하면 uv trampoline 에러가 발생합니다. 반드시 `uv run python -m streamlit run ...` 형태로 실행하세요.

### 모델 파이프라인 (선택 — 이미 학습된 모델을 쓰는 경우 생략 가능)

```bash
# 1) 피처 생성 → 2) 모델링 데이터셋 조립 → 3) 모델 학습 → 4) DB 적재
python build_modeling_dataset.py
python dl_train_tm.py                              # DNN
[TODO: ExtraTrees/LightGBM 학습 스크립트 커맨드]
python load_models_and_predictions.py --auto-promote-best
```

## ⚠️ 한계점 및 트러블슈팅

| 행정동 코드 버전 불일치 | 클래스 불균형 | 모델링 제외 업종 | 좌표 중복 |
| --- | --- | --- | --- |
| 🗺️ 2016년 vs 현행 코드 | ⚖️ 폐업률 10.6% | 🚫 학습 대상 아닌 업종군 존재 | 🏢 대형 복합건물 |

- **행정동 코드 버전 불일치**: 생활인구 데이터는 2016년 행정동 코드, 상가업소 데이터는 현행 코드 기준이라 강북구 수유동 등 12개 동에서 매칭이 안 됐습니다. `BallTree` 최근접 이웃 대체 + `population_is_proxied` 플래그로 투명하게 구분해 처리했습니다.
- **클래스 불균형(폐업률 10.6%)**: Accuracy만으로는 성능을 판단할 수 없어 ROC-AUC/PR-AUC/F1을 함께 리포트했고, 서비스 화면에서도 절대 확률 대신 상대 순위로 표현해 과신을 방지했습니다.
- **모델링 제외 업종**: 과학·기술/부동산/시설관리·임대 업종군은 애초에 학습 대상이 아니라서, 챗봇/화면에서 해당 업종을 물어보면 "예측 대상 아님"으로 명확히 안내하도록 처리했습니다.
- **좌표 중복(대형 복합건물)**: 송파구 등에서 883개 상점이 동일 좌표를 공유하는 문제를 `DBSCAN` 클러스터링으로 탐지하고 `coord_cluster_size` 피처로 반영했습니다.
- **업종전환 이벤트 쏠림**: 전환 이벤트의 97%가 특정 시점(202406→202412)에 몰려 있어, 실제 전환 트렌드라기보다 데이터 일괄 재분류일 가능성을 결과서에 명시했습니다.

## 🔗 앞으로의 개선 내역

- [ ] DNN 5-fold 앙상블과 ExtraTrees/LightGBM 최종 3파전 비교표 완성 (DNN·LightGBM 세부 지표 보강)
- [ ] ExtraTrees PR-AUC 정확한 수치 산출 (현재 약 0.40으로 추정치)
- [ ] `coord_cluster_size` 피처의 최종 데이터셋 내 성능 기여도 재평가
- [ ] '지금 뜨는 사업' 탐지에 네이버 데이터랩 검색어트렌드 API 교차검증 추가
- [ ] 마이페이지(`app/pages/mypage.py`) 연동
- [ ] SHAP 근거 문장을 사람이 읽기 쉬운 문장으로 매핑하는 테이블 구축
