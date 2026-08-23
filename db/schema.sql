-- ============================================================
-- 서울 상권 폐업예측 프로젝트 — DB 스키마 (TiDB, MySQL 호환)
-- db/erd.png / db/테이블_설명.md 와 1:1 대응
-- ============================================================
-- 참고
--   - TiDB는 MySQL 프로토콜/문법을 대부분 그대로 지원한다.
--   - AUTO_INCREMENT 그대로 사용해도 되지만, 분산 환경에서 PK 핫스팟을
--     피하고 싶으면 AUTO_RANDOM으로 바꾸는 것도 검토 가능 (팀 논의 필요).
--   - FOREIGN KEY 제약이 실제로 강제되는지는 사용 중인 TiDB 버전에서
--     한 번 확인할 것 (과거 버전은 파싱만 되고 강제 안 된 이력이 있음).
--   - CSV에는 가독성을 위해 업종명 등을 코드 옆에 같이 넣어뒀지만,
--     DB는 정규화 원칙대로 코드+FK만 두고 이름은 industries/administrative_dongs와
--     조인해서 가져온다 (중복저장 방지).
-- ============================================================

-- ------------------------------------------------------------
-- 1. 원본 데이터 기반 테이블
-- ------------------------------------------------------------

CREATE TABLE administrative_dongs (
    dong_code       VARCHAR(20)     NOT NULL,
    dong_name       VARCHAR(50)     NOT NULL,
    gu_name         VARCHAR(50)     NOT NULL,
    PRIMARY KEY (dong_code)
);

CREATE TABLE industries (
    industry_code       VARCHAR(20)     NOT NULL,
    industry_name        VARCHAR(100)    NOT NULL,
    industry_jung_code   VARCHAR(20)     NOT NULL,
    industry_jung_name   VARCHAR(100)    NOT NULL,
    industry_dae_code    VARCHAR(20)     NOT NULL,
    custom_group          VARCHAR(50)     NOT NULL,   -- 대분류명을 그대로 채택 (10개 그룹)
    PRIMARY KEY (industry_code)
);

CREATE TABLE stores (
    store_id                VARCHAR(30)     NOT NULL,
    current_industry_code   VARCHAR(20)     NOT NULL,
    dong_code               VARCHAR(20)     NOT NULL,
    first_seen_snapshot     VARCHAR(6)      NOT NULL,   -- 'YYYYMM' 형식
    last_seen_snapshot      VARCHAR(6)      NOT NULL,
    n_snapshots_observed    SMALLINT        NOT NULL,
    is_closed                BOOLEAN         NOT NULL DEFAULT FALSE,
    had_temporary_gap        BOOLEAN         NOT NULL DEFAULT FALSE,  -- 중간에 관측 공백이 있었던 매장(데이터 품질 플래그)
    PRIMARY KEY (store_id),
    FOREIGN KEY (current_industry_code) REFERENCES industries(industry_code),
    FOREIGN KEY (dong_code) REFERENCES administrative_dongs(dong_code)
);

CREATE TABLE store_snapshots (
    snapshot_id         BIGINT          NOT NULL AUTO_INCREMENT,
    store_id             VARCHAR(30)     NOT NULL,
    snapshot_date        VARCHAR(6)      NOT NULL,   -- 202312~202606
    industry_code        VARCHAR(20)     NOT NULL,
    store_name            VARCHAR(200)    NOT NULL,
    lng                    DECIMAL(10, 7)  NOT NULL,
    lat                    DECIMAL(10, 7)  NOT NULL,
    is_closed_next         BOOLEAN         NOT NULL,   -- 다음 스냅샷에서 사라졌는지
    transitioned_next      BOOLEAN         NOT NULL,   -- 다음 스냅샷으로 갈 때 업종이 바뀌었는지
    label_available         BOOLEAN         NOT NULL,   -- 마지막 스냅샷(202606)만 FALSE, 서빙 전용
    PRIMARY KEY (snapshot_id),
    UNIQUE KEY uq_store_snapshot (store_id, snapshot_date),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (industry_code) REFERENCES industries(industry_code)
);

-- ------------------------------------------------------------
-- 2. 파생 피처 테이블
-- ------------------------------------------------------------

CREATE TABLE population_features (
    dong_code               VARCHAR(20)     NOT NULL,
    korean_pop                DECIMAL(12, 4)  NOT NULL,
    foreign_long_pop          DECIMAL(12, 4)  NOT NULL,
    foreign_short_pop         DECIMAL(12, 4)  NOT NULL,
    total_pop_avg              DECIMAL(12, 4)  NOT NULL,
    foreign_short_ratio        DECIMAL(6, 5)   NOT NULL,
    tourist_zone_candidate     BOOLEAN         NOT NULL DEFAULT FALSE,  -- foreign_short_ratio 상위 10% (임계값 조정 가능)
    PRIMARY KEY (dong_code),
    FOREIGN KEY (dong_code) REFERENCES administrative_dongs(dong_code)
);

CREATE TABLE spatial_density_features (
    store_id                             VARCHAR(30)     NOT NULL,
    snapshot_date                         VARCHAR(6)      NOT NULL,
    same_industry_count_300m               INT             NOT NULL,   -- 반경 300m 내 동일업종 매장 수
    total_count_300m                       INT             NOT NULL,   -- 반경 300m 내 전체 업종 매장 수
    nearest_same_industry_distance_m        DECIMAL(10, 3)  NULL,       -- 동일업종이 자기뿐이면 NULL
    dong_industry_count                     INT             NOT NULL,   -- 행정동 전체 기준 동일업종 매장 수
    PRIMARY KEY (store_id, snapshot_date),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

CREATE TABLE trend_keywords (
    keyword          VARCHAR(50)     NOT NULL,
    snapshot_date     VARCHAR(6)      NOT NULL,
    store_count        INT             NOT NULL,
    growth_rate         DECIMAL(8, 4)   NULL,   -- (최신 스냅샷 - 최초 스냅샷) / 최초 스냅샷. 최초값이 0이면 NULL
    PRIMARY KEY (keyword, snapshot_date)
    -- 참고: 원본 산출물(trend_keywords.csv)은 keyword x snapshot_date 와이드 포맷.
    --       적재 시 위 롱 포맷으로 pivot 필요.
);

CREATE TABLE industry_transitions (
    transition_id        BIGINT          NOT NULL AUTO_INCREMENT,
    store_id               VARCHAR(30)     NOT NULL,
    from_snapshot            VARCHAR(6)      NOT NULL,
    to_snapshot              VARCHAR(6)      NOT NULL,
    from_industry_code       VARCHAR(20)     NOT NULL,
    to_industry_code         VARCHAR(20)     NOT NULL,
    PRIMARY KEY (transition_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (from_industry_code) REFERENCES industries(industry_code),
    FOREIGN KEY (to_industry_code) REFERENCES industries(industry_code)
);

CREATE TABLE industry_survival_stats (
    from_industry_code    VARCHAR(20)     NOT NULL,
    to_industry_code       VARCHAR(20)     NOT NULL,
    sample_size              INT             NOT NULL,
    survival_rate             DECIMAL(6, 5)   NOT NULL,
    PRIMARY KEY (from_industry_code, to_industry_code),
    FOREIGN KEY (from_industry_code) REFERENCES industries(industry_code),
    FOREIGN KEY (to_industry_code) REFERENCES industries(industry_code)
);

-- ------------------------------------------------------------
-- 3. 사용자 테이블
-- ------------------------------------------------------------

CREATE TABLE users (
    user_id         VARCHAR(30)     NOT NULL,
    user_type        ENUM('owner', 'founder', 'admin')   NOT NULL,
    store_id          VARCHAR(30)     NULL,   -- user_type='owner'일 때만 값 존재
    login_id           VARCHAR(50)     NOT NULL,
    password_hash       VARCHAR(255)    NOT NULL,   -- 데모 단계: store_id 시드 결정적 값 / 실서비스: bcrypt 등 진짜 해시로 교체 필수
    created_at            DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id),
    UNIQUE KEY uq_login_id (login_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

-- ------------------------------------------------------------
-- 4. 서빙/운영 테이블
-- ------------------------------------------------------------

CREATE TABLE models (
    model_id          VARCHAR(50)     NOT NULL,
    model_name          VARCHAR(100)    NOT NULL,
    version               VARCHAR(20)     NOT NULL,
    model_type             ENUM('ML', 'DL')  NOT NULL,
    accuracy                 DECIMAL(6, 5)   NULL,
    precision_score            DECIMAL(6, 5)   NULL,   -- PRECISION은 예약어라 컬럼명에 _score 붙임
    recall_score               DECIMAL(6, 5)   NULL,
    f1_score                    DECIMAL(6, 5)   NULL,
    roc_auc                       DECIMAL(6, 5)   NULL,
    trained_at                      DATETIME        NOT NULL,
    is_production                     BOOLEAN         NOT NULL DEFAULT FALSE,
    PRIMARY KEY (model_id)
);

CREATE TABLE predictions (
    prediction_id       BIGINT          NOT NULL AUTO_INCREMENT,
    model_id               VARCHAR(50)     NOT NULL,
    user_id                  VARCHAR(30)     NULL,       -- 익명 조회 허용 시 NULL 가능
    query_type                  ENUM('existing_store', 'new_location')  NOT NULL,
    store_id                       VARCHAR(30)     NULL,  -- query_type='existing_store'일 때만
    query_lat                        DECIMAL(10, 7)  NULL,  -- query_type='new_location'일 때만
    query_lng                          DECIMAL(10, 7)  NULL,
    industry_code                        VARCHAR(20)     NOT NULL,
    score                                   DECIMAL(6, 5)   NOT NULL,
    created_at                                DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (industry_code) REFERENCES industries(industry_code)
    -- 애플리케이션 레벨 검증 필요: query_type='existing_store'면 store_id NOT NULL,
    -- query_type='new_location'이면 query_lat/query_lng NOT NULL이 되도록 보장할 것.
);

CREATE TABLE support_actions (
    action_id                BIGINT          NOT NULL AUTO_INCREMENT,
    store_id                    VARCHAR(30)     NOT NULL,
    admin_user_id                  VARCHAR(30)     NOT NULL,
    action_type                       VARCHAR(50)     NOT NULL,
    action_date                          DATE            NOT NULL,
    follow_up_closure_status                BOOLEAN         NULL,   -- 조치 이후 실제 폐업 여부 (closed-loop 추적)
    notes                                       TEXT            NULL,
    PRIMARY KEY (action_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (admin_user_id) REFERENCES users(user_id)
);
