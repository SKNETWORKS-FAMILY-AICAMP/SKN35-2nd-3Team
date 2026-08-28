-- schema.sql의 models 1건 + predictions(SHAP 포함) 2건 적재 자료
-- predictions는 SHAP 표본 중 위험점수가 높은 두 매장을 사용했다.
START TRANSACTION;

INSERT INTO models (model_id, model_name, version, model_type, accuracy, precision_score, recall_score, f1_score, roc_auc, trained_at, is_production)
VALUES ('DNN_20260828_094006_7e84a4a_pjw_official', 'PJW DNN 34F E100 P5', '2026.08.28-e100p5', 'DL', 0.88525, 0.44918, 0.35856, 0.39878, 0.74577, '2026-08-28 09:40:06', 0)
ON DUPLICATE KEY UPDATE
    model_name = VALUES(model_name),
    version = VALUES(version),
    model_type = VALUES(model_type),
    accuracy = VALUES(accuracy),
    precision_score = VALUES(precision_score),
    recall_score = VALUES(recall_score),
    f1_score = VALUES(f1_score),
    roc_auc = VALUES(roc_auc),
    trained_at = VALUES(trained_at),
    is_production = VALUES(is_production);

-- prediction 1
INSERT INTO predictions (model_id, user_id, query_type, store_id, query_lat, query_lng, industry_code, score, shap_top_features, created_at)
SELECT 'DNN_20260828_094006_7e84a4a_pjw_official', NULL, 'existing_store', 'MA0106202201A0025935', NULL, NULL, 'G21202', 0.99386, '[{"feature":"dong_industry_historical_rate","shap_value":2.26920748,"feature_value":0.671875},{"feature":"store_age_months","shap_value":0.65520215,"feature_value":6.0},{"feature":"is_left_censored_age","shap_value":0.63866752,"feature_value":0.0},{"feature":"snapshot_month_index","shap_value":0.59310991,"feature_value":18.0},{"feature":"nearest_same_industry_distance_m","shap_value":0.49035883,"feature_value":0.0}]', '2026-08-28 10:00:14'
WHERE NOT EXISTS (
    SELECT 1 FROM predictions
    WHERE model_id = 'DNN_20260828_094006_7e84a4a_pjw_official'
      AND query_type = 'existing_store'
      AND store_id = 'MA0106202201A0025935'
      AND industry_code = 'G21202'
      AND created_at = '2026-08-28 10:00:14'
);

-- prediction 2
INSERT INTO predictions (model_id, user_id, query_type, store_id, query_lat, query_lng, industry_code, score, shap_top_features, created_at)
SELECT 'DNN_20260828_094006_7e84a4a_pjw_official', NULL, 'existing_store', 'MA0106202509A1423510', NULL, NULL, 'I21001', 0.99171, '[{"feature":"store_age_months","shap_value":2.30236149,"feature_value":0.0},{"feature":"snapshot_month_index","shap_value":1.66985428,"feature_value":24.0},{"feature":"is_left_censored_age","shap_value":-1.02432692,"feature_value":0.0},{"feature":"floor_category_enc","shap_value":0.67917931,"feature_value":2.0},{"feature":"industry_group_enc","shap_value":0.40052232,"feature_value":6.0}]', '2026-08-28 10:00:14'
WHERE NOT EXISTS (
    SELECT 1 FROM predictions
    WHERE model_id = 'DNN_20260828_094006_7e84a4a_pjw_official'
      AND query_type = 'existing_store'
      AND store_id = 'MA0106202509A1423510'
      AND industry_code = 'I21001'
      AND created_at = '2026-08-28 10:00:14'
);

COMMIT;
