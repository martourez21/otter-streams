-- ============================================================================
-- Fraud Detection Pipeline with ML Inference
-- ============================================================================
--
-- This example demonstrates real-time fraud detection using ML models
-- in Flink SQL with Otter Stream SQL module.
--
-- Model: TensorFlow SavedModel trained on transaction fraud data
-- Features: amount, merchant_category, location_risk, time_features
-- Output: fraud_score (0.0 - 1.0)
-- ============================================================================

-- Register ML prediction function
CREATE TEMPORARY FUNCTION ML_PREDICT AS
    'com.codedstreams.otterstream.sql.udf.MLPredictScalarFunction';

-- ============================================================================
-- SOURCE: Kafka Transactions Stream
-- ============================================================================

CREATE TABLE transactions (
                              transaction_id STRING,
                              user_id STRING,
                              account_id STRING,
                              amount DOUBLE,
                              currency STRING,
                              merchant STRING,
                              merchant_category STRING,
                              merchant_country STRING,
                              card_last_four STRING,
                              card_type STRING,
                              transaction_type STRING,
                              location STRING,
                              device_id STRING,
                              ip_address STRING,
                              transaction_time TIMESTAMP(3),
                              WATERMARK FOR transaction_time AS transaction_time - INTERVAL '5' SECOND
) WITH (
      'connector' = 'kafka',
      'topic' = 'financial.transactions',
      'properties.bootstrap.servers' = 'kafka:9092',
      'properties.group.id' = 'fraud-detection',
      'scan.startup.mode' = 'latest-offset',
      'format' = 'json',
      'json.fail-on-missing-field' = 'false',
      'json.ignore-parse-errors' = 'true'
      );

-- ============================================================================
-- ENRICHMENT: User Profile Lookup
-- ============================================================================

CREATE TABLE user_profiles (
                               user_id STRING,
                               account_age_days INT,
                               total_transactions INT,
                               avg_transaction_amount DOUBLE,
                               high_risk_flag BOOLEAN,
                               country STRING,
                               PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
      'connector' = 'jdbc',
      'url' = 'jdbc:postgresql://postgres:5432/userdb',
      'table-name' = 'user_profiles',
      'username' = 'flink',
      'password' = 'flinkpwd'
      );

-- ============================================================================
-- SINK: Fraud Alerts
-- ============================================================================

CREATE TABLE fraud_alerts (
                              alert_id STRING,
                              transaction_id STRING,
                              user_id STRING,
                              amount DOUBLE,
                              merchant STRING,
                              fraud_score DOUBLE,
                              fraud_confidence DOUBLE,
                              risk_level STRING,
                              risk_factors ARRAY<STRING>,
                              model_version STRING,
                              alert_time TIMESTAMP(3),
                              processing_time_ms BIGINT
) WITH (
      'connector' = 'kafka',
      'topic' = 'fraud.alerts',
      'properties.bootstrap.servers' = 'kafka:9092',
      'format' = 'json'
      );

-- ============================================================================
-- SINK: All Scored Transactions (for analytics)
-- ============================================================================

CREATE TABLE scored_transactions (
                                     transaction_id STRING,
                                     user_id STRING,
                                     amount DOUBLE,
                                     fraud_score DOUBLE,
                                     scored_at TIMESTAMP(3)
) WITH (
      'connector' = 'jdbc',
      'url' = 'jdbc:postgresql://postgres:5432/analyticsdb',
      'table-name' = 'scored_transactions',
      'username' = 'flink',
      'password' = 'flinkpwd'
      );

-- ============================================================================
-- MAIN QUERY: Real-Time Fraud Detection with ML
-- ============================================================================

-- Create enriched view with features
CREATE TEMPORARY VIEW enriched_transactions AS
SELECT
    t.transaction_id,
    t.user_id,
    t.account_id,
    t.amount,
    t.currency,
    t.merchant,
    t.merchant_category,
    t.merchant_country,
    t.location,
    t.transaction_time,

    -- User profile enrichment
    COALESCE(u.account_age_days, 0) AS account_age_days,
    COALESCE(u.total_transactions, 0) AS total_transactions,
    COALESCE(u.avg_transaction_amount, 0.0) AS avg_transaction_amount,
    COALESCE(u.high_risk_flag, false) AS high_risk_flag,

    -- Time-based features
    HOUR(t.transaction_time) AS hour_of_day,
    DAYOFWEEK(t.transaction_time) AS day_of_week,
    CASE
    WHEN DAYOFWEEK(t.transaction_time) IN (1, 7) THEN 1
    ELSE 0
END AS is_weekend,

    -- Location risk scoring
    CASE t.merchant_country
        WHEN 'US' THEN 0.1
        WHEN 'CA' THEN 0.15
        WHEN 'UK' THEN 0.2
        WHEN 'DE' THEN 0.2
        WHEN 'FR' THEN 0.25
        ELSE 0.8
END AS location_risk,

    -- Amount deviation from user average
    CASE
        WHEN u.avg_transaction_amount > 0
        THEN ABS(t.amount - u.avg_transaction_amount) / u.avg_transaction_amount
        ELSE 0.0
END AS amount_deviation,

    -- Processing timestamp for latency tracking
    PROCTIME() AS proc_time

FROM transactions t
LEFT JOIN user_profiles FOR SYSTEM_TIME AS OF t.transaction_time AS u
    ON t.user_id = u.user_id;

-- Create ML predictions view
CREATE TEMPORARY VIEW ml_predictions AS
SELECT
    transaction_id,
    user_id,
    account_id,
    amount,
    currency,
    merchant,
    merchant_category,
    merchant_country,
    location,
    transaction_time,
    account_age_days,
    high_risk_flag,

    -- ML Inference: Call fraud detection model
    ML_PREDICT(
            'fraud-detector-v3',
            JSON_OBJECT(
                -- Numerical features
                    'amount', amount,
                    'amount_log', LN(amount + 1),
                    'account_age_days', account_age_days,
                    'total_transactions', total_transactions,
                    'avg_transaction_amount', avg_transaction_amount,
                    'amount_deviation', amount_deviation,

                -- Categorical features (encoded as risk scores)
                    'location_risk', location_risk,
                    'merchant_category_risk',
                    CASE merchant_category
                        WHEN 'gambling' THEN 0.9
                        WHEN 'crypto' THEN 0.85
                        WHEN 'wire_transfer' THEN 0.8
                        WHEN 'atm_withdrawal' THEN 0.6
                        WHEN 'online_shopping' THEN 0.3
                        ELSE 0.2
                        END,

                -- Temporal features
                    'hour_of_day', hour_of_day,
                    'is_weekend', is_weekend,
                    'is_late_night', CASE WHEN hour_of_day BETWEEN 1 AND 5 THEN 1 ELSE 0 END,

                -- Boolean features
                    'high_risk_flag', CASE WHEN high_risk_flag THEN 1.0 ELSE 0.0 END
            )
    ) AS fraud_score,

    -- Second model for confidence scoring (optional)
    0.95 AS fraud_confidence,  -- Placeholder, could be another model
    'v3.2.1' AS model_version,

    -- Calculate processing latency
    TIMESTAMPDIFF(MILLISECOND, transaction_time, PROCTIME()) AS processing_time_ms

FROM enriched_transactions;

-- ============================================================================
-- OUTPUT 1: High-Risk Fraud Alerts
-- ============================================================================

INSERT INTO fraud_alerts
SELECT
    CONCAT('ALERT-', transaction_id, '-', UNIX_TIMESTAMP()) AS alert_id,
    transaction_id,
    user_id,
    amount,
    merchant,
    fraud_score,
    fraud_confidence,

    -- Risk level classification
    CASE
        WHEN fraud_score >= 0.9 THEN 'CRITICAL'
        WHEN fraud_score >= 0.7 THEN 'HIGH'
        WHEN fraud_score >= 0.5 THEN 'MEDIUM'
        ELSE 'LOW'
        END AS risk_level,

    -- Risk factors array
    ARRAY[
        CASE WHEN amount > 5000 THEN 'high_amount' ELSE NULL END,
    CASE WHEN fraud_score > 0.8 THEN 'ml_high_score' ELSE NULL END,
    CASE WHEN high_risk_flag THEN 'high_risk_user' ELSE NULL END,
    CASE WHEN merchant_country NOT IN ('US', 'CA', 'UK') THEN 'foreign_merchant' ELSE NULL END,
    CASE WHEN hour_of_day BETWEEN 1 AND 5 THEN 'unusual_time' ELSE NULL END
    ] AS risk_factors,

    model_version,
    CURRENT_TIMESTAMP AS alert_time,
    processing_time_ms

FROM ml_predictions
WHERE fraud_score >= 0.5;  -- Alert threshold

-- ============================================================================
-- OUTPUT 2: All Scored Transactions (for analytics/audit)
-- ============================================================================

INSERT INTO scored_transactions
SELECT
    transaction_id,
    user_id,
    amount,
    fraud_score,
    CURRENT_TIMESTAMP AS scored_at
FROM ml_predictions;

-- ============================================================================
-- MONITORING QUERY: Real-time fraud statistics
-- ============================================================================

-- This can be run separately for monitoring
SELECT
    TUMBLE_START(transaction_time, INTERVAL '1' MINUTE) AS window_start,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN fraud_score > 0.5 THEN 1 ELSE 0 END) AS flagged_count,
    AVG(fraud_score) AS avg_fraud_score,
    MAX(fraud_score) AS max_fraud_score,
    SUM(amount) AS total_amount,
    SUM(CASE WHEN fraud_score > 0.5 THEN amount ELSE 0 END) AS flagged_amount,
    AVG(processing_time_ms) AS avg_processing_time_ms
FROM ml_predictions
GROUP BY TUMBLE(transaction_time, INTERVAL '1' MINUTE);

-- ============================================================================
-- NOTES:
-- ============================================================================
--
-- 1. Model Location: Ensure fraud-detector-v3 model is accessible at:
--    - S3: s3://your-bucket/models/fraud-detector-v3/
--    - Local: file:///opt/flink/models/fraud-detector-v3/
--    - HTTP: https://model-server.com/models/fraud-detector-v3/
--
-- 2. Model Format: TensorFlow SavedModel expected
--
-- 3. Performance Tuning:
--    - Adjust watermark interval based on late data tolerance
--    - Set appropriate Kafka consumer parallelism
--    - Configure model cache size based on memory
--
-- 4. Alerting Threshold: Currently set to 0.5, adjust based on:
--    - False positive rate tolerance
--    - Investigation capacity
--    - Business requirements
--
-- 5. Required Permissions:
--    - Kafka: read from transactions, write to fraud.alerts
--    - PostgreSQL: read from user_profiles, write to scored_transactions
--    - S3/Storage: read model artifacts
--
-- ============================================================================