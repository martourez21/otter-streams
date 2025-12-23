-- ============================================================================
-- Real-Time Anomaly Detection for IoT/Monitoring Systems
-- ============================================================================
--
-- This example demonstrates anomaly detection on sensor data streams
-- using ML models for predictive maintenance and alerting.
--
-- Model: TensorFlow SavedModel (autoencoder or isolation forest)
-- Features: sensor readings, statistical aggregates, temporal patterns
-- Output: anomaly_score (0.0 = normal, 1.0 = highly anomalous)
-- ============================================================================

-- Register ML functions
CREATE TEMPORARY FUNCTION ML_PREDICT AS
    'com.codedstreams.otterstream.sql.udf.MLPredictScalarFunction';

-- ============================================================================
-- SOURCE: IoT Sensor Data Stream
-- ============================================================================

CREATE TABLE sensor_readings (
                                 sensor_id STRING,
                                 device_id STRING,
                                 device_type STRING,
                                 location STRING,
                                 temperature DOUBLE,
                                 vibration DOUBLE,
                                 pressure DOUBLE,
                                 humidity DOUBLE,
                                 power_consumption DOUBLE,
                                 rotation_speed DOUBLE,
                                 reading_time TIMESTAMP(3),
                                 WATERMARK FOR reading_time AS reading_time - INTERVAL '5' SECOND
) WITH (
      'connector' = 'kafka',
      'topic' = 'iot.sensor.readings',
      'properties.bootstrap.servers' = 'kafka:9092',
      'properties.group.id' = 'anomaly-detection',
      'format' = 'json',
      'scan.startup.mode' = 'latest-offset'
      );

-- ============================================================================
-- ENRICHMENT: Device Metadata
-- ============================================================================

CREATE TABLE device_metadata (
                                 device_id STRING,
                                 manufacturer STRING,
                                 model STRING,
                                 installation_date DATE,
                                 last_maintenance_date DATE,
                                 expected_lifetime_hours INT,
                                 criticality_level STRING,  -- 'low', 'medium', 'high', 'critical'
                                 baseline_temperature DOUBLE,
                                 baseline_vibration DOUBLE,
                                 baseline_pressure DOUBLE,
                                 PRIMARY KEY (device_id) NOT ENFORCED
) WITH (
      'connector' = 'jdbc',
      'url' = 'jdbc:postgresql://postgres:5432/iotdb',
      'table-name' = 'devices'
      );

-- ============================================================================
-- SINK: Anomaly Alerts
-- ============================================================================

CREATE TABLE anomaly_alerts (
                                alert_id STRING,
                                sensor_id STRING,
                                device_id STRING,
                                device_type STRING,
                                location STRING,
                                anomaly_score DOUBLE,
                                anomaly_type STRING,
                                severity STRING,
                                affected_metrics ARRAY<STRING>,
                                recommended_action STRING,
                                alert_time TIMESTAMP(3),
                                window_start TIMESTAMP(3),
                                window_end TIMESTAMP(3)
) WITH (
      'connector' = 'kafka',
      'topic' = 'iot.anomaly.alerts',
      'properties.bootstrap.servers' = 'kafka:9092',
      'format' = 'json'
      );

-- ============================================================================
-- SINK: Anomaly Analytics
-- ============================================================================

CREATE TABLE anomaly_analytics (
                                   device_id STRING,
                                   window_start TIMESTAMP(3),
                                   anomaly_score DOUBLE,
                                   temperature_avg DOUBLE,
                                   vibration_avg DOUBLE,
                                   pressure_avg DOUBLE,
                                   is_anomaly BOOLEAN
) WITH (
      'connector' = 'jdbc',
      'url' = 'jdbc:postgresql://postgres:5432/analyticsdb',
      'table-name' = 'device_anomalies'
      );

-- ============================================================================
-- TEMPORARY VIEW: Statistical Features (5-minute windows)
-- ============================================================================

CREATE TEMPORARY VIEW sensor_statistics AS
SELECT
    sensor_id,
    device_id,
    device_type,
    location,
    TUMBLE_START(reading_time, INTERVAL '5' MINUTES) AS window_start,
    TUMBLE_END(reading_time, INTERVAL '5' MINUTES) AS window_end,

    -- Temperature statistics
    AVG(temperature) AS temp_avg,
    MIN(temperature) AS temp_min,
    MAX(temperature) AS temp_max,
    STDDEV_POP(temperature) AS temp_stddev,

    -- Vibration statistics
    AVG(vibration) AS vibration_avg,
    MIN(vibration) AS vibration_min,
    MAX(vibration) AS vibration_max,
    STDDEV_POP(vibration) AS vibration_stddev,

    -- Pressure statistics
    AVG(pressure) AS pressure_avg,
    MIN(pressure) AS pressure_min,
    MAX(pressure) AS pressure_max,
    STDDEV_POP(pressure) AS pressure_stddev,

    -- Humidity statistics
    AVG(humidity) AS humidity_avg,
    STDDEV_POP(humidity) AS humidity_stddev,

    -- Power consumption
    AVG(power_consumption) AS power_avg,
    MAX(power_consumption) AS power_max,

    -- Rotation speed
    AVG(rotation_speed) AS rotation_avg,
    STDDEV_POP(rotation_speed) AS rotation_stddev,

    -- Data quality metrics
    COUNT(*) AS reading_count,
    COUNT(CASE WHEN temperature IS NULL THEN 1 END) AS missing_temp_count,

    -- Trend indicators (comparing first half vs second half of window)
    (AVG(CASE WHEN reading_time >= TUMBLE_START(reading_time, INTERVAL '5' MINUTES) + INTERVAL '2.5' MINUTES
         THEN temperature END) -
     AVG(CASE WHEN reading_time < TUMBLE_START(reading_time, INTERVAL '5' MINUTES) + INTERVAL '2.5' MINUTES
         THEN temperature END)) AS temp_trend

FROM sensor_readings
GROUP BY
    sensor_id,
    device_id,
    device_type,
    location,
    TUMBLE(reading_time, INTERVAL '5' MINUTES);

-- ============================================================================
-- TEMPORARY VIEW: Enriched Sensor Data with Baselines
-- ============================================================================

CREATE TEMPORARY VIEW enriched_sensor_data AS
SELECT
    s.sensor_id,
    s.device_id,
    s.device_type,
    s.location,
    s.window_start,
    s.window_end,
    s.temp_avg,
    s.temp_min,
    s.temp_max,
    s.temp_stddev,
    s.vibration_avg,
    s.vibration_stddev,
    s.pressure_avg,
    s.pressure_stddev,
    s.humidity_avg,
    s.power_avg,
    s.rotation_avg,
    s.reading_count,
    s.temp_trend,

    -- Device metadata
    d.manufacturer,
    d.model,
    d.installation_date,
    d.last_maintenance_date,
    d.criticality_level,
    d.baseline_temperature,
    d.baseline_vibration,
    d.baseline_pressure,

    -- Deviation from baseline (normalized)
    ABS(s.temp_avg - COALESCE(d.baseline_temperature, s.temp_avg)) /
    NULLIF(COALESCE(d.baseline_temperature, 1.0), 0.0) AS temp_deviation,
    ABS(s.vibration_avg - COALESCE(d.baseline_vibration, s.vibration_avg)) /
    NULLIF(COALESCE(d.baseline_vibration, 1.0), 0.0) AS vibration_deviation,
    ABS(s.pressure_avg - COALESCE(d.baseline_pressure, s.pressure_avg)) /
    NULLIF(COALESCE(d.baseline_pressure, 1.0), 0.0) AS pressure_deviation,

    -- Time since last maintenance (days)
    DATEDIFF(DAY, d.last_maintenance_date, CURRENT_DATE) AS days_since_maintenance,

    -- Age of device (days)
    DATEDIFF(DAY, d.installation_date, CURRENT_DATE) AS device_age_days

FROM sensor_statistics s
         LEFT JOIN device_metadata FOR SYSTEM_TIME AS OF s.window_end AS d
                   ON s.device_id = d.device_id;

-- ============================================================================
-- MAIN QUERY: ML-Based Anomaly Detection
-- ============================================================================

CREATE TEMPORARY VIEW ml_anomaly_detection AS
SELECT
    sensor_id,
    device_id,
    device_type,
    location,
    window_start,
    window_end,
    temp_avg,
    vibration_avg,
    pressure_avg,
    criticality_level,

    -- Call anomaly detection model
    ML_PREDICT(
            'anomaly-detector-autoencoder',
            JSON_OBJECT(
                -- Statistical features
                    'temp_avg', temp_avg,
                    'temp_stddev', temp_stddev,
                    'temp_min', temp_min,
                    'temp_max', temp_max,
                    'temp_trend', temp_trend,

                    'vibration_avg', vibration_avg,
                    'vibration_stddev', vibration_stddev,

                    'pressure_avg', pressure_avg,
                    'pressure_stddev', pressure_stddev,

                    'humidity_avg', humidity_avg,
                    'power_avg', power_avg,
                    'rotation_avg', rotation_avg,

                -- Deviation features (key for anomaly detection)
                    'temp_deviation', temp_deviation,
                    'vibration_deviation', vibration_deviation,
                    'pressure_deviation', pressure_deviation,

                -- Temporal features
                    'hour_of_day', HOUR(window_end),
                    'day_of_week', DAYOFWEEK(window_end),

                -- Device context
                    'days_since_maintenance', days_since_maintenance,
                    'device_age_days', device_age_days,
                    'reading_count', reading_count,

                -- Criticality encoding
                    'criticality_encoded',
                    CASE criticality_level
                        WHEN 'low' THEN 0.0
                        WHEN 'medium' THEN 0.33
                        WHEN 'high' THEN 0.66
                        WHEN 'critical' THEN 1.0
                        ELSE 0.0
                        END
            )
    ) AS anomaly_score,

    -- Additional context
    temp_deviation,
    vibration_deviation,
    pressure_deviation,
    days_since_maintenance,
    device_age_days

FROM enriched_sensor_data
WHERE reading_count >= 50;  -- Ensure sufficient data points

-- ============================================================================
-- OUTPUT 1: High-Severity Anomaly Alerts
-- ============================================================================

INSERT INTO anomaly_alerts
SELECT
    CONCAT('ANOMALY-', device_id, '-', UNIX_TIMESTAMP(window_start)) AS alert_id,
    sensor_id,
    device_id,
    device_type,
    location,
    anomaly_score,

    -- Classify anomaly type
    CASE
        WHEN temp_deviation > 0.5 AND vibration_deviation > 0.5 THEN 'MULTI_PARAMETER'
        WHEN temp_deviation > 0.5 THEN 'TEMPERATURE'
        WHEN vibration_deviation > 0.5 THEN 'VIBRATION'
        WHEN pressure_deviation > 0.5 THEN 'PRESSURE'
        ELSE 'GENERAL'
        END AS anomaly_type,

    -- Severity based on anomaly score and criticality
    CASE
        WHEN anomaly_score >= 0.9 AND criticality_level = 'critical' THEN 'CRITICAL'
        WHEN anomaly_score >= 0.8 THEN 'HIGH'
        WHEN anomaly_score >= 0.6 THEN 'MEDIUM'
        ELSE 'LOW'
        END AS severity,

    -- List affected metrics
    ARRAY[
        CASE WHEN temp_deviation > 0.3 THEN 'temperature' ELSE NULL END,
    CASE WHEN vibration_deviation > 0.3 THEN 'vibration' ELSE NULL END,
    CASE WHEN pressure_deviation > 0.3 THEN 'pressure' ELSE NULL END
    ] AS affected_metrics,

    -- Recommended action
    CASE
        WHEN anomaly_score >= 0.9 THEN 'IMMEDIATE_SHUTDOWN_AND_INSPECTION'
        WHEN anomaly_score >= 0.8 THEN 'SCHEDULE_URGENT_MAINTENANCE'
        WHEN anomaly_score >= 0.6 THEN 'SCHEDULE_MAINTENANCE'
        WHEN days_since_maintenance > 90 THEN 'ROUTINE_MAINTENANCE_DUE'
        ELSE 'MONITOR_CLOSELY'
END AS recommended_action,

    CURRENT_TIMESTAMP AS alert_time,
    window_start,
    window_end

FROM ml_anomaly_detection
WHERE anomaly_score >= 0.6  -- Alert threshold
   OR (anomaly_score >= 0.5 AND criticality_level = 'critical');

-- ============================================================================
-- OUTPUT 2: All Anomaly Scores (for analytics and model improvement)
-- ============================================================================

INSERT INTO anomaly_analytics
SELECT
    device_id,
    window_start,
    anomaly_score,
    temp_avg,
    vibration_avg,
    pressure_avg,
    CASE WHEN anomaly_score >= 0.6 THEN true ELSE false END AS is_anomaly
FROM ml_anomaly_detection;

-- ============================================================================
-- MONITORING QUERY: Anomaly detection statistics
-- ============================================================================

SELECT
    TUMBLE_START(window_start, INTERVAL '1' HOUR) AS hour_start,
    COUNT(DISTINCT device_id) AS devices_monitored,
    COUNT(*) AS total_windows_analyzed,
    SUM(CASE WHEN anomaly_score >= 0.6 THEN 1 ELSE 0 END) AS anomalies_detected,
    AVG(anomaly_score) AS avg_anomaly_score,
    MAX(anomaly_score) AS max_anomaly_score,
    SUM(CASE WHEN criticality_level = 'critical' AND anomaly_score >= 0.6 THEN 1 ELSE 0 END) AS critical_device_anomalies
FROM ml_anomaly_detection
GROUP BY TUMBLE(window_start, INTERVAL '1' HOUR);

-- ============================================================================
-- ADVANCED: Predictive Maintenance Score
-- ============================================================================

CREATE TEMPORARY VIEW predictive_maintenance AS
SELECT
    device_id,
    window_end,
    anomaly_score,
    days_since_maintenance,

    -- Calculate maintenance urgency score
    (anomaly_score * 0.7 +
     LEAST(days_since_maintenance / 180.0, 1.0) * 0.3) AS maintenance_urgency,

    -- Estimate remaining useful life (days)
    CASE
        WHEN anomaly_score >= 0.9 THEN 7
        WHEN anomaly_score >= 0.8 THEN 14
        WHEN anomaly_score >= 0.7 THEN 30
        WHEN anomaly_score >= 0.6 THEN 60
        ELSE 90
        END AS estimated_rul_days,

    criticality_level

FROM ml_anomaly_detection
WHERE anomaly_score >= 0.5;

-- ============================================================================
-- NOTES:
-- ============================================================================
--
-- 1. Model Configuration:
--    - Model: Autoencoder or Isolation Forest trained on normal behavior
--    - Location: s3://your-bucket/models/anomaly-detector-autoencoder/
--    - Format: TensorFlow SavedModel
--    - Training: Use historical data with labeled normal/anomalous periods
--
-- 2. Feature Engineering:
--    - Statistical aggregates over 5-minute windows
--    - Deviation from baseline (critical for anomaly detection)
--    - Temporal patterns (time of day, day of week)
--    - Device age and maintenance history
--
-- 3. Threshold Tuning:
--    - Current alert threshold: 0.6
--    - Adjust based on false positive rate
--    - Consider device criticality in thresholds
--    - Use historical data to calibrate
--
-- 4. Performance Considerations:
--    - Window size: 5 minutes (balance between responsiveness and stability)
--    - Minimum readings: 50 per window (data quality)
--    - Consider batch inference for high-volume scenarios
--
-- 5. Alert Routing:
--    - CRITICAL severity: immediate paging
--    - HIGH severity: urgent ticket
--    - MEDIUM severity: maintenance queue
--    - LOW severity: monitoring dashboard
--
-- 6. Model Maintenance:
--    - Retrain monthly with new normal behavior patterns
--    - Update baselines quarterly
--    - Monitor model performance (false positive/negative rates)
--    - A/B test model versions in production
--
-- 7. Integration Points:
--    - CMMS (Computerized Maintenance Management System)
--    - SCADA (Supervisory Control and Data Acquisition)
--    - Alerting systems (PagerDuty, Opsgenie)
--    - Dashboards (Grafana, Kibana)
--
-- ============================================================================