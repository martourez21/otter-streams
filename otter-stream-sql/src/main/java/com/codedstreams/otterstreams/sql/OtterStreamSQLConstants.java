package com.codedstreams.otterstreams.sql;

/**
 * Constants used throughout Otter Stream SQL module.
 *
 * @author Nestor Martourez A.
 * @since 1.0.0
 */
public final class OtterStreamSQLConstants {

    // Function names
    public static final String FUNCTION_ML_PREDICT = "ML_PREDICT";
    public static final String FUNCTION_ML_PREDICT_ASYNC = "ML_PREDICT_ASYNC";
    public static final String FUNCTION_ML_PREDICT_TABLE = "ML_PREDICT_TABLE";
    public static final String FUNCTION_ML_PREDICT_AGG = "ML_PREDICT_AGG";

    // Connector identifier
    public static final String CONNECTOR_IDENTIFIER = "ml-inference";

    // Default configuration values
    public static final int DEFAULT_BATCH_SIZE = 1;
    public static final long DEFAULT_BATCH_TIMEOUT_MS = 50;
    public static final long DEFAULT_ASYNC_TIMEOUT_MS = 5000;
    public static final int DEFAULT_MAX_RETRIES = 3;
    public static final long DEFAULT_RETRY_BACKOFF_MS = 100;
    public static final int DEFAULT_CACHE_SIZE = 100;
    public static final long DEFAULT_CACHE_TTL_MINUTES = 30;

    // Model formats
    public static final String FORMAT_TENSORFLOW_SAVEDMODEL = "tensorflow-savedmodel";
    public static final String FORMAT_TENSORFLOW_GRAPHDEF = "tensorflow-graphdef";
    public static final String FORMAT_ONNX = "onnx";
    public static final String FORMAT_PYTORCH = "pytorch";
    public static final String FORMAT_XGBOOST = "xgboost";

    // Metric names
    public static final String METRIC_REQUESTS_TOTAL = "ml_inference.requests_total";
    public static final String METRIC_SUCCESS_TOTAL = "ml_inference.success_total";
    public static final String METRIC_FAILURES_TOTAL = "ml_inference.failures_total";
    public static final String METRIC_LATENCY_MS = "ml_inference.latency_ms";
    public static final String METRIC_CACHE_HITS = "ml_inference.cache_hits";
    public static final String METRIC_CACHE_MISSES = "ml_inference.cache_misses";

    private OtterStreamSQLConstants() {
        // Prevent instantiation
    }
}
