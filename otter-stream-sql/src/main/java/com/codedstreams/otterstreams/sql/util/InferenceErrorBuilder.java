package com.codedstreams.otterstreams.sql.util;

/**
 * Builder for user-friendly error messages.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class InferenceErrorBuilder {

    public static String buildModelNotFoundError(String modelName) {
        return String.format(
                "Model '%s' not found. Please ensure:\n" +
                        "1. Model is registered using ModelRegistrationManager\n" +
                        "2. Model path is accessible from all TaskManagers\n" +
                        "3. Required credentials are configured\n" +
                        "4. Model format is supported such as (tensorflow-savedmodel, tensorflow-graphdef)",
                modelName
        );
    }

    public static String buildInferenceTimeoutError(String modelName, long timeoutMs) {
        return String.format(
                "Inference timeout for model '%s' after %dms. Consider:\n" +
                        "1. Increasing timeout: 'async.timeout-ms' = '%d'\n" +
                        "2. Enabling async mode: 'async.enabled' = 'true'\n" +
                        "3. Reducing batch size\n" +
                        "4. Checking model server health",
                modelName, timeoutMs, timeoutMs * 2
        );
    }

    public static String buildConfigurationError(String option, String issue) {
        return String.format(
                "Configuration error for option '%s': %s\n" +
                        "Please check the configuration reference in documentation.",
                option, issue
        );
    }

    public static String buildFeatureExtractionError(String featureJson, Exception cause) {
        return String.format(
                "Failed to extract features from JSON: %s\n" +
                        "Error: %s\n" +
                        "Expected format: JSON_OBJECT('feature1', value1, 'feature2', value2, ...)",
                featureJson.substring(0, Math.min(100, featureJson.length())),
                cause.getMessage()
        );
    }
}

