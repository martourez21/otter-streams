package com.codedstreams.otterstreams.sql.config;

import com.codedstreams.otterstreams.sql.util.ValidationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Validates SQL inference configurations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ConfigurationValidator {
    private static final Logger LOG = LoggerFactory.getLogger(ConfigurationValidator.class);

    /**
     * Validates SQL inference configuration.
     */
    public static ValidationResult validate(SqlInferenceConfig config) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        // Validate model name
        if (config.getModelName() == null || config.getModelName().trim().isEmpty()) {
            errors.add("Model name is required");
        }

        // Validate model source
        if (config.getModelSource() == null) {
            errors.add("Model source configuration is required");
        } else {
            validateModelSource(config.getModelSource(), errors, warnings);
        }

        // Validate batch configuration
        if (config.getBatchSize() < 1) {
            errors.add("Batch size must be at least 1");
        } else if (config.getBatchSize() > 1000) {
            warnings.add("Batch size > 1000 may cause memory issues");
        }

        if (config.getBatchTimeoutMs() < 1) {
            errors.add("Batch timeout must be positive");
        }

        // Validate async configuration
        if (config.isAsyncEnabled()) {
            if (config.getAsyncTimeoutMs() < 100) {
                warnings.add("Async timeout < 100ms may be too aggressive");
            }
        }

        // Validate retry configuration
        if (config.getMaxRetries() < 0) {
            errors.add("Max retries cannot be negative");
        } else if (config.getMaxRetries() > 10) {
            warnings.add("Max retries > 10 may cause excessive delays");
        }

        if (config.getRetryBackoffMs() < 10) {
            warnings.add("Retry backoff < 10ms may be too aggressive");
        }

        return new ValidationResult(errors, warnings);
    }

    private static void validateModelSource(ModelSourceConfig source,
                                            List<String> errors,
                                            List<String> warnings) {
        // Validate model path
        if (!ValidationUtils.validateModelPath(source.getModelPath())) {
            errors.add("Invalid model path: " + source.getModelPath());
        }

        // Validate authentication for remote sources
        if (source.getSourceType() == ModelSourceConfig.SourceType.S3 ||
                source.getSourceType() == ModelSourceConfig.SourceType.MINIO) {
            if (!source.hasAuthentication()) {
                warnings.add("No authentication configured for remote storage");
            }
        }

        // Validate timeouts
        if (source.getConnectionTimeoutMs() < 1000) {
            warnings.add("Connection timeout < 1s may be too short");
        }
    }

    /**
     * Result of configuration validation.
     */
    public static class ValidationResult {
        private final List<String> errors;
        private final List<String> warnings;

        public ValidationResult(List<String> errors, List<String> warnings) {
            this.errors = List.copyOf(errors);
            this.warnings = List.copyOf(warnings);
        }

        public boolean isValid() {
            return errors.isEmpty();
        }

        public List<String> getErrors() { return errors; }
        public List<String> getWarnings() { return warnings; }

        public void logResults(Logger logger) {
            if (!errors.isEmpty()) {
                logger.error("Configuration validation errors: {}", errors);
            }
            if (!warnings.isEmpty()) {
                logger.warn("Configuration validation warnings: {}", warnings);
            }
        }
    }
}