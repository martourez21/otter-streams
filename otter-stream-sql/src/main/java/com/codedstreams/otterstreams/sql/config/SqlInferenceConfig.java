package com.codedstreams.otterstreams.sql.config;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.model.ModelFormat;

import java.io.Serializable;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration for SQL-based ML inference operations.
 *
 * <p>This class extends the core {@link InferenceConfig} with SQL-specific settings
 * including model source configuration, caching policies, and SQL DDL options.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class SqlInferenceConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    // Model configuration
    private final String modelName;
    private final ModelSourceConfig modelSource;  // FIXED: Changed from ModelSource to ModelSourceConfig
    private final String modelVersion;

    // Performance configuration
    private final int batchSize;
    private final long batchTimeoutMs;
    private final boolean asyncEnabled;
    private final long asyncTimeoutMs;

    // Cache configuration
    private final CacheConfig cacheConfig;

    // Retry configuration
    private final int maxRetries;
    private final long retryBackoffMs;

    // Additional options
    private final Map<String, String> additionalOptions;

    /**
     * Private constructor - use Builder.
     */
    private SqlInferenceConfig(Builder builder) {
        this.modelName = Objects.requireNonNull(builder.modelName, "modelName is required");
        this.modelSource = Objects.requireNonNull(builder.modelSource, "modelSource is required");
        this.modelVersion = builder.modelVersion;
        this.batchSize = builder.batchSize;
        this.batchTimeoutMs = builder.batchTimeoutMs;
        this.asyncEnabled = builder.asyncEnabled;
        this.asyncTimeoutMs = builder.asyncTimeoutMs;
        this.cacheConfig = builder.cacheConfig;
        this.maxRetries = builder.maxRetries;
        this.retryBackoffMs = builder.retryBackoffMs;
        this.additionalOptions = Map.copyOf(builder.additionalOptions);
    }

    /**
     * Creates configuration from SQL DDL options.
     *
     * @param options map of option key-value pairs from WITH clause
     * @return configured SqlInferenceConfig
     * @throws IllegalArgumentException if required options are missing
     */
    public static SqlInferenceConfig fromOptions(Map<String, String> options) {
        Builder builder = builder();

        // Required options
        String modelName = options.get("model.name");
        if (modelName == null) {
            throw new IllegalArgumentException("Required option 'model.name' is missing");
        }
        builder.modelName(modelName);

        String modelPath = options.get("model.path");
        if (modelPath == null) {
            throw new IllegalArgumentException("Required option 'model.path' is missing");
        }

        String modelFormatStr = options.getOrDefault("model.format", "tensorflow-savedmodel");
        ModelFormat format = parseModelFormat(modelFormatStr);

        // Build model source
        ModelSourceConfig modelSource = ModelSourceConfig.builder()
                .modelPath(modelPath)
                .modelFormat(format)
                .build();
        builder.modelSource(modelSource);

        // Optional options
        if (options.containsKey("model.version")) {
            builder.modelVersion(options.get("model.version"));
        }

        if (options.containsKey("batch.size")) {
            builder.batchSize(Integer.parseInt(options.get("batch.size")));
        }

        if (options.containsKey("batch.timeout-ms")) {
            builder.batchTimeoutMs(Long.parseLong(options.get("batch.timeout-ms")));
        }

        if (options.containsKey("async.enabled")) {
            builder.asyncEnabled(Boolean.parseBoolean(options.get("async.enabled")));
        }

        if (options.containsKey("async.timeout-ms")) {
            builder.asyncTimeoutMs(Long.parseLong(options.get("async.timeout-ms")));
        }

        // Cache configuration
        boolean cacheEnabled = Boolean.parseBoolean(options.getOrDefault("cache.enabled", "true"));
        int cacheMaxSize = Integer.parseInt(options.getOrDefault("cache.max-size", "100"));
        long cacheTtlMinutes = Long.parseLong(options.getOrDefault("cache.ttl-minutes", "30"));

        CacheConfig cacheConfig = CacheConfig.builder()
                .enabled(cacheEnabled)
                .maxSize(cacheMaxSize)
                .ttl(Duration.ofMinutes(cacheTtlMinutes))
                .build();
        builder.cacheConfig(cacheConfig);

        // Retry configuration
        if (options.containsKey("retry.max-attempts")) {
            builder.maxRetries(Integer.parseInt(options.get("retry.max-attempts")));
        }

        if (options.containsKey("retry.backoff-ms")) {
            builder.retryBackoffMs(Long.parseLong(options.get("retry.backoff-ms")));
        }

        // Store all additional options
        builder.additionalOptions(new HashMap<>(options));

        return builder.build();
    }

    /**
     * Converts this config to core InferenceConfig for compatibility.
     *
     * @return InferenceConfig instance
     */
    public InferenceConfig toCoreConfig() {
        ModelConfig modelConfig = ModelConfig.builder()
                .modelId(modelName)
                .modelPath(modelSource.getModelPath())
                .format(modelSource.getModelFormat())
                .modelVersion(modelVersion != null ? modelVersion : "latest")
                .build();

        return InferenceConfig.builder()
                .modelConfig(modelConfig)
                .batchSize(batchSize)
                .timeoutMs(asyncTimeoutMs)
                .maxRetries(maxRetries)
                .enableMetrics(true)
                .build();
    }

    /**
     * Parses model format string to ModelFormat enum.
     * Uses the EXACT enum values from core ModelFormat.
     */
    private static ModelFormat parseModelFormat(String format) {
        switch (format.toLowerCase()) {
            case "tensorflow-savedmodel":
            case "savedmodel":
                return ModelFormat.TENSORFLOW_SAVEDMODEL;
            case "tensorflow-graphdef":
            case "graphdef":
                return ModelFormat.TENSORFLOW_GRAPHDEF;
            case "onnx":
                return ModelFormat.ONNX;
            case "pytorch":
            case "torchscript":
            case "pytorch-torchscript":
                return ModelFormat.PYTORCH_TORCHSCRIPT;
            case "xgboost":
            case "xgboost-binary":
                return ModelFormat.XGBOOST_BINARY;
            case "xgboost-json":
                return ModelFormat.XGBOOST_JSON;
            case "pmml":
                return ModelFormat.PMML;
            case "remote-http":
            case "http":
                return ModelFormat.REMOTE_HTTP;
            case "remote-grpc":
            case "grpc":
                return ModelFormat.REMOTE_GRPC;
            case "sagemaker":
                return ModelFormat.SAGEMAKER;
            case "vertex-ai":
            case "vertexai":
                return ModelFormat.VERTEX_AI;
            case "azure-ml":
            case "azureml":
                return ModelFormat.AZURE_ML;
            default:
                throw new IllegalArgumentException("Unsupported model format: " + format);
        }
    }

    // Getters
    public String getModelName() { return modelName; }
    public ModelSourceConfig getModelSource() { return modelSource; }
    public String getModelVersion() { return modelVersion; }
    public int getBatchSize() { return batchSize; }
    public long getBatchTimeoutMs() { return batchTimeoutMs; }
    public boolean isAsyncEnabled() { return asyncEnabled; }
    public long getAsyncTimeoutMs() { return asyncTimeoutMs; }
    public CacheConfig getCacheConfig() { return cacheConfig; }
    public int getMaxRetries() { return maxRetries; }
    public long getRetryBackoffMs() { return retryBackoffMs; }
    public Map<String, String> getAdditionalOptions() { return additionalOptions; }

    /**
     * Builder for SqlInferenceConfig with sensible defaults.
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelName;
        private ModelSourceConfig modelSource;
        private String modelVersion = "latest";
        private int batchSize = 1;
        private long batchTimeoutMs = 50;
        private boolean asyncEnabled = false;
        private long asyncTimeoutMs = 5000;
        private CacheConfig cacheConfig = CacheConfig.builder().build();
        private int maxRetries = 3;
        private long retryBackoffMs = 100;
        private Map<String, String> additionalOptions = new HashMap<>();

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder modelSource(ModelSourceConfig modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelPath(String modelPath) {
            if (this.modelSource == null) {
                this.modelSource = ModelSourceConfig.builder()
                        .modelPath(modelPath)
                        .modelFormat(ModelFormat.TENSORFLOW_SAVEDMODEL)
                        .build();
            }
            return this;
        }

        public Builder modelFormat(ModelFormat format) {
            if (this.modelSource != null) {
                this.modelSource = ModelSourceConfig.builder()
                        .modelPath(this.modelSource.getModelPath())
                        .modelFormat(format)
                        .build();
            }
            return this;
        }

        public Builder modelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder batchTimeoutMs(long batchTimeoutMs) {
            this.batchTimeoutMs = batchTimeoutMs;
            return this;
        }

        public Builder batchTimeout(Duration timeout) {
            this.batchTimeoutMs = timeout.toMillis();
            return this;
        }

        public Builder asyncEnabled(boolean asyncEnabled) {
            this.asyncEnabled = asyncEnabled;
            return this;
        }

        public Builder asyncTimeoutMs(long asyncTimeoutMs) {
            this.asyncTimeoutMs = asyncTimeoutMs;
            return this;
        }

        public Builder asyncTimeout(Duration timeout) {
            this.asyncTimeoutMs = timeout.toMillis();
            return this;
        }

        public Builder cacheConfig(CacheConfig cacheConfig) {
            this.cacheConfig = cacheConfig;
            return this;
        }

        public Builder cacheEnabled(boolean enabled) {
            this.cacheConfig = CacheConfig.builder()
                    .enabled(enabled)
                    .maxSize(this.cacheConfig.getMaxSize())
                    .ttl(Duration.ofMillis(this.cacheConfig.getTtlMs()))
                    .build();
            return this;
        }

        public Builder cacheMaxSize(int maxSize) {
            this.cacheConfig = CacheConfig.builder()
                    .enabled(this.cacheConfig.isEnabled())
                    .maxSize(maxSize)
                    .ttl(Duration.ofMillis(this.cacheConfig.getTtlMs()))
                    .build();
            return this;
        }

        public Builder cacheTtl(Duration ttl) {
            this.cacheConfig = CacheConfig.builder()
                    .enabled(this.cacheConfig.isEnabled())
                    .maxSize(this.cacheConfig.getMaxSize())
                    .ttl(ttl)
                    .build();
            return this;
        }

        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public Builder retryBackoffMs(long retryBackoffMs) {
            this.retryBackoffMs = retryBackoffMs;
            return this;
        }

        public Builder additionalOptions(Map<String, String> options) {
            this.additionalOptions = new HashMap<>(options);
            return this;
        }

        public Builder addOption(String key, String value) {
            this.additionalOptions.put(key, value);
            return this;
        }

        public SqlInferenceConfig build() {
            return new SqlInferenceConfig(this);
        }
    }

    @Override
    public String toString() {
        return "SqlInferenceConfig{" +
                "modelName='" + modelName + '\'' +
                ", modelSource=" + modelSource +
                ", modelVersion='" + modelVersion + '\'' +
                ", batchSize=" + batchSize +
                ", batchTimeoutMs=" + batchTimeoutMs +
                ", asyncEnabled=" + asyncEnabled +
                ", asyncTimeoutMs=" + asyncTimeoutMs +
                ", cacheConfig=" + cacheConfig +
                ", maxRetries=" + maxRetries +
                ", retryBackoffMs=" + retryBackoffMs +
                '}';
    }
}