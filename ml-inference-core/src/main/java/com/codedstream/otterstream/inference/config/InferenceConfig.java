package com.codedstream.otterstream.inference.config;

import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * Comprehensive configuration for ML inference operations in Apache Flink streams.
 *
 * <p>This class configures all aspects of inference including model settings,
 * performance tuning (batching, timeouts), retry policies, and monitoring.
 *
 * <h2>Key Configuration Areas:</h2>
 * <ul>
 *   <li><b>Model:</b> Which model to use and how to load it</li>
 *   <li><b>Performance:</b> Batch size and timeout settings</li>
 *   <li><b>Reliability:</b> Retry logic for failed inferences</li>
 *   <li><b>Monitoring:</b> Metrics collection enablement</li>
 *   <li><b>Engine:</b> Custom engine-specific options</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Configure TensorFlow inference
 * ModelConfig modelConfig = ModelConfig.builder()
 *     .modelId("sentiment-model")
 *     .modelPath("/models/sentiment.pb")
 *     .format(ModelFormat.TENSORFLOW_SAVEDMODEL)
 *     .build();
 *
 * InferenceConfig config = InferenceConfig.builder()
 *     .modelConfig(modelConfig)
 *     .batchSize(32)                    // Process 32 records at once
 *     .timeout(Duration.ofSeconds(30))   // 30 second timeout
 *     .maxRetries(3)                     // Retry up to 3 times
 *     .enableMetrics(true)               // Collect performance metrics
 *     .build();
 *
 * // Use in Flink stream
 * DataStream<Prediction> predictions = input
 *     .async(new AsyncModelInferenceFunction<>(config, engineFactory));
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see ModelConfig
 */
public class InferenceConfig {
    private final ModelConfig modelConfig;
    private final int batchSize;
    private final long timeoutMs;
    private final int maxRetries;
    private final boolean enableMetrics;
    private final Map<String, Object> engineOptions;

    /**
     * Constructs inference configuration.
     *
     * @param modelConfig model configuration
     * @param batchSize number of records to batch together
     * @param timeoutMs inference timeout in milliseconds
     * @param maxRetries maximum retry attempts for failed inferences
     * @param enableMetrics whether to collect metrics
     * @param engineOptions engine-specific configuration options
     */
    public InferenceConfig(ModelConfig modelConfig, int batchSize, long timeoutMs,
                           int maxRetries, boolean enableMetrics, Map<String, Object> engineOptions) {
        this.modelConfig = Objects.requireNonNull(modelConfig);
        this.batchSize = batchSize;
        this.timeoutMs = timeoutMs;
        this.maxRetries = maxRetries;
        this.enableMetrics = enableMetrics;
        this.engineOptions = Map.copyOf(engineOptions);
    }

    /**
     * Creates a new builder for InferenceConfig.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    public ModelConfig getModelConfig() { return modelConfig; }
    public int getBatchSize() { return batchSize; }
    public long getTimeoutMs() { return timeoutMs; }
    public int getMaxRetries() { return maxRetries; }
    public boolean isEnableMetrics() { return enableMetrics; }
    public Map<String, Object> getEngineOptions() { return engineOptions; }

    /**
     * Builder for creating InferenceConfig instances with sensible defaults.
     */
    public static class Builder {
        private ModelConfig modelConfig;
        private int batchSize = 1;
        private long timeoutMs = TimeUnit.SECONDS.toMillis(30);
        private int maxRetries = 3;
        private boolean enableMetrics = true;
        private Map<String, Object> engineOptions = Map.of();

        /**
         * Sets the model configuration.
         *
         * @param modelConfig model configuration
         * @return this builder
         */
        public Builder modelConfig(ModelConfig modelConfig) {
            this.modelConfig = modelConfig;
            return this;
        }

        /**
         * Sets the batch size for inference operations.
         * <p>Larger batch sizes improve throughput but increase latency.
         * Default is 1 (no batching).
         *
         * @param batchSize number of records to batch (must be > 0)
         * @return this builder
         */
        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        /**
         * Sets the inference timeout in milliseconds.
         *
         * @param timeoutMs timeout in milliseconds
         * @return this builder
         */
        public Builder timeoutMs(long timeoutMs) {
            this.timeoutMs = timeoutMs;
            return this;
        }

        /**
         * Sets the inference timeout using Duration.
         *
         * @param duration timeout duration
         * @return this builder
         */
        public Builder timeout(Duration duration) {
            this.timeoutMs = duration.toMillis();
            return this;
        }

        /**
         * Sets maximum retry attempts for failed inferences.
         *
         * @param maxRetries maximum retries (0 for no retries)
         * @return this builder
         */
        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        /**
         * Enables or disables metrics collection.
         * <p>When enabled, collects latency, throughput, and error metrics.
         *
         * @param enableMetrics true to enable metrics
         * @return this builder
         */
        public Builder enableMetrics(boolean enableMetrics) {
            this.enableMetrics = enableMetrics;
            return this;
        }

        /**
         * Sets engine-specific options.
         * <p>Options vary by engine (TensorFlow, ONNX, PyTorch, etc.)
         *
         * @param engineOptions map of option name-value pairs
         * @return this builder
         */
        public Builder engineOptions(Map<String, Object> engineOptions) {
            this.engineOptions = Map.copyOf(engineOptions);
            return this;
        }

        /**
         * Builds the InferenceConfig instance.
         *
         * @return configured InferenceConfig
         * @throws NullPointerException if modelConfig is not set
         */
        public InferenceConfig build() {
            return new InferenceConfig(modelConfig, batchSize, timeoutMs, maxRetries,
                    enableMetrics, engineOptions);
        }
    }
}