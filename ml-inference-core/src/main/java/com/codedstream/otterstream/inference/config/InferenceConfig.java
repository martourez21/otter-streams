package com.codedstream.otterstream.inference.config;

import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

public class InferenceConfig {
    private final ModelConfig modelConfig;
    private final int batchSize;
    private final long timeoutMs;
    private final int maxRetries;
    private final boolean enableMetrics;
    private final Map<String, Object> engineOptions;

    public InferenceConfig(ModelConfig modelConfig, int batchSize, long timeoutMs,
                           int maxRetries, boolean enableMetrics, Map<String, Object> engineOptions) {
        this.modelConfig = Objects.requireNonNull(modelConfig);
        this.batchSize = batchSize;
        this.timeoutMs = timeoutMs;
        this.maxRetries = maxRetries;
        this.enableMetrics = enableMetrics;
        this.engineOptions = Map.copyOf(engineOptions);
    }

    public static Builder builder() {
        return new Builder();
    }

    public ModelConfig getModelConfig() { return modelConfig; }
    public int getBatchSize() { return batchSize; }
    public long getTimeoutMs() { return timeoutMs; }
    public int getMaxRetries() { return maxRetries; }
    public boolean isEnableMetrics() { return enableMetrics; }
    public Map<String, Object> getEngineOptions() { return engineOptions; }

    public static class Builder {
        private ModelConfig modelConfig;
        private int batchSize = 1;
        private long timeoutMs = TimeUnit.SECONDS.toMillis(30);
        private int maxRetries = 3;
        private boolean enableMetrics = true;
        private Map<String, Object> engineOptions = Map.of();

        public Builder modelConfig(ModelConfig modelConfig) {
            this.modelConfig = modelConfig;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder timeoutMs(long timeoutMs) {
            this.timeoutMs = timeoutMs;
            return this;
        }

        public Builder timeout(Duration duration) {
            this.timeoutMs = duration.toMillis();
            return this;
        }

        public Builder maxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public Builder enableMetrics(boolean enableMetrics) {
            this.enableMetrics = enableMetrics;
            return this;
        }

        public Builder engineOptions(Map<String, Object> engineOptions) {
            this.engineOptions = Map.copyOf(engineOptions);
            return this;
        }

        public InferenceConfig build() {
            return new InferenceConfig(modelConfig, batchSize, timeoutMs, maxRetries,
                    enableMetrics, engineOptions);
        }
    }
}
