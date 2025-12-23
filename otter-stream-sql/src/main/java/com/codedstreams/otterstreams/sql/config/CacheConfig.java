package com.codedstreams.otterstreams.sql.config;

import java.io.Serializable;
import java.time.Duration;
import java.util.Objects;

/**
 * Configuration for model and result caching in SQL inference.
 *
 * <p>Controls the caching behavior for both loaded models and inference results
 * to improve performance and reduce redundant computations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class CacheConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    private final boolean enabled;
    private final int maxSize;
    private final long ttlMs;
    private final boolean resultCacheEnabled;
    private final int resultCacheMaxSize;
    private final long resultCacheTtlMs;

    private CacheConfig(Builder builder) {
        this.enabled = builder.enabled;
        this.maxSize = builder.maxSize;
        this.ttlMs = builder.ttlMs;
        this.resultCacheEnabled = builder.resultCacheEnabled;
        this.resultCacheMaxSize = builder.resultCacheMaxSize;
        this.resultCacheTtlMs = builder.resultCacheTtlMs;
    }

    public boolean isEnabled() { return enabled; }
    public int getMaxSize() { return maxSize; }
    public long getTtlMs() { return ttlMs; }
    public boolean isResultCacheEnabled() { return resultCacheEnabled; }
    public int getResultCacheMaxSize() { return resultCacheMaxSize; }
    public long getResultCacheTtlMs() { return resultCacheTtlMs; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean enabled = true;
        private int maxSize = 100;
        private long ttlMs = Duration.ofMinutes(30).toMillis();
        private boolean resultCacheEnabled = false;
        private int resultCacheMaxSize = 10000;
        private long resultCacheTtlMs = Duration.ofMinutes(5).toMillis();

        public Builder enabled(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public Builder maxSize(int maxSize) {
            this.maxSize = maxSize;
            return this;
        }

        public Builder ttl(Duration ttl) {
            this.ttlMs = ttl.toMillis();
            return this;
        }

        public Builder ttlMs(long ttlMs) {
            this.ttlMs = ttlMs;
            return this;
        }

        public Builder resultCacheEnabled(boolean enabled) {
            this.resultCacheEnabled = enabled;
            return this;
        }

        public Builder resultCacheMaxSize(int size) {
            this.resultCacheMaxSize = size;
            return this;
        }

        public Builder resultCacheTtl(Duration ttl) {
            this.resultCacheTtlMs = ttl.toMillis();
            return this;
        }

        public CacheConfig build() {
            return new CacheConfig(this);
        }
    }

    @Override
    public String toString() {
        return "CacheConfig{" +
                "enabled=" + enabled +
                ", maxSize=" + maxSize +
                ", ttlMs=" + ttlMs +
                ", resultCacheEnabled=" + resultCacheEnabled +
                '}';
    }
}