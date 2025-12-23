package com.codedstreams.otterstreams.sql.runtime;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Collects and tracks inference metrics per model.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class InferenceMetricsCollector {
    private static final Logger LOG = LoggerFactory.getLogger(InferenceMetricsCollector.class);
    private static final InferenceMetricsCollector INSTANCE = new InferenceMetricsCollector();

    private final ConcurrentHashMap<String, ModelMetrics> metricsMap;

    private InferenceMetricsCollector() {
        this.metricsMap = new ConcurrentHashMap<>();
    }

    public static InferenceMetricsCollector getInstance() {
        return INSTANCE;
    }

    public void recordInference(String modelName, long latencyMs, boolean success) {
        ModelMetrics metrics = metricsMap.computeIfAbsent(modelName, k -> new ModelMetrics());
        metrics.recordInference(latencyMs, success);
    }

    public void recordCacheHit(String modelName) {
        ModelMetrics metrics = metricsMap.computeIfAbsent(modelName, k -> new ModelMetrics());
        metrics.recordCacheHit();
    }

    public void recordCacheMiss(String modelName) {
        ModelMetrics metrics = metricsMap.computeIfAbsent(modelName, k -> new ModelMetrics());
        metrics.recordCacheMiss();
    }

    public ModelMetrics getMetrics(String modelName) {
        return metricsMap.get(modelName);
    }

    public void reset(String modelName) {
        metricsMap.remove(modelName);
    }

    public void resetAll() {
        metricsMap.clear();
    }

    /**
     * Metrics for a single model.
     */
    public static class ModelMetrics {
        private final LongAdder totalInferences = new LongAdder();
        private final LongAdder successfulInferences = new LongAdder();
        private final LongAdder failedInferences = new LongAdder();
        private final LongAdder totalLatencyMs = new LongAdder();
        private final AtomicLong minLatencyMs = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong maxLatencyMs = new AtomicLong(0);
        private final LongAdder cacheHits = new LongAdder();
        private final LongAdder cacheMisses = new LongAdder();

        void recordInference(long latencyMs, boolean success) {
            totalInferences.increment();
            if (success) {
                successfulInferences.increment();
            } else {
                failedInferences.increment();
            }

            totalLatencyMs.add(latencyMs);
            updateMinLatency(latencyMs);
            updateMaxLatency(latencyMs);
        }

        void recordCacheHit() {
            cacheHits.increment();
        }

        void recordCacheMiss() {
            cacheMisses.increment();
        }

        private void updateMinLatency(long latency) {
            long current;
            do {
                current = minLatencyMs.get();
                if (latency >= current) return;
            } while (!minLatencyMs.compareAndSet(current, latency));
        }

        private void updateMaxLatency(long latency) {
            long current;
            do {
                current = maxLatencyMs.get();
                if (latency <= current) return;
            } while (!maxLatencyMs.compareAndSet(current, latency));
        }

        public long getTotalInferences() { return totalInferences.sum(); }
        public long getSuccessfulInferences() { return successfulInferences.sum(); }
        public long getFailedInferences() { return failedInferences.sum(); }
        public double getSuccessRate() {
            long total = getTotalInferences();
            return total > 0 ? (double) getSuccessfulInferences() / total : 0.0;
        }
        public double getAverageLatencyMs() {
            long total = getTotalInferences();
            return total > 0 ? (double) totalLatencyMs.sum() / total : 0.0;
        }
        public long getMinLatencyMs() {
            long min = minLatencyMs.get();
            return min == Long.MAX_VALUE ? 0 : min;
        }
        public long getMaxLatencyMs() { return maxLatencyMs.get(); }
        public long getCacheHits() { return cacheHits.sum(); }
        public long getCacheMisses() { return cacheMisses.sum(); }
        public double getCacheHitRate() {
            long total = getCacheHits() + getCacheMisses();
            return total > 0 ? (double) getCacheHits() / total : 0.0;
        }

        @Override
        public String toString() {
            return String.format(
                    "ModelMetrics{total=%d, success=%d, failed=%d, successRate=%.2f%%, " +
                            "avgLatency=%.2fms, minLatency=%dms, maxLatency=%dms, cacheHitRate=%.2f%%}",
                    getTotalInferences(), getSuccessfulInferences(), getFailedInferences(),
                    getSuccessRate() * 100, getAverageLatencyMs(), getMinLatencyMs(),
                    getMaxLatencyMs(), getCacheHitRate() * 100
            );
        }
    }
}
