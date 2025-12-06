package com.codedstream.otterstream.inference.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.DistributionSummary;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class InferenceMetrics {
    private final MeterRegistry meterRegistry;
    private final String modelId;
    private final ConcurrentHashMap<String, Counter> counters;
    private final ConcurrentHashMap<String, Timer> timers;
    private final ConcurrentHashMap<String, DistributionSummary> summaries;

    public InferenceMetrics(MeterRegistry meterRegistry, String modelId) {
        this.meterRegistry = meterRegistry;
        this.modelId = modelId;
        this.counters = new ConcurrentHashMap<>();
        this.timers = new ConcurrentHashMap<>();
        this.summaries = new ConcurrentHashMap<>();
        initializeBaseMetrics();
    }

    private void initializeBaseMetrics() {
        getCounter("inference_requests_total");
        getCounter("inference_success_total");
        getCounter("inference_failures_total");
        getTimer("inference_duration_ms");
        getDistributionSummary("inference_latency_ms");
    }

    public void recordInference(long durationMs, boolean success) {
        getCounter("inference_requests_total").increment();

        if (success) {
            getCounter("inference_success_total").increment();
        } else {
            getCounter("inference_failures_total").increment();
        }

        getTimer("inference_duration_ms").record(durationMs, TimeUnit.MILLISECONDS);
        getDistributionSummary("inference_latency_ms").record(durationMs);
    }

    public void recordCacheHit() {
        getCounter("cache_hits_total").increment();
    }

    public void recordCacheMiss() {
        getCounter("cache_misses_total").increment();
    }

    public void recordBatchSize(int batchSize) {
        getDistributionSummary("batch_size").record(batchSize);
    }

    private Counter getCounter(String name) {
        return counters.computeIfAbsent(name,
                key -> Counter.builder(key)
                        .tag("model_id", modelId)
                        .register(meterRegistry));
    }

    private Timer getTimer(String name) {
        return timers.computeIfAbsent(name,
                key -> Timer.builder(name)
                        .tag("model_id", modelId)
                        .register(meterRegistry));
    }

    private DistributionSummary getDistributionSummary(String name) {
        return summaries.computeIfAbsent(name,
                key -> DistributionSummary.builder(name)
                        .tag("model_id", modelId)
                        .register(meterRegistry));
    }
}
