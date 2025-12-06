package com.codedstream.otterstream.inference.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.ConcurrentHashMap;

public class MetricsCollector {
    private final MeterRegistry meterRegistry;
    private final ConcurrentHashMap<String, InferenceMetrics> metricsMap;

    public MetricsCollector() {
        this.meterRegistry = new SimpleMeterRegistry();
        this.metricsMap = new ConcurrentHashMap<>();
    }

    public MetricsCollector(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.metricsMap = new ConcurrentHashMap<>();
    }

    public InferenceMetrics getMetrics(String modelId) {
        return metricsMap.computeIfAbsent(modelId,
                id -> new InferenceMetrics(meterRegistry, id));
    }

    public MeterRegistry getMeterRegistry() {
        return meterRegistry;
    }
}

