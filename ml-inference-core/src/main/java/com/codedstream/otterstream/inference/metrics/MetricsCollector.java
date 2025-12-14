package com.codedstream.otterstream.inference.metrics;

import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central collector managing metrics for multiple models.
 *
 * <p>Maintains separate {@link InferenceMetrics} instances for each model,
 * all reporting to a shared {@link MeterRegistry}.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create collector with Prometheus registry
 * PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
 * MetricsCollector collector = new MetricsCollector(registry);
 *
 * // Get metrics for specific models
 * InferenceMetrics fraudMetrics = collector.getMetrics("fraud-detector");
 * InferenceMetrics recommendMetrics = collector.getMetrics("recommender");
 *
 * // Use in inference functions
 * fraudMetrics.recordInference(latency, success);
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class MetricsCollector {
    private final MeterRegistry meterRegistry;
    private final ConcurrentHashMap<String, InferenceMetrics> metricsMap;

    /**
     * Creates collector with default SimpleMeterRegistry.
     */
    public MetricsCollector() {
        this.meterRegistry = new SimpleMeterRegistry();
        this.metricsMap = new ConcurrentHashMap<>();
    }

    /**
     * Creates collector with custom meter registry.
     *
     * @param meterRegistry registry for exporting metrics
     */
    public MetricsCollector(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.metricsMap = new ConcurrentHashMap<>();
    }

    /**
     * Gets or creates metrics instance for a specific model.
     *
     * @param modelId unique identifier for the model
     * @return metrics instance for this model
     */
    public InferenceMetrics getMetrics(String modelId) {
        return metricsMap.computeIfAbsent(modelId,
                id -> new InferenceMetrics(meterRegistry, id));
    }

    /**
     * Gets the underlying meter registry.
     *
     * @return the meter registry used by all metrics
     */
    public MeterRegistry getMeterRegistry() {
        return meterRegistry;
    }
}