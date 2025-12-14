package com.codedstream.otterstream.inference.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.DistributionSummary;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * Collects and records metrics for ML inference operations.
 *
 * <p>Tracks key performance indicators including:
 * <ul>
 *   <li>Request counts (total, success, failure)</li>
 *   <li>Latency distribution</li>
 *   <li>Cache hit/miss rates</li>
 *   <li>Batch sizes</li>
 * </ul>
 *
 * <h2>Integration with Monitoring:</h2>
 * <p>Metrics are exported via Micrometer and can be viewed in:
 * <ul>
 *   <li>Prometheus + Grafana</li>
 *   <li>InfluxDB</li>
 *   <li>DataDog</li>
 *   <li>New Relic</li>
 *   <li>Any Micrometer-compatible backend</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Setup with Prometheus
 * PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
 * InferenceMetrics metrics = new InferenceMetrics(registry, "fraud-detector");
 *
 * // Record inference
 * long startTime = System.currentTimeMillis();
 * InferenceResult result = engine.infer(inputs);
 * long duration = System.currentTimeMillis() - startTime;
 * metrics.recordInference(duration, result.isSuccess());
 *
 * // Record cache hit
 * if (cachedResult != null) {
 *     metrics.recordCacheHit();
 * }
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see MetricsCollector
 */
public class InferenceMetrics {
    private final MeterRegistry meterRegistry;
    private final String modelId;
    private final ConcurrentHashMap<String, Counter> counters;
    private final ConcurrentHashMap<String, Timer> timers;
    private final ConcurrentHashMap<String, DistributionSummary> summaries;

    /**
     * Constructs inference metrics collector.
     *
     * @param meterRegistry the meter registry for metric export
     * @param modelId identifier for this model (used in metric tags)
     */
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

    /**
     * Records an inference operation with duration and outcome.
     *
     * @param durationMs inference duration in milliseconds
     * @param success whether inference succeeded
     */
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

    /**
     * Records a cache hit event.
     */
    public void recordCacheHit() {
        getCounter("cache_hits_total").increment();
    }

    /**
     * Records a cache miss event.
     */
    public void recordCacheMiss() {
        getCounter("cache_misses_total").increment();
    }

    /**
     * Records the size of a batch inference operation.
     *
     * @param batchSize number of records in the batch
     */
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