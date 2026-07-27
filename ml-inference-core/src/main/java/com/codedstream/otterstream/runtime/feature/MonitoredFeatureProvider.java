package com.codedstream.otterstream.runtime.feature;

import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.LongAdder;

/**
 * Wraps any {@link FeatureProvider} with call-level monitoring — latency and error rate — the
 * "feature monitoring" piece of the Feature Store Integration roadmap item. Purely a decorator:
 * it delegates every {@code fetch()} call to the wrapped provider unchanged and adds no caching,
 * retry, or other behavior — composability without surprises, matching how
 * {@code otter-stream-rules}'s connectors stay single-purpose.
 *
 * <p>Uses {@link LongAdder} for the counters, not {@code AtomicLong} — the same reasoning as
 * {@code DefaultRuleEngine}'s metrics (write-heavy under concurrent fetches, read-light on
 * snapshot polling).
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * FeatureProvider redis = new RedisFeatureProvider("localhost", 6379, "features:user:");
 * FeatureProvider monitored = new MonitoredFeatureProvider(redis);
 *
 * Map<String, Object> values = monitored.fetch("42", List.of("age", "country"));
 * FeatureMetricsSnapshot snapshot = monitored.getMetrics();
 * }</pre>
 *
 * @since 0.1.0
 */
public class MonitoredFeatureProvider implements FeatureProvider {

    private final FeatureProvider delegate;
    private final LongAdder totalFetches = new LongAdder();
    private final LongAdder errorCount = new LongAdder();
    private final LongAdder totalLatencyMicros = new LongAdder();

    public MonitoredFeatureProvider(FeatureProvider delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate cannot be null");
    }

    @Override
    public String getProviderId() {
        return delegate.getProviderId();
    }

    @Override
    public Map<String, Object> fetch(String entityId, List<String> featureNames) throws Exception {
        long start = System.nanoTime();
        totalFetches.increment();
        try {
            return delegate.fetch(entityId, featureNames);
        } catch (Exception e) {
            errorCount.increment();
            throw e;
        } finally {
            totalLatencyMicros.add((System.nanoTime() - start) / 1000);
        }
    }

    public FeatureMetricsSnapshot getMetrics() {
        long fetches = totalFetches.sum();
        long avgLatency = fetches == 0 ? 0 : totalLatencyMicros.sum() / fetches;
        return new FeatureMetricsSnapshot(
                delegate.getProviderId(), fetches, errorCount.sum(), avgLatency, System.currentTimeMillis());
    }
}
