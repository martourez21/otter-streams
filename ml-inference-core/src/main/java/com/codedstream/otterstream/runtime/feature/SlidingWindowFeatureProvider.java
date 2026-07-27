package com.codedstream.otterstream.runtime.feature;

import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * A {@link FeatureProvider} that computes its feature value in real time from a sliding window
 * of recently-recorded values, rather than looking one up from external storage — the
 * "real-time feature computation" piece of the Feature Store Integration roadmap item. Feed it
 * from your Flink pipeline as events arrive (e.g. one {@link #record} call per transaction
 * amount), and it answers "what's the rolling {@link AggregationFunction} for this entity over
 * the last N minutes" on every {@code fetch()} — genuinely computed on the fly, not a cached
 * lookup of a value someone else computed elsewhere.
 *
 * <p>Uses the same sliding-window-with-pruning technique as
 * {@code otter-control-plane-server}'s {@code TopologyService} on the TypeScript side: samples
 * are pruned as a side effect of new writes for the same entity (bounding that entity's own
 * memory), plus an explicit {@link #evictEntitiesIdleSince} for the caller to invoke
 * periodically to bound total memory across all entities.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * SlidingWindowFeatureProvider avgTxnAmount = new SlidingWindowFeatureProvider(
 *         "avg-transaction-amount-5m", Duration.ofMinutes(5), AggregationFunction.AVG);
 *
 * // In your Flink pipeline, as each transaction arrives:
 * avgTxnAmount.record(entityId, transaction.getAmount());
 *
 * // Feed the running average into the next inference call as a feature:
 * Map<String, Object> features = avgTxnAmount.fetch(entityId, List.of());
 * double runningAvg = (double) features.get("avg-transaction-amount-5m");
 * }</pre>
 *
 * <p><b>Cardinality caveat, stated plainly:</b> memory is bounded per entity (old samples are
 * pruned) but is <em>not</em> automatically bounded across the total number of distinct entity
 * ids ever seen — a workload with unbounded/high-cardinality entity ids (e.g. one-off session
 * ids rather than stable user/account ids) will grow this provider's memory unboundedly unless
 * you call {@link #evictEntitiesIdleSince} periodically. This is the same caller-driven-pruning
 * trade-off {@code TopologyService} makes on the Control Plane side, not an oversight.
 *
 * @since 0.1.0
 */
public class SlidingWindowFeatureProvider implements FeatureProvider {

    private record TimestampedValue(long timestampMillis, double value) {
    }

    private final String providerId;
    private final long windowMillis;
    private final AggregationFunction function;
    private final ConcurrentHashMap<String, List<TimestampedValue>> samplesByEntity = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Long> lastWriteMillisByEntity = new ConcurrentHashMap<>();

    /**
     * @param providerId a stable identifier — also used as the feature name returned by {@link #fetch}
     * @param window     how far back to aggregate
     * @param function   which aggregation to compute over the windowed samples
     */
    public SlidingWindowFeatureProvider(String providerId, Duration window, AggregationFunction function) {
        this.providerId = Objects.requireNonNull(providerId, "providerId cannot be null");
        this.windowMillis = Objects.requireNonNull(window, "window cannot be null").toMillis();
        this.function = Objects.requireNonNull(function, "function cannot be null");
    }

    /**
     * Records one new value for an entity — call this from your pipeline as events arrive.
     * Prunes this entity's own samples older than the window as a side effect, bounding its
     * memory regardless of how long the provider has been running.
     *
     * @param entityId the entity this value belongs to
     * @param value    the value to record
     */
    public void record(String entityId, double value) {
        long now = System.currentTimeMillis();
        List<TimestampedValue> samples = samplesByEntity.computeIfAbsent(entityId, id -> new CopyOnWriteArrayList<>());
        samples.add(new TimestampedValue(now, value));
        pruneEntity(samples, now);
        lastWriteMillisByEntity.put(entityId, now);
    }

    @Override
    public String getProviderId() {
        return providerId;
    }

    /**
     * Returns the current aggregate for {@code entityId} over the trailing window. Ignores
     * {@code featureNames} — this provider computes exactly one feature (itself), named after
     * {@link #getProviderId()}; a caller asking for a specific list of feature names from a
     * generic {@link FeatureProvider}-typed reference will still get this one value back under
     * that name.
     */
    @Override
    public Map<String, Object> fetch(String entityId, List<String> featureNames) {
        long now = System.currentTimeMillis();
        List<TimestampedValue> samples = samplesByEntity.get(entityId);
        if (samples == null) {
            return Map.of(providerId, defaultValue());
        }
        pruneEntity(samples, now);
        List<Double> windowed = samples.stream().map(TimestampedValue::value).toList();
        return Map.of(providerId, aggregate(windowed));
    }

    /**
     * Removes entities whose last {@link #record} call was more than {@code maxIdleMillis} ago
     * — call this periodically (e.g. on a scheduled tick in your application) to bound total
     * memory when entity cardinality is high or unbounded. See the class Javadoc's cardinality
     * caveat.
     *
     * @return how many entities were evicted
     */
    public int evictEntitiesIdleSince(long maxIdleMillis) {
        long cutoff = System.currentTimeMillis() - maxIdleMillis;
        int evicted = 0;
        for (var entry : lastWriteMillisByEntity.entrySet()) {
            if (entry.getValue() < cutoff) {
                samplesByEntity.remove(entry.getKey());
                lastWriteMillisByEntity.remove(entry.getKey());
                evicted++;
            }
        }
        return evicted;
    }

    public int getTrackedEntityCount() {
        return samplesByEntity.size();
    }

    private void pruneEntity(List<TimestampedValue> samples, long now) {
        long cutoff = now - windowMillis;
        samples.removeIf(s -> s.timestampMillis() < cutoff);
    }

    private double aggregate(List<Double> values) {
        if (values.isEmpty()) {
            return defaultValue();
        }
        return switch (function) {
            case COUNT -> values.size();
            case SUM -> values.stream().mapToDouble(Double::doubleValue).sum();
            case AVG -> values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            case MIN -> values.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
            case MAX -> values.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        };
    }

    private double defaultValue() {
        return 0.0;
    }
}
