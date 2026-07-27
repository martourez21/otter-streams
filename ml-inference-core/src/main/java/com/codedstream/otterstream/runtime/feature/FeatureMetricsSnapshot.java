package com.codedstream.otterstream.runtime.feature;

/**
 * A point-in-time snapshot of a {@link FeatureProvider}'s call metrics, produced by
 * {@link MonitoredFeatureProvider} — the "feature monitoring" piece of the Feature Store
 * Integration roadmap item. Mirrors the shape of
 * {@code com.codedstream.otterstream.rules.spi.RuleMetricsSnapshot} deliberately, for
 * consistency across the project's metrics-snapshot types.
 *
 * @param providerId       which provider this snapshot is for
 * @param totalFetches     total {@code fetch()} calls observed
 * @param errorCount       how many of those threw
 * @param avgLatencyMicros average observed latency across all fetches (success and failure)
 * @param takenAtMillis    when this snapshot was produced
 * @since 0.1.0
 */
public record FeatureMetricsSnapshot(
        String providerId, long totalFetches, long errorCount, long avgLatencyMicros, long takenAtMillis) {

    public double errorRatePercent() {
        return totalFetches == 0 ? 0.0 : (errorCount * 100.0) / totalFetches;
    }
}
