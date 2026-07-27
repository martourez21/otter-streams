package com.codedstream.otterstream.runtime.feature;

/**
 * Aggregation applied over a {@link SlidingWindowFeatureProvider}'s windowed samples.
 *
 * @since 0.1.0
 */
public enum AggregationFunction {
    COUNT,
    SUM,
    AVG,
    MIN,
    MAX
}
