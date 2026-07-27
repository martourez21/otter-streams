package com.codedstream.otterstream.experiments.model;

import java.util.Objects;

/**
 * One recorded observation for an experiment: which group served the request, and a numeric
 * outcome metric. What that metric <em>means</em> is entirely up to the caller — a rule
 * engine's confidence score, a binary 1.0/0.0 for "flagged as fraud," a latency in
 * milliseconds, a downstream business metric fed back in later. See
 * {@code ExperimentManager}'s Javadoc for the two metric shapes its statistical comparison
 * supports (continuous vs. binary/proportion).
 *
 * @param experimentId which experiment this belongs to
 * @param group         CONTROL or VARIANT
 * @param metricValue   the observed value
 * @param timestampMillis when this observation was recorded
 * @since 0.1.0
 */
public record ExperimentOutcome(String experimentId, ExperimentGroup group, double metricValue, long timestampMillis) {

    public ExperimentOutcome {
        Objects.requireNonNull(experimentId, "experimentId cannot be null");
        Objects.requireNonNull(group, "group cannot be null");
    }
}
