package com.codedstream.otterstream.rules.spi;

import java.util.Map;

/**
 * A point-in-time, immutable snapshot of rule-evaluation metrics — what the rule dashboard
 * (Otter Control Plane) polls or receives via telemetry to show per-rule/per-flag hit counts
 * and transaction volume.
 *
 * @param engineId          which {@link RuleEngine} this snapshot came from
 * @param totalEvaluations  total calls to {@code evaluate}/{@code evaluateBatch} (counting each
 *                          batch item individually) since the engine was created
 * @param unflaggedCount    how many of those produced {@link com.codedstream.otterstream.rules.model.Decision#unflagged}
 * @param hitsByRuleId      per-rule match counts, keyed by {@link com.codedstream.otterstream.rules.model.Rule#id()}
 * @param hitsByFlag        per-flag counts, keyed by {@link com.codedstream.otterstream.rules.model.Decision#flag()}
 * @param takenAtMillis     when this snapshot was produced
 * @since 0.1.0
 */
public record RuleMetricsSnapshot(
        String engineId,
        long totalEvaluations,
        long unflaggedCount,
        Map<String, Long> hitsByRuleId,
        Map<String, Long> hitsByFlag,
        long takenAtMillis) {

    public RuleMetricsSnapshot {
        hitsByRuleId = hitsByRuleId == null ? Map.of() : Map.copyOf(hitsByRuleId);
        hitsByFlag = hitsByFlag == null ? Map.of() : Map.copyOf(hitsByFlag);
    }
}
