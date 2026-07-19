package com.codedstream.otterstream.rules.model;

import java.util.List;
import java.util.Map;

/**
 * The outcome of evaluating an {@link InferenceResult} against a {@link RuleSet}.
 *
 * <p>{@code flag} is the primary decision label (e.g. {@code "FRAUD"}, {@code "APPROVE"},
 * {@code "REVIEW"} — entirely user-defined via rule YAML, not a fixed enum, since the set of
 * meaningful flags is domain-specific). {@code matchedRuleIds} lists every rule that
 * contributed to this decision — for {@link RuleEvaluationMode#SINGLE} that's exactly one id
 * (the highest-priority match); for {@link RuleEvaluationMode#MULTIPLE} it can be several.
 *
 * @param flag            primary decision label; {@code "UNFLAGGED"} if no rule matched
 * @param category        optional secondary classification (e.g. a risk tier); may be null
 * @param confidence       carried through from the triggering rule/inference, 0.0–1.0; 0.0 if unflagged
 * @param matchedRuleIds   ids of every rule that matched, highest priority first
 * @param timestampMillis  when this decision was produced
 * @param metadata         free-form pass-through data (e.g. which engine produced this, input snapshot)
 * @since 0.1.0
 */
public record Decision(
        String flag,
        String category,
        double confidence,
        List<String> matchedRuleIds,
        long timestampMillis,
        Map<String, Object> metadata) {

    public Decision {
        matchedRuleIds = matchedRuleIds == null ? List.of() : List.copyOf(matchedRuleIds);
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    /** @return true if at least one rule matched */
    public boolean isFlagged() {
        return !matchedRuleIds.isEmpty();
    }

    public static Decision unflagged(long timestampMillis) {
        return new Decision("UNFLAGGED", null, 0.0, List.of(), timestampMillis, Map.of());
    }
}
