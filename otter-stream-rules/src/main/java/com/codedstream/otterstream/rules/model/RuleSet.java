package com.codedstream.otterstream.rules.model;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * A named, versioned collection of {@link Rule}s plus the evaluation mode they should be
 * applied with. This is what a rule YAML file (or a programmatic {@code RuleSet.builder()...})
 * ultimately produces.
 *
 * @param name           display name, shown on the rule dashboard
 * @param version        free-form version string (e.g. {@code "3"}, {@code "2025-07-16"}) — purely informational, not compared
 * @param evaluationMode SINGLE or MULTIPLE, see {@link RuleEvaluationMode}
 * @param rules          the rules themselves, in declaration order
 * @since 0.1.0
 */
public record RuleSet(String name, String version, RuleEvaluationMode evaluationMode, List<Rule> rules) {

    public RuleSet {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(evaluationMode, "evaluationMode cannot be null");
        rules = rules == null ? List.of() : List.copyOf(rules);

        long distinctIds = rules.stream().map(Rule::id).distinct().count();
        if (distinctIds != rules.size()) {
            throw new IllegalArgumentException("RuleSet '" + name + "' contains duplicate rule ids");
        }
    }

    /**
     * @return enabled rules only, ordered by descending priority (ties broken by declaration
     *         order) — the order {@link com.codedstream.otterstream.rules.spi.RuleEngine}
     *         implementations should evaluate in.
     */
    public List<Rule> enabledRulesByPriority() {
        return rules.stream()
                .filter(Rule::enabled)
                .sorted(Comparator.comparingInt(Rule::priority).reversed())
                .toList();
    }
}
