package com.codedstream.otterstream.rules.model;

import java.util.Objects;

/**
 * A single declarative rule: {@code if condition then flag = decisionFlag}.
 *
 * <p>{@code condition} is a small boolean expression (see
 * {@link com.codedstream.otterstream.rules.expr.ExpressionEvaluator}) evaluated against the
 * inference result's outputs/attributes plus any extra context passed in — e.g.
 * {@code "output.risk_score > 0.85 && output.country == 'US'"}.
 *
 * <p>{@code color} is an optional hex color (e.g. {@code "#c0392b"}) purely for dashboard
 * display — evaluation never depends on it. This is what lets an operator configure "fraud
 * flags render red, review flags render amber" once, at rule-definition time, per the
 * dashboard requirement.
 *
 * @param id               stable unique identifier, referenced in {@link Decision#matchedRuleIds()}
 * @param name             human-readable name shown on the dashboard
 * @param priority         higher wins in {@link RuleEvaluationMode#SINGLE} mode; ties broken by declaration order
 * @param condition        the boolean expression to evaluate
 * @param decisionFlag     the flag applied when this rule matches
 * @param decisionCategory optional secondary classification; may be null
 * @param enabled          disabled rules are parsed but never evaluated — lets you stage a rule before turning it on
 * @param color            optional hex color for dashboard rendering; may be null
 * @param description      optional human-readable explanation
 * @since 0.1.0
 */
public record Rule(
        String id,
        String name,
        int priority,
        String condition,
        String decisionFlag,
        String decisionCategory,
        boolean enabled,
        String color,
        String description) {

    public Rule {
        Objects.requireNonNull(id, "id cannot be null");
        Objects.requireNonNull(condition, "condition cannot be null");
        Objects.requireNonNull(decisionFlag, "decisionFlag cannot be null");
        if (color != null && !color.matches("^#[0-9a-fA-F]{6}$")) {
            throw new IllegalArgumentException(
                    "Rule '" + id + "': color must be a 6-digit hex code like '#c0392b', got: " + color);
        }
    }
}
