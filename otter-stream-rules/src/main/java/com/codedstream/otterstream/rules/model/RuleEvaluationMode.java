package com.codedstream.otterstream.rules.model;

/**
 * Governs how {@link com.codedstream.otterstream.rules.spi.RuleEngine#evaluate} treats multiple
 * matching rules.
 *
 * @since 0.1.0
 */
public enum RuleEvaluationMode {
    /** Stop at the first (highest-priority) matching rule; {@link Decision#matchedRuleIds()} has at most one entry. */
    SINGLE,
    /** Evaluate every enabled rule; {@link Decision#matchedRuleIds()} can have several entries, flag is the highest-priority match's. */
    MULTIPLE
}
