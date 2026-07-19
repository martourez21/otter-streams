package com.codedstream.otterstream.rules.model;

/**
 * Thrown when rule evaluation fails — a malformed condition expression, an unresolvable field
 * reference, or an external {@link com.codedstream.otterstream.rules.spi.DecisionEngineConnector}
 * call failing.
 *
 * @since 0.1.0
 */
public class RuleEvaluationException extends Exception {

    public RuleEvaluationException(String message) {
        super(message);
    }

    public RuleEvaluationException(String message, Throwable cause) {
        super(message, cause);
    }
}
