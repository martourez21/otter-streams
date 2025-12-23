package com.codedstreams.otterstreams.sql.rules;

/**
 * Mutable rule outcome passed as a Drools global.
 *
 * <p>Rules set {@code triggered=true} when conditions match.</p>
 */
public class RuleOutcome {

    private boolean triggered;

    public boolean isTriggered() {
        return triggered;
    }

    public void trigger() {
        this.triggered = true;
    }
}
