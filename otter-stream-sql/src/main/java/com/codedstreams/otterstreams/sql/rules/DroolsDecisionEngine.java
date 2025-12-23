package com.codedstreams.otterstreams.sql.rules;

import com.codedstreams.otterstreams.sql.rules.DecisionEngine;
import com.codedstreams.otterstreams.sql.rules.RuleOutcome;
import org.kie.api.runtime.StatelessKieSession;

/**
 * Stateless Drools-based decision engine.
 *
 * <p>Designed for Flink streaming workloads:
 * <ul>
 *   <li>No mutable global state</li>
 *   <li>Thread-safe per task</li>
 *   <li>Fast rule evaluation</li>
 * </ul>
 */
public class DroolsDecisionEngine implements DecisionEngine<Object> {

    private final StatelessKieSession session;

    public DroolsDecisionEngine(StatelessKieSession session) {
        this.session = session;
    }

    @Override
    public boolean evaluate(Object input) {
        RuleOutcome outcome = new RuleOutcome();
        session.setGlobal("outcome", outcome);
        session.execute(input);
        return outcome.isTriggered();
    }
}
