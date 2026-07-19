package com.codedstream.otterstream.rules.spi;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.model.Decision;
import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Extension point for delegating decisions to an <em>external</em> enterprise decision engine —
 * Drools/Red Hat Decision Manager (KIE Server), Camunda DMN, IBM ODM, or any other system that
 * accepts a fact/input and returns a decision — rather than evaluating rules in-process via
 * {@link RuleEngine}.
 *
 * <p><b>Why one interface instead of a connector per vendor:</b> almost every enterprise
 * decision engine in production today exposes a REST (or REST-fronted) decision endpoint —
 * KIE Server's REST API, Camunda's DMN REST API, IBM ODM's Decision Server REST API, and
 * countless in-house systems. {@code otter-stream-rules} ships one configurable
 * {@code RestDecisionEngineConnector} that works against any of them (request/response mapping
 * is configuration, not code), rather than bespoke SDK integrations per vendor — several of
 * which (IBM ODM in particular) are licensed client libraries this project has no ability to
 * bundle. For teams that embed Drools directly in-process (not over REST), see the separate
 * {@code otter-stream-rules-drools} module, which implements this same interface — kept as its
 * own module specifically so Drools's KIE dependency tree is never forced onto a project that
 * doesn't use it (see {@code otter-stream-rules-drools/README.md}).
 *
 * @since 0.1.0
 */
public interface DecisionEngineConnector {

    /** A stable identifier, e.g. {@code "kie-server"}, {@code "camunda-dmn"}, {@code "drools-embedded"}. */
    String getConnectorId();

    /**
     * Sends one inference result to the external engine and returns its decision.
     *
     * @param inferenceResult the inference result to submit as input/facts
     * @param context         additional context fields to submit alongside it
     * @return the external engine's decision, mapped into Otter's {@link Decision} shape
     * @throws RuleEvaluationException if the external call fails or its response can't be mapped
     */
    Decision evaluate(InferenceResult inferenceResult, Map<String, Object> context) throws RuleEvaluationException;

    /**
     * Batch form. Default implementation calls {@link #evaluate} once per item — connectors
     * whose external engine has a real batch endpoint (most REST decision services do) should
     * override this to use it instead, for one round-trip instead of N.
     */
    default List<Decision> evaluateBatch(List<InferenceResult> results, Map<String, Object> context)
            throws RuleEvaluationException {
        List<Decision> decisions = new ArrayList<>(results.size());
        for (InferenceResult result : results) {
            decisions.add(evaluate(result, context));
        }
        return decisions;
    }
}
