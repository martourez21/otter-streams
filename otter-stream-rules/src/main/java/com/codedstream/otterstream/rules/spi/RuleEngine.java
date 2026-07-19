package com.codedstream.otterstream.rules.spi;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.model.Decision;
import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import com.codedstream.otterstream.rules.model.RuleSet;
import java.util.List;
import java.util.Map;

/**
 * Evaluates {@link InferenceResult}s against a {@link RuleSet}, producing {@link Decision}s.
 *
 * <p>This is the extension point for the Rule Engine feature: {@code YamlRuleEngine} (the
 * default, in this same module) is one implementation; a project can supply its own by
 * implementing this interface directly — the standard/default path is YAML, but a hand-written
 * class is fully supported and is exactly this interface.
 *
 * <p>Implementations must be safe for concurrent use — {@code evaluate}/{@code evaluateBatch}
 * are expected to sit on or near the inference hot path (see
 * {@code otter-stream-rules/ARCHITECTURE.md} performance notes), so implementations should
 * avoid locking across an entire rule set evaluation; {@code YamlRuleEngine} uses per-rule
 * atomic counters for metrics specifically to avoid that.
 *
 * @since 0.1.0
 */
public interface RuleEngine {

    /** A stable, unique identifier for this engine instance/configuration, e.g. {@code "fraud-rules-v3"}. */
    String getEngineId();

    /** @return the rule set currently loaded/active */
    RuleSet getRuleSet();

    /**
     * Evaluates one inference result. Mode (SINGLE vs MULTIPLE matches) is governed by
     * {@link RuleSet#evaluationMode()}, not a parameter here — it's a property of the rule set,
     * not of an individual call.
     *
     * @param inferenceResult the inference result to evaluate
     * @param context         additional fields to make available to rule conditions beyond what's
     *                        in {@code inferenceResult} itself (e.g. request-scoped metadata);
     *                        may be empty
     * @return the resulting decision — {@link Decision#unflagged} if no rule matched
     * @throws RuleEvaluationException if a rule's condition fails to evaluate
     */
    Decision evaluate(InferenceResult inferenceResult, Map<String, Object> context) throws RuleEvaluationException;

    /**
     * Evaluates a batch of inference results — the "batch" flagging mode. Implementations may
     * parallelize this (the default {@code YamlRuleEngine} does, above a configurable batch-size
     * threshold) since each result's evaluation is independent.
     *
     * @param inferenceResults the inference results to evaluate, in order
     * @param context          shared context applied to every result in the batch
     * @return one decision per input result, same order
     * @throws RuleEvaluationException if any result's evaluation fails
     */
    List<Decision> evaluateBatch(List<InferenceResult> inferenceResults, Map<String, Object> context)
            throws RuleEvaluationException;

    /** @return a point-in-time snapshot of per-rule/per-flag hit counters, for the rule dashboard */
    RuleMetricsSnapshot getMetrics();
}
