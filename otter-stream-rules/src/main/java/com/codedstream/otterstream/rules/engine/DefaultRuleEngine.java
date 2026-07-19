package com.codedstream.otterstream.rules.engine;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.expr.CompiledCondition;
import com.codedstream.otterstream.rules.expr.ExpressionEvaluator;
import com.codedstream.otterstream.rules.model.Decision;
import com.codedstream.otterstream.rules.model.Rule;
import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import com.codedstream.otterstream.rules.model.RuleEvaluationMode;
import com.codedstream.otterstream.rules.model.RuleSet;
import com.codedstream.otterstream.rules.spi.RuleEngine;
import com.codedstream.otterstream.rules.spi.RuleMetricsSnapshot;
import com.codedstream.otterstream.rules.spi.RuleSetSource;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * The concrete {@link RuleEngine} used regardless of where the {@link RuleSet} came from
 * (YAML, properties, or a hand-written {@link RuleSetSource}) — all three configuration paths
 * converge here.
 *
 * <h2>Performance notes</h2>
 * <ul>
 *   <li>Every rule's condition is compiled once, at construction, via
 *       {@link ExpressionEvaluator#compile} — {@code evaluate}/{@code evaluateBatch} never parse
 *       a string. This is the single most important thing keeping evaluation off the "slow"
 *       side of the sub-5ms inference latency target.</li>
 *   <li>Metrics use {@link LongAdder}, not {@code AtomicLong} — under high concurrent write
 *       contention (many parallel inference calls all incrementing the same rule's hit counter)
 *       {@code LongAdder} scales better by striping updates across cells, at the cost of a
 *       slightly more expensive read ({@link LongAdder#sum()}) — exactly the right trade-off
 *       here, since hit counters are written constantly but only read occasionally (dashboard
 *       polling).</li>
 *   <li>{@code evaluateBatch} evaluates sequentially below {@link #PARALLEL_BATCH_THRESHOLD}
 *       items and via a parallel stream above it — avoids paying fork/join overhead on small
 *       batches while still scaling for large ones.</li>
 * </ul>
 *
 * @since 0.1.0
 */
public final class DefaultRuleEngine implements RuleEngine {

    /** Batches at or below this size evaluate sequentially — avoids fork/join overhead for small batches. */
    private static final int PARALLEL_BATCH_THRESHOLD = 64;

    private final String engineId;
    private final RuleSet ruleSet;
    private final List<Rule> rulesByPriority;
    private final Map<String, CompiledCondition> compiledConditions;

    private final LongAdder totalEvaluations = new LongAdder();
    private final LongAdder unflaggedCount = new LongAdder();
    private final Map<String, LongAdder> hitsByRuleId = new ConcurrentHashMap<>();
    private final Map<String, LongAdder> hitsByFlag = new ConcurrentHashMap<>();

    public DefaultRuleEngine(RuleSetSource source) throws RuleEvaluationException {
        this(source, null);
    }

    public DefaultRuleEngine(RuleSetSource source, String engineIdOverride) throws RuleEvaluationException {
        Objects.requireNonNull(source, "source cannot be null");
        this.ruleSet = source.load();
        this.engineId = engineIdOverride != null ? engineIdOverride : (ruleSet.name() + ":" + ruleSet.version());
        this.rulesByPriority = ruleSet.enabledRulesByPriority();

        Map<String, CompiledCondition> compiled = new LinkedHashMap<>();
        for (Rule rule : ruleSet.rules()) {
            compiled.put(rule.id(), ExpressionEvaluator.compile(rule.condition()));
        }
        this.compiledConditions = Map.copyOf(compiled);
    }

    @Override
    public String getEngineId() {
        return engineId;
    }

    @Override
    public RuleSet getRuleSet() {
        return ruleSet;
    }

    @Override
    public Decision evaluate(InferenceResult inferenceResult, Map<String, Object> context) throws RuleEvaluationException {
        Map<String, Object> evalContext = buildEvalContext(inferenceResult, context);
        totalEvaluations.increment();

        List<String> matchedIds = new ArrayList<>();
        String flag = null;
        String category = null;

        for (Rule rule : rulesByPriority) {
            CompiledCondition condition = compiledConditions.get(rule.id());
            boolean matched;
            try {
                matched = condition.evaluate(evalContext);
            } catch (Exception e) {
                throw new RuleEvaluationException(
                        "Failed to evaluate rule '" + rule.id() + "' (" + rule.condition() + ")", e);
            }
            if (!matched) {
                continue;
            }
            matchedIds.add(rule.id());
            hitsByRuleId.computeIfAbsent(rule.id(), id -> new LongAdder()).increment();
            if (flag == null) {
                flag = rule.decisionFlag();
                category = rule.decisionCategory();
            }
            if (ruleSet.evaluationMode() == RuleEvaluationMode.SINGLE) {
                break;
            }
        }

        long now = System.currentTimeMillis();
        if (flag == null) {
            unflaggedCount.increment();
            hitsByFlag.computeIfAbsent("UNFLAGGED", f -> new LongAdder()).increment();
            return Decision.unflagged(now);
        }

        hitsByFlag.computeIfAbsent(flag, f -> new LongAdder()).increment();
        double confidence = extractConfidence(inferenceResult);
        return new Decision(flag, category, confidence, matchedIds, now, Map.of("engineId", engineId));
    }

    @Override
    public List<Decision> evaluateBatch(List<InferenceResult> inferenceResults, Map<String, Object> context)
            throws RuleEvaluationException {
        if (inferenceResults.isEmpty()) {
            return List.of();
        }
        if (inferenceResults.size() <= PARALLEL_BATCH_THRESHOLD) {
            List<Decision> decisions = new ArrayList<>(inferenceResults.size());
            for (InferenceResult result : inferenceResults) {
                decisions.add(evaluate(result, context));
            }
            return decisions;
        }

        // Above the threshold: parallelize. Each evaluate() call only touches thread-safe
        // structures (LongAdder counters, immutable compiled conditions), so this is safe
        // without additional synchronization.
        List<RuleEvaluationException> failures = new ArrayList<>();
        List<Decision> results = inferenceResults.parallelStream()
                .map(result -> {
                    try {
                        return evaluate(result, context);
                    } catch (RuleEvaluationException e) {
                        synchronized (failures) {
                            failures.add(e);
                        }
                        return null;
                    }
                })
                .toList();

        if (!failures.isEmpty()) {
            throw failures.get(0);
        }
        return results;
    }

    @Override
    public RuleMetricsSnapshot getMetrics() {
        Map<String, Long> ruleHits = new LinkedHashMap<>();
        hitsByRuleId.forEach((id, adder) -> ruleHits.put(id, adder.sum()));
        Map<String, Long> flagHits = new LinkedHashMap<>();
        hitsByFlag.forEach((flag, adder) -> flagHits.put(flag, adder.sum()));

        return new RuleMetricsSnapshot(
                engineId,
                totalEvaluations.sum(),
                unflaggedCount.sum(),
                ruleHits,
                flagHits,
                System.currentTimeMillis());
    }

    private Map<String, Object> buildEvalContext(InferenceResult inferenceResult, Map<String, Object> extraContext) {
        Map<String, Object> context = new HashMap<>();
        context.put("output", inferenceResult.getOutputs());
        context.put("modelId", inferenceResult.getModelId());
        context.put("inferenceTimeMs", inferenceResult.getInferenceTimeMs());
        context.put("success", inferenceResult.isSuccess());
        if (extraContext != null && !extraContext.isEmpty()) {
            context.putAll(extraContext);
        }
        return context;
    }

    private double extractConfidence(InferenceResult inferenceResult) {
        Object confidence = inferenceResult.getOutputs().get("confidence");
        if (confidence instanceof Number number) {
            return number.doubleValue();
        }
        return 0.0;
    }
}
