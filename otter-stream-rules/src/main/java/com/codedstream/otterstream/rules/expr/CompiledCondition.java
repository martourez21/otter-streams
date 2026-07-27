package com.codedstream.otterstream.rules.expr;

import java.util.Map;

/**
 * A parsed, ready-to-evaluate boolean condition. Compiling once and evaluating many times (via
 * {@link ExpressionEvaluator#compile(String)}) is deliberate — re-parsing a rule's condition
 * string on every inference call would put string parsing directly on the hot path, which is
 * exactly what this module's performance notes (see {@code otter-stream-rules/ARCHITECTURE.md})
 * call out as unacceptable given the sub-5ms inference latency target.
 *
 * @since 0.1.0
 */
@FunctionalInterface
public interface CompiledCondition {
    /**
     * @param context field values the condition's field references resolve against — typically
     *                a flattened view of an {@link com.codedstream.otterstream.inference.model.InferenceResult}
     *                merged with caller-supplied extra context
     * @return true if the condition holds
     */
    boolean evaluate(Map<String, Object> context);
}
