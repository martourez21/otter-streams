package com.codedstreams.otterstreams.sql.rules;

/**
 * Generic decision engine abstraction.
 *
 * @param <T> input type (e.g. Map<String, Object>)
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public interface DecisionEngine<T> {

    /**
     * Evaluates the input against configured rules.
     *
     * @param input input facts
     * @return true if any rule is triggered
     */
    boolean evaluate(T input);
}


