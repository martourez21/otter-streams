package com.codedstreams.otterstreams.sql.runtime;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Circuit breaker to prevent cascading failures in inference operations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class InferenceCircuitBreaker {
    private static final Logger LOG = LoggerFactory.getLogger(InferenceCircuitBreaker.class);

    private final String name;
    private final int failureThreshold;
    private final long resetTimeoutMs;
    private final AtomicInteger failureCount;
    private final AtomicLong lastFailureTime;
    private final AtomicReference<State> state;

    public enum State {
        CLOSED,     // Normal operation
        OPEN,       // Circuit is open, requests fail fast
        HALF_OPEN   // Testing if service recovered
    }

    public InferenceCircuitBreaker(String name, int failureThreshold, long resetTimeoutMs) {
        this.name = name;
        this.failureThreshold = failureThreshold;
        this.resetTimeoutMs = resetTimeoutMs;
        this.failureCount = new AtomicInteger(0);
        this.lastFailureTime = new AtomicLong(0);
        this.state = new AtomicReference<>(State.CLOSED);
    }

    /**
     * Attempts to execute operation through circuit breaker.
     */
    public <T> T execute(java.util.function.Supplier<T> operation) throws Exception {
        State currentState = state.get();

        // Check if circuit should transition to HALF_OPEN
        if (currentState == State.OPEN && shouldAttemptReset()) {
            state.compareAndSet(State.OPEN, State.HALF_OPEN);
            currentState = State.HALF_OPEN;
            LOG.info("Circuit breaker '{}' transitioning to HALF_OPEN", name);
        }

        // Fail fast if circuit is OPEN
        if (currentState == State.OPEN) {
            throw new Exception("Circuit breaker '" + name + "' is OPEN");
        }

        // Attempt operation
        try {
            T result = operation.get();
            onSuccess();
            return result;
        } catch (Exception e) {
            onFailure();
            throw e;
        }
    }

    private void onSuccess() {
        if (state.get() == State.HALF_OPEN) {
            reset();
            LOG.info("Circuit breaker '{}' recovered, transitioning to CLOSED", name);
        }
        failureCount.set(0);
    }

    private void onFailure() {
        int failures = failureCount.incrementAndGet();
        lastFailureTime.set(System.currentTimeMillis());

        if (failures >= failureThreshold && state.compareAndSet(State.CLOSED, State.OPEN)) {
            LOG.warn("Circuit breaker '{}' opened after {} failures", name, failures);
        } else if (state.get() == State.HALF_OPEN) {
            state.set(State.OPEN);
            LOG.warn("Circuit breaker '{}' reopened", name);
        }
    }

    private boolean shouldAttemptReset() {
        long timeSinceLastFailure = System.currentTimeMillis() - lastFailureTime.get();
        return timeSinceLastFailure >= resetTimeoutMs;
    }

    private void reset() {
        failureCount.set(0);
        state.set(State.CLOSED);
    }

    public State getState() {
        return state.get();
    }

    public int getFailureCount() {
        return failureCount.get();
    }
}
