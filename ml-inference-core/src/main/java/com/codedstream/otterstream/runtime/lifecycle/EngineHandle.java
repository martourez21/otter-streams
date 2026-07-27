package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Package-private wrapper pairing an {@link InferenceEngine} with an in-flight request
 * counter and its {@link ModelVersion} metadata.
 *
 * <p>This is what makes graceful hot-swap draining possible: instead of retiring an engine
 * the instant a new one is swapped in, {@link LifecycleManager} can watch
 * {@link #getInFlightCount()} drop to zero (up to a timeout) before calling {@code close()},
 * so requests that were already in flight against the old version get to finish normally.
 *
 * @since 0.1.0
 */
final class EngineHandle {

    private final InferenceEngine<?> engine;
    private final ModelVersion version;
    private final AtomicInteger inFlight = new AtomicInteger(0);

    EngineHandle(InferenceEngine<?> engine, ModelVersion version) {
        this.engine = engine;
        this.version = version;
    }

    InferenceEngine<?> getEngine() {
        return engine;
    }

    ModelVersion getVersion() {
        return version;
    }

    /**
     * Marks the start of a request against this handle's engine.
     *
     * @return the new in-flight count
     */
    int enter() {
        return inFlight.incrementAndGet();
    }

    /**
     * Marks the end of a request against this handle's engine. Always call from a
     * {@code finally} block paired with {@link #enter()}.
     *
     * @return the new in-flight count
     */
    int exit() {
        return inFlight.decrementAndGet();
    }

    int getInFlightCount() {
        return inFlight.get();
    }
}
