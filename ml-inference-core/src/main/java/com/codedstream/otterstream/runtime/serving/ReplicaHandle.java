package com.codedstream.otterstream.runtime.serving;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pairs one replica engine instance with an in-flight request counter — the same technique
 * {@code LifecycleManager}'s package-private {@code EngineHandle} uses for graceful hot-swap
 * draining, made public here since {@link LoadBalancingStrategy} implementations need to read
 * in-flight counts to make routing decisions (e.g. {@link LeastConnectionsStrategy}).
 *
 * @since 0.1.0
 */
public final class ReplicaHandle {

    private final InferenceEngine<?> engine;
    private final AtomicInteger inFlight = new AtomicInteger(0);

    public ReplicaHandle(InferenceEngine<?> engine) {
        this.engine = engine;
    }

    public InferenceEngine<?> getEngine() {
        return engine;
    }

    public int getInFlightCount() {
        return inFlight.get();
    }

    int enter() {
        return inFlight.incrementAndGet();
    }

    int exit() {
        return inFlight.decrementAndGet();
    }
}
