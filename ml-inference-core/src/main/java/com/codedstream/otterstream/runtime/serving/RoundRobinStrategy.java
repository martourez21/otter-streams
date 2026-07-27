package com.codedstream.otterstream.runtime.serving;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Cycles through replicas in order — the simplest strategy, and the right default when replicas
 * are homogeneous and requests are roughly uniform cost.
 *
 * @since 0.1.0
 */
public final class RoundRobinStrategy implements LoadBalancingStrategy {

    private final AtomicInteger counter = new AtomicInteger(0);

    @Override
    public int selectReplicaIndex(List<ReplicaHandle> replicas) {
        int index = Math.floorMod(counter.getAndIncrement(), replicas.size());
        return index;
    }
}
