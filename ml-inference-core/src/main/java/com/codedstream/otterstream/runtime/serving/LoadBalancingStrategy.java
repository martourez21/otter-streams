package com.codedstream.otterstream.runtime.serving;

import java.util.List;

/**
 * Picks which replica in a {@link ReplicaPool} should serve the next request.
 *
 * @since 0.1.0
 */
public interface LoadBalancingStrategy {

    /**
     * @param replicas the pool's current replicas, in a stable order
     * @return the index into {@code replicas} to route this request to
     */
    int selectReplicaIndex(List<ReplicaHandle> replicas);
}
