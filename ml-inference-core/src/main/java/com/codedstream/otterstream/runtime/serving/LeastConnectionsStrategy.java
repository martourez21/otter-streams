package com.codedstream.otterstream.runtime.serving;

import java.util.List;

/**
 * Routes to whichever replica currently has the fewest in-flight requests — better than
 * round-robin when request cost varies (some inputs are more expensive than others), since it
 * naturally avoids piling more work onto a replica that's already busy with a slow request.
 *
 * @since 0.1.0
 */
public final class LeastConnectionsStrategy implements LoadBalancingStrategy {

    @Override
    public int selectReplicaIndex(List<ReplicaHandle> replicas) {
        int bestIndex = 0;
        int bestCount = Integer.MAX_VALUE;
        for (int i = 0; i < replicas.size(); i++) {
            int count = replicas.get(i).getInFlightCount();
            if (count < bestCount) {
                bestCount = count;
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
