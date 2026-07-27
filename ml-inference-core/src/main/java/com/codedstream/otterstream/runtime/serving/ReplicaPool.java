package com.codedstream.otterstream.runtime.serving;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A pool of {@link InferenceEngine} instances — all the same model version — load-balanced
 * across via a {@link LoadBalancingStrategy}. This is the "Distributed Model Serving" roadmap
 * item's <em>in-process</em> piece: more replicas of a model within one JVM/TaskManager than
 * {@code OtterRuntime}'s single-engine-per-{@code ManagedModel} default gives you, useful when
 * one engine instance can't keep up with the throughput a single Flink subtask needs to push
 * through it (e.g. a CPU-bound model where thread-level parallelism across several loaded
 * instances helps).
 *
 * <p><b>What this is not, stated plainly:</b> this is not cross-node / cross-JVM model sharding.
 * Distributing replicas across different physical TaskManagers is what Flink's own parallelism
 * and scheduling already do — {@code OtterRuntime} is deliberately an embedded, per-TaskManager
 * runtime (see the root README's "Embedded Runtime" deployment mode), and the Control Plane's
 * command fan-out (ARCHITECTURE.md §6.5) already spreads deploy/canary/shadow commands across
 * every TaskManager instance serving a model — that's the project's actual answer to
 * multi-node distribution today. Literal model-weight sharding across nodes (splitting one
 * model too large for a single machine, the way large-model serving frameworks handle
 * multi-GPU models) is a fundamentally different, much larger problem than this project's
 * target model sizes (ONNX/XGBoost/PMML-scale fraud-detection/recommendation models, not
 * multi-GPU LLM-scale) call for, and isn't attempted here.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ReplicaPool pool = new ReplicaPool("fraud-detector", new LeastConnectionsStrategy());
 * pool.addReplica(engineFactory.get()); // repeat to add more replicas
 *
 * InferenceResult result = pool.infer(inputs);
 * }</pre>
 *
 * @since 0.1.0
 * @see ReplicaAutoScaler
 */
public class ReplicaPool {

    private static final Logger LOG = LoggerFactory.getLogger(ReplicaPool.class);

    private final String modelId;
    private final LoadBalancingStrategy strategy;
    private final List<ReplicaHandle> replicas = new CopyOnWriteArrayList<>();

    public ReplicaPool(String modelId, LoadBalancingStrategy strategy) {
        this.modelId = Objects.requireNonNull(modelId, "modelId cannot be null");
        this.strategy = Objects.requireNonNull(strategy, "strategy cannot be null");
    }

    /** Adds an already-initialized engine replica to the pool. */
    public void addReplica(InferenceEngine<?> engine) {
        replicas.add(new ReplicaHandle(engine));
        LOG.info("Added replica to pool for model '{}' (now {} replica(s))", modelId, replicas.size());
    }

    /**
     * Removes and closes one replica, if the pool has more than {@code minReplicas} — used by
     * {@link ReplicaAutoScaler} to scale down. Removes the least-busy replica, to minimize
     * disruption to in-flight requests on the one being removed.
     *
     * @return true if a replica was removed
     */
    public boolean removeReplicaIfAbove(int minReplicas) {
        if (replicas.size() <= minReplicas) {
            return false;
        }
        ReplicaHandle leastBusy = replicas.stream()
                .min((a, b) -> Integer.compare(a.getInFlightCount(), b.getInFlightCount()))
                .orElse(null);
        if (leastBusy == null) {
            return false;
        }
        replicas.remove(leastBusy);
        try {
            leastBusy.getEngine().close();
        } catch (InferenceException e) {
            LOG.warn("Failed to cleanly close removed replica for model '{}': {}", modelId, e.getMessage(), e);
        }
        LOG.info("Removed replica from pool for model '{}' (now {} replica(s))", modelId, replicas.size());
        return true;
    }

    /**
     * Routes one inference call to whichever replica {@link LoadBalancingStrategy} selects.
     *
     * @throws IllegalStateException if the pool has no replicas
     */
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        if (replicas.isEmpty()) {
            throw new IllegalStateException("ReplicaPool for model '" + modelId + "' has no replicas");
        }
        List<ReplicaHandle> snapshot = new ArrayList<>(replicas);
        int index = strategy.selectReplicaIndex(snapshot);
        ReplicaHandle handle = snapshot.get(index);

        handle.enter();
        try {
            return handle.getEngine().infer(inputs);
        } finally {
            handle.exit();
        }
    }

    public int getReplicaCount() {
        return replicas.size();
    }

    /** @return total in-flight requests summed across every replica — the signal {@link ReplicaAutoScaler} scales on. */
    public int getTotalInFlight() {
        int total = 0;
        for (ReplicaHandle handle : replicas) {
            total += handle.getInFlightCount();
        }
        return total;
    }

    public String getModelId() {
        return modelId;
    }

    /** Closes every replica. The pool should not be reused afterward. */
    public void closeAll() {
        for (ReplicaHandle handle : replicas) {
            try {
                handle.getEngine().close();
            } catch (InferenceException e) {
                LOG.warn("Failed to cleanly close replica for model '{}': {}", modelId, e.getMessage(), e);
            }
        }
        replicas.clear();
    }
}
