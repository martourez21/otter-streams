package com.codedstream.otterstream.runtime.serving;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Automatically scales a {@link ReplicaPool}'s replica count up or down based on average
 * in-flight requests per replica — the "automatic scaling" piece of the Distributed Model
 * Serving roadmap item.
 *
 * <p><b>Why this can honestly automate both directions, unlike
 * {@link com.codedstream.otterstream.runtime.hardware.ExecutionTargetManager}'s GPU scaling:</b>
 * in-flight request count is a real, directly-measured, present-tense signal — "this pool is
 * currently backed up" is not a prediction, it's an observation. GPU scale-up needed a
 * forward-looking signal ("traffic is about to spike") that utilization alone can't provide,
 * which is why that class deliberately left scale-up as an explicit trigger. This class has no
 * equivalent gap: both scale-up (average in-flight exceeds a high-watermark, sustained) and
 * scale-down (falls below a low-watermark, sustained) are decided from the same kind of signal,
 * so both are automated here.
 *
 * <p>The "sustained for" duration exists specifically to avoid flapping — scaling up and back
 * down repeatedly in response to a brief burst — by requiring the threshold to stay crossed for
 * a configurable duration before acting.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ReplicaAutoScaler scaler = new ReplicaAutoScaler(
 *         pool, () -> engineFactory.create(),
 *         1, 8,                       // min/max replicas
 *         2.0, 0.2,                   // scale up above 2.0 avg in-flight, down below 0.2
 *         Duration.ofSeconds(30));    // sustained for 30s before acting
 * scaler.start(Duration.ofSeconds(5));
 * }</pre>
 *
 * @since 0.1.0
 */
public class ReplicaAutoScaler {

    private static final Logger LOG = LoggerFactory.getLogger(ReplicaAutoScaler.class);

    private final ReplicaPool pool;
    private final Supplier<InferenceEngine<?>> replicaFactory;
    private final int minReplicas;
    private final int maxReplicas;
    private final double scaleUpThreshold;
    private final double scaleDownThreshold;
    private final long sustainedMillis;

    private final ScheduledExecutorService scheduler;
    private ScheduledFuture<?> tickFuture;
    private long aboveThresholdSinceMillis = -1;
    private long belowThresholdSinceMillis = -1;

    /**
     * @param pool               the pool to scale
     * @param replicaFactory     creates a new, ready-to-use engine replica on scale-up
     * @param minReplicas        never scale below this many replicas
     * @param maxReplicas        never scale above this many replicas
     * @param scaleUpThreshold   scale up when average in-flight per replica exceeds this, sustained
     * @param scaleDownThreshold scale down when average in-flight per replica falls below this, sustained
     * @param sustainedFor       how long a threshold must stay crossed before acting (anti-flapping)
     */
    public ReplicaAutoScaler(
            ReplicaPool pool,
            Supplier<InferenceEngine<?>> replicaFactory,
            int minReplicas,
            int maxReplicas,
            double scaleUpThreshold,
            double scaleDownThreshold,
            Duration sustainedFor) {
        if (minReplicas < 1 || maxReplicas < minReplicas) {
            throw new IllegalArgumentException(
                    "Require 1 <= minReplicas <= maxReplicas, got min=" + minReplicas + " max=" + maxReplicas);
        }
        if (scaleDownThreshold >= scaleUpThreshold) {
            throw new IllegalArgumentException("scaleDownThreshold must be less than scaleUpThreshold");
        }
        this.pool = pool;
        this.replicaFactory = replicaFactory;
        this.minReplicas = minReplicas;
        this.maxReplicas = maxReplicas;
        this.scaleUpThreshold = scaleUpThreshold;
        this.scaleDownThreshold = scaleDownThreshold;
        this.sustainedMillis = sustainedFor.toMillis();

        ThreadFactory threadFactory = runnable -> {
            Thread t = new Thread(runnable, "otter-replica-autoscaler-" + pool.getModelId());
            t.setDaemon(true);
            return t;
        };
        this.scheduler = Executors.newSingleThreadScheduledExecutor(threadFactory);
    }

    public void start(Duration tickInterval) {
        long millis = Math.max(1000, tickInterval.toMillis());
        this.tickFuture = scheduler.scheduleWithFixedDelay(this::tick, millis, millis, TimeUnit.MILLISECONDS);
    }

    private void tick() {
        int replicaCount = pool.getReplicaCount();
        if (replicaCount == 0) {
            return;
        }
        double avgInFlight = (double) pool.getTotalInFlight() / replicaCount;
        long now = System.currentTimeMillis();

        if (avgInFlight > scaleUpThreshold && replicaCount < maxReplicas) {
            belowThresholdSinceMillis = -1;
            if (aboveThresholdSinceMillis < 0) {
                aboveThresholdSinceMillis = now;
            } else if (now - aboveThresholdSinceMillis >= sustainedMillis) {
                LOG.info("Scaling up model '{}': avg in-flight {} > threshold {} sustained for {}ms",
                        pool.getModelId(), avgInFlight, scaleUpThreshold, sustainedMillis);
                pool.addReplica(replicaFactory.get());
                aboveThresholdSinceMillis = -1;
            }
        } else if (avgInFlight < scaleDownThreshold && replicaCount > minReplicas) {
            aboveThresholdSinceMillis = -1;
            if (belowThresholdSinceMillis < 0) {
                belowThresholdSinceMillis = now;
            } else if (now - belowThresholdSinceMillis >= sustainedMillis) {
                LOG.info("Scaling down model '{}': avg in-flight {} < threshold {} sustained for {}ms",
                        pool.getModelId(), avgInFlight, scaleDownThreshold, sustainedMillis);
                pool.removeReplicaIfAbove(minReplicas);
                belowThresholdSinceMillis = -1;
            }
        } else {
            aboveThresholdSinceMillis = -1;
            belowThresholdSinceMillis = -1;
        }
    }

    public void shutdown() {
        if (tickFuture != null) {
            tickFuture.cancel(false);
        }
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
