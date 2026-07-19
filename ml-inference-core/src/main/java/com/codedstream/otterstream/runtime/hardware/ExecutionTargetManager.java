package com.codedstream.otterstream.runtime.hardware;

import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Watches {@link ExecutionTargetAware} engines on a fixed tick and automatically switches an
 * idle GPU-resident engine back to CPU, freeing GPU memory/compute for other work — the "scale
 * down or switch off the feature when operations return to normal" behavior.
 *
 * <h2>What this does and does not do (read before relying on it)</h2>
 * <ul>
 *   <li><b>Implemented: automatic scale-down.</b> If a GPU engine's
 *       {@link ExecutionTargetAware#getRecentUtilization()} stays below
 *       {@code idleUtilizationThreshold} for {@code idleDuration}, this manager calls
 *       {@link ExecutionTargetAware#switchTo(ExecutionTarget) switchTo(CPU)} automatically, on
 *       its own background thread, no operator action required.</li>
 *   <li><b>Not implemented: automatic scale-up.</b> Deciding "traffic just spiked, move this
 *       engine back to GPU" needs a forward-looking signal (queue depth, incoming request rate)
 *       that a CPU-resident engine's own utilization reading can't provide by itself — CPU
 *       utilization tells you the engine is busy on CPU, not that GPU would meaningfully help
 *       right now. Rather than fabricate a heuristic here, scale-up is exposed as an explicit
 *       {@link #requestScaleUp(String)} call — wire it to whatever signal your deployment
 *       actually has (a queue-depth alert, a Kubernetes HPA-style metric, a manual runbook
 *       step). This is the same honesty trade-off already made for topology backpressure
 *       detection (see {@code otter-control-plane/ARCHITECTURE.md} §7.3) — flagged as a real
 *       limitation rather than papered over with a guess.</li>
 *   <li><b>Not implemented: actual GPU driver/CUDA control.</b> This class is a policy/scheduling
 *       layer only. The real GPU release/acquire work happens inside each engine's
 *       {@link ExecutionTargetAware#switchTo} implementation (e.g. an ONNX Runtime engine
 *       rebuilding its session with a CPU execution provider) — this manager never touches
 *       CUDA/driver APIs directly.</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ExecutionTargetManager manager = new ExecutionTargetManager(
 *         0.05,                     // idle = under 5% utilization
 *         Duration.ofMinutes(10));  // ...for at least 10 minutes
 * manager.register("fraud-detector", onnxEngine);  // onnxEngine implements ExecutionTargetAware
 * manager.start(Duration.ofSeconds(30));            // check every 30s
 * // ...
 * manager.requestScaleUp("fraud-detector");         // e.g. wired to a traffic-spike alert
 * // ...
 * manager.shutdown();
 * }</pre>
 *
 * @since 0.1.0
 */
public class ExecutionTargetManager {

    private static final Logger LOG = LoggerFactory.getLogger(ExecutionTargetManager.class);

    private final double idleUtilizationThreshold;
    private final long idleDurationMillis;

    private final ConcurrentHashMap<String, ExecutionTargetAware> engines = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Long> lastAboveThresholdMillis = new ConcurrentHashMap<>();

    private final ScheduledExecutorService scheduler;
    private ScheduledFuture<?> tickFuture;

    /**
     * @param idleUtilizationThreshold utilization below this (0.0-1.0) counts as "idle" for a given tick
     * @param idleDuration             how long an engine must stay idle before it's scaled down
     */
    public ExecutionTargetManager(double idleUtilizationThreshold, Duration idleDuration) {
        if (idleUtilizationThreshold < 0.0 || idleUtilizationThreshold > 1.0) {
            throw new IllegalArgumentException("idleUtilizationThreshold must be between 0.0 and 1.0");
        }
        this.idleUtilizationThreshold = idleUtilizationThreshold;
        this.idleDurationMillis = Objects.requireNonNull(idleDuration, "idleDuration cannot be null").toMillis();

        ThreadFactory threadFactory = runnable -> {
            Thread t = new Thread(runnable, "otter-execution-target-manager");
            t.setDaemon(true);
            return t;
        };
        this.scheduler = Executors.newSingleThreadScheduledExecutor(threadFactory);
    }

    /** Registers an engine for automatic idle scale-down monitoring. */
    public void register(String engineId, ExecutionTargetAware engine) {
        engines.put(engineId, engine);
        lastAboveThresholdMillis.put(engineId, System.currentTimeMillis());
    }

    public void unregister(String engineId) {
        engines.remove(engineId);
        lastAboveThresholdMillis.remove(engineId);
    }

    /** Starts the monitoring tick. Safe to call once; call {@link #shutdown()} before re-starting. */
    public void start(Duration tickInterval) {
        long millis = Math.max(1000, tickInterval.toMillis());
        this.tickFuture = scheduler.scheduleWithFixedDelay(this::tick, millis, millis, TimeUnit.MILLISECONDS);
        LOG.info("ExecutionTargetManager started (tick every {}ms, idle threshold {}, idle duration {}ms)",
                millis, idleUtilizationThreshold, idleDurationMillis);
    }

    /**
     * Explicit scale-up trigger — see the class Javadoc for why this isn't automatic. Wire this
     * to whatever forward-looking load signal your deployment has.
     *
     * @return true if the engine was found and the switch succeeded
     */
    public boolean requestScaleUp(String engineId) {
        ExecutionTargetAware engine = engines.get(engineId);
        if (engine == null) {
            LOG.warn("requestScaleUp called for unknown/unregistered engine '{}'", engineId);
            return false;
        }
        boolean switched = engine.switchTo(ExecutionTarget.GPU);
        if (switched) {
            lastAboveThresholdMillis.put(engineId, System.currentTimeMillis());
            LOG.info("Engine '{}' scaled up to GPU on explicit request", engineId);
        } else {
            LOG.warn("Engine '{}' declined scale-up to GPU (switchTo returned false)", engineId);
        }
        return switched;
    }

    private void tick() {
        long now = System.currentTimeMillis();
        for (var entry : engines.entrySet()) {
            String engineId = entry.getKey();
            ExecutionTargetAware engine = entry.getValue();
            try {
                evaluateEngine(engineId, engine, now);
            } catch (Exception e) {
                LOG.warn("ExecutionTargetManager tick failed for engine '{}': {}", engineId, e.getMessage(), e);
            }
        }
    }

    private void evaluateEngine(String engineId, ExecutionTargetAware engine, long now) {
        double utilization = engine.getRecentUtilization();
        if (utilization >= idleUtilizationThreshold) {
            lastAboveThresholdMillis.put(engineId, now);
            return;
        }

        if (engine.getCurrentExecutionTarget() != ExecutionTarget.GPU) {
            return; // already on CPU, nothing to scale down
        }

        long idleSince = lastAboveThresholdMillis.getOrDefault(engineId, now);
        long idleForMillis = now - idleSince;
        if (idleForMillis < idleDurationMillis) {
            return;
        }

        LOG.info("Engine '{}' idle for {}ms (utilization {} < threshold {}) — scaling down to CPU",
                engineId, idleForMillis, utilization, idleUtilizationThreshold);
        boolean switched = engine.switchTo(ExecutionTarget.CPU);
        if (switched) {
            LOG.info("Engine '{}' scaled down to CPU", engineId);
        } else {
            LOG.warn("Engine '{}' declined scale-down to CPU (switchTo returned false) — will retry next tick", engineId);
        }
    }

    /** Stops the monitoring tick and releases the scheduler thread. Registered engines are left as-is. */
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
