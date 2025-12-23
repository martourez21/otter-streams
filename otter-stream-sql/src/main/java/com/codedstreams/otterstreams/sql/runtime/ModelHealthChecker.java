package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Performs periodic health checks on loaded models.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ModelHealthChecker {
    private static final Logger LOG = LoggerFactory.getLogger(ModelHealthChecker.class);
    private static final ModelHealthChecker INSTANCE = new ModelHealthChecker();

    private final ConcurrentHashMap<String, HealthStatus> healthStatusMap;
    private final ScheduledExecutorService scheduler;
    private volatile boolean started = false;

    private ModelHealthChecker() {
        this.healthStatusMap = new ConcurrentHashMap<>();
        this.scheduler = Executors.newScheduledThreadPool(1);
    }

    public static ModelHealthChecker getInstance() {
        return INSTANCE;
    }

    public void start(long checkIntervalSeconds) {
        if (!started) {
            scheduler.scheduleAtFixedRate(
                    this::performHealthChecks,
                    checkIntervalSeconds,
                    checkIntervalSeconds,
                    TimeUnit.SECONDS
            );
            started = true;
            LOG.info("Model health checker started with interval: {}s", checkIntervalSeconds);
        }
    }

    public void stop() {
        scheduler.shutdown();
        started = false;
        LOG.info("Model health checker stopped");
    }

    public void registerModel(String modelName, InferenceEngine<?> engine) {
        healthStatusMap.put(modelName, new HealthStatus(modelName, engine));
    }

    public void unregisterModel(String modelName) {
        healthStatusMap.remove(modelName);
    }

    public HealthStatus getHealthStatus(String modelName) {
        return healthStatusMap.get(modelName);
    }

    private void performHealthChecks() {
        for (Map.Entry<String, HealthStatus> entry : healthStatusMap.entrySet()) {
            try {
                HealthStatus status = entry.getValue();
                boolean healthy = checkModelHealth(status.getEngine());
                status.updateHealth(healthy);

                if (!healthy) {
                    LOG.warn("Model '{}' failed health check", entry.getKey());
                }
            } catch (Exception e) {
                LOG.error("Error checking health for model: {}", entry.getKey(), e);
            }
        }
    }

    private boolean checkModelHealth(InferenceEngine<?> engine) {
        try {
            // Simple health check: verify engine is ready
            return engine.isReady();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Health status for a single model.
     */
    public static class HealthStatus {
        private final String modelName;
        private final InferenceEngine<?> engine;
        private volatile boolean healthy = true;
        private volatile long lastCheckTime = 0;
        private volatile int consecutiveFailures = 0;

        public HealthStatus(String modelName, InferenceEngine<?> engine) {
            this.modelName = modelName;
            this.engine = engine;
        }

        void updateHealth(boolean healthy) {
            this.healthy = healthy;
            this.lastCheckTime = System.currentTimeMillis();

            if (healthy) {
                consecutiveFailures = 0;
            } else {
                consecutiveFailures++;
            }
        }

        public String getModelName() { return modelName; }
        public InferenceEngine<?> getEngine() { return engine; }
        public boolean isHealthy() { return healthy; }
        public long getLastCheckTime() { return lastCheckTime; }
        public int getConsecutiveFailures() { return consecutiveFailures; }
    }
}
