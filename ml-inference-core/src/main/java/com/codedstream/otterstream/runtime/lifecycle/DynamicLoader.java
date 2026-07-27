package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.runtime.spi.ModelReference;
import com.codedstream.otterstream.runtime.spi.ModelRegistry;
import java.time.Duration;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Polls a {@link ModelRegistry} on a per-model schedule and automatically deploys newly
 * discovered versions through a {@link LifecycleManager} — this is the "dynamic loading"
 * milestone: instead of an operator calling {@code deploy()} by hand every time a new model
 * version is published, {@code DynamicLoader} notices it and rolls it out on its own, going
 * through the same validate → warm → atomic swap → retire path as a manual deployment.
 *
 * <p>Each watched model id gets its own scheduled poll; polls for a given model never overlap
 * (if a deploy triggered by one poll is still running when the next poll fires, that poll is
 * skipped rather than queued).
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * DynamicLoader loader = new DynamicLoader(lifecycleManager, modelRegistry);
 * loader.watch("fraud-detector", Duration.ofSeconds(30));
 * // ... new versions registered in modelRegistry are now picked up automatically ...
 * loader.unwatch("fraud-detector");
 * loader.shutdown();
 * }</pre>
 *
 * @since 0.1.0
 * @see ModelRegistry
 * @see LifecycleManager
 */
public class DynamicLoader {

    private static final Logger LOG = LoggerFactory.getLogger(DynamicLoader.class);

    private final LifecycleManager lifecycleManager;
    private final ModelRegistry modelRegistry;
    private final ScheduledExecutorService scheduler;
    private final ConcurrentHashMap<String, ScheduledFuture<?>> watches = new ConcurrentHashMap<>();
    private final Set<String> pollingInProgress = ConcurrentHashMap.newKeySet();

    public DynamicLoader(LifecycleManager lifecycleManager, ModelRegistry modelRegistry) {
        this.lifecycleManager = Objects.requireNonNull(lifecycleManager, "lifecycleManager cannot be null");
        this.modelRegistry = Objects.requireNonNull(modelRegistry, "modelRegistry cannot be null");
        ThreadFactory threadFactory = runnable -> {
            Thread t = new Thread(runnable, "otter-dynamic-loader");
            t.setDaemon(true);
            return t;
        };
        this.scheduler = Executors.newScheduledThreadPool(1, threadFactory);
    }

    /**
     * Starts polling the registry for a model id on a fixed delay. Replaces any existing watch
     * for the same model id.
     *
     * @param modelId      the logical model id to watch (resolved against the registry as
     *                     {@link ModelReference#of(String)}, i.e. "whatever is latest")
     * @param pollInterval delay between the end of one poll and the start of the next
     */
    public void watch(String modelId, Duration pollInterval) {
        Objects.requireNonNull(modelId, "modelId cannot be null");
        Objects.requireNonNull(pollInterval, "pollInterval cannot be null");
        unwatch(modelId);
        long millis = Math.max(1, pollInterval.toMillis());
        ScheduledFuture<?> future = scheduler.scheduleWithFixedDelay(
                () -> pollOnce(modelId), millis, millis, TimeUnit.MILLISECONDS);
        watches.put(modelId, future);
        LOG.info("Watching model '{}' for new versions every {}ms", modelId, millis);
    }

    /**
     * Stops polling for a model id. No-op if it wasn't being watched.
     *
     * @param modelId the model id to stop watching
     */
    public void unwatch(String modelId) {
        ScheduledFuture<?> future = watches.remove(modelId);
        if (future != null) {
            future.cancel(false);
            LOG.info("Stopped watching model '{}'", modelId);
        }
    }

    public boolean isWatching(String modelId) {
        return watches.containsKey(modelId);
    }

    public Set<String> getWatchedModelIds() {
        return java.util.Collections.unmodifiableSet(watches.keySet());
    }

    private void pollOnce(String modelId) {
        if (!pollingInProgress.add(modelId)) {
            LOG.debug("Skipping poll for model '{}': previous poll/deploy still in progress", modelId);
            return;
        }
        try {
            ModelConfig latest;
            try {
                latest = modelRegistry.resolve(ModelReference.of(modelId));
            } catch (Exception e) {
                LOG.warn("Dynamic loader failed to resolve model '{}' from registry: {}", modelId, e.getMessage());
                return;
            }

            String latestVersion = latest.getModelVersion();
            String activeVersion = lifecycleManager.isDeployed(modelId)
                    ? safeActiveVersion(modelId)
                    : null;

            if (!Objects.equals(latestVersion, activeVersion)) {
                LOG.info("Dynamic loader detected new version for model '{}': '{}' -> '{}'",
                        modelId, activeVersion, latestVersion);
                try {
                    lifecycleManager.deploy(latest);
                } catch (Exception e) {
                    LOG.warn("Dynamic loader failed to deploy model '{}' version '{}': {}",
                            modelId, latestVersion, e.getMessage(), e);
                }
            }
        } finally {
            pollingInProgress.remove(modelId);
        }
    }

    private String safeActiveVersion(String modelId) {
        ModelVersion version = lifecycleManager.getManagedModel(modelId).getActiveVersion();
        return version != null ? version.getVersion() : null;
    }

    /**
     * Cancels every active watch and shuts down the polling thread. The loader should not be
     * reused afterward.
     */
    public void shutdown() {
        watches.values().forEach(f -> f.cancel(false));
        watches.clear();
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
