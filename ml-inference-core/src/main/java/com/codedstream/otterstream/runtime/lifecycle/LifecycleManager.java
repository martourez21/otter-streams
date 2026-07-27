package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.DeploymentException;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.runtime.registry.ProviderRegistry;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.Collections;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Orchestrates the deployment lifecycle of a model version:
 *
 * <pre>
 * resolve provider → validate (initialize) → warm (optional probe) → atomic swap → drain → retire old
 * </pre>
 *
 * <p>This is the separation the v1.0 architecture calls for: {@link InferenceEngine}
 * implementations know nothing about versioning, traffic splitting, or deployment safety —
 * they just load a model and answer {@code infer()} calls. {@link LifecycleManager} is solely
 * responsible for deciding <em>when</em> a new version is safe to receive traffic and for making
 * cutovers atomic and graceful.
 *
 * <p>Beyond the base validate/warm/swap/retire flow, this class also supports:
 * <ul>
 *   <li><b>Graceful draining</b> — a retiring engine is only closed once its in-flight
 *       requests finish (or a timeout elapses), so hot swaps under load never sever
 *       in-progress calls.</li>
 *   <li><b>Rollback</b> — {@link #rollback(String)} redeploys the last known-good
 *       (previously active) version.</li>
 *   <li><b>Canary deployments</b> — {@link #deployCanary} runs a new version alongside the
 *       current primary at a configurable traffic percentage; {@link #promoteCanary} or
 *       {@link #rollbackCanary} conclude it.</li>
 *   <li><b>Shadow deployments</b> — {@link #deployShadow} mirrors a sampled copy of traffic to
 *       a version asynchronously, purely for comparison, never affecting served responses.</li>
 * </ul>
 *
 * @since 0.1.0
 */
public class LifecycleManager {

    private static final Logger LOG = LoggerFactory.getLogger(LifecycleManager.class);
    private static final long DEFAULT_DRAIN_TIMEOUT_MS = 5_000L;
    private static final long DRAIN_POLL_INTERVAL_MS = 10L;

    private final ProviderRegistry providerRegistry;
    private final ConcurrentHashMap<String, ManagedModel> managedModels = new ConcurrentHashMap<>();
    private final List<LifecycleListener> listeners = new CopyOnWriteArrayList<>();
    private final List<ShadowListener> shadowListeners = new CopyOnWriteArrayList<>();
    private final ExecutorService shadowExecutor;
    private final long drainTimeoutMillis;

    public LifecycleManager(ProviderRegistry providerRegistry) {
        this(providerRegistry, DEFAULT_DRAIN_TIMEOUT_MS);
    }

    public LifecycleManager(ProviderRegistry providerRegistry, long drainTimeoutMillis) {
        this.providerRegistry = Objects.requireNonNull(providerRegistry, "providerRegistry cannot be null");
        this.drainTimeoutMillis = drainTimeoutMillis;
        ThreadFactory shadowThreadFactory = runnable -> {
            Thread t = new Thread(runnable, "otter-shadow-inference");
            t.setDaemon(true);
            return t;
        };
        this.shadowExecutor = Executors.newCachedThreadPool(shadowThreadFactory);
    }

    public void addListener(LifecycleListener listener) {
        listeners.add(Objects.requireNonNull(listener));
    }

    public void addShadowListener(ShadowListener listener) {
        shadowListeners.add(Objects.requireNonNull(listener));
    }

    // ------------------------------------------------------------------
    // Primary deployment (Milestone 3 base flow, now EngineHandle-based)
    // ------------------------------------------------------------------

    /**
     * Deploys a model version with no warmup probe (validation is still performed).
     */
    public ManagedModel deploy(ModelConfig config) throws DeploymentException {
        return deploy(config, Map.of());
    }

    /**
     * Deploys (or hot-swaps) a model version as the primary.
     *
     * <p>The new version is validated and warmed independently of whatever is currently active;
     * only once that succeeds is it atomically swapped in. The previously-active engine is then
     * drained (in-flight requests allowed to finish, up to the configured drain timeout) and
     * closed.
     *
     * @param config      the model configuration to deploy
     * @param warmupProbe optional sample input map exercised once before the swap; empty map
     *                    skips warmup
     * @return the {@link ManagedModel} now serving this version
     * @throws DeploymentException if resolving a provider, validation, or warmup fails
     */
    public ManagedModel deploy(ModelConfig config, Map<String, Object> warmupProbe) throws DeploymentException {
        Objects.requireNonNull(config, "config cannot be null");
        String modelId = config.getModelId();
        ManagedModel managed = managedModel(modelId);

        EngineHandle handle = validateAndWarm(config, warmupProbe, managed);

        handle.getVersion().setStatus(ModelVersion.Status.ACTIVE);
        EngineHandle previous = managed.swapPrimary(handle);
        LOG.info("Activated model '{}' version '{}'", modelId, handle.getVersion().getVersion());
        fireActivated(modelId, handle.getVersion());

        if (previous != null) {
            retire(modelId, previous);
        }

        return managed;
    }

    /**
     * Re-deploys the last previously-active (retired) version for a model, effectively undoing
     * the most recent hot swap. This goes through the full validate/warm flow again — it does
     * not reuse the old, already-closed engine instance.
     *
     * @param modelId the model id to roll back
     * @return the managed model, now back on the rolled-back version
     * @throws DeploymentException  if the rollback deployment itself fails validation/warmup
     * @throws IllegalStateException if there is no prior retired version with a usable config
     */
    public ManagedModel rollback(String modelId) throws DeploymentException {
        ManagedModel managed = getManagedModel(modelId);
        List<ModelVersion> history = managed.getHistory();
        for (int i = history.size() - 1; i >= 0; i--) {
            ModelVersion candidate = history.get(i);
            if (candidate.getStatus() == ModelVersion.Status.RETIRED && candidate.getConfig() != null) {
                LOG.info("Rolling back model '{}' to version '{}'", modelId, candidate.getVersion());
                return deploy(candidate.getConfig());
            }
        }
        throw new IllegalStateException("No previous retired version available to roll back to for model '" + modelId + "'");
    }

    // ------------------------------------------------------------------
    // Canary deployments (Milestone 6)
    // ------------------------------------------------------------------

    /**
     * Deploys a candidate version alongside the current primary, receiving {@code trafficPercent}
     * percent of {@link ManagedModel#infer(Map)} calls (routed randomly per-call).
     *
     * @param config         the candidate model configuration
     * @param trafficPercent percentage of traffic (0-100) to route to the canary
     * @return the managed model, now serving both primary and canary
     * @throws DeploymentException     if validation/warmup of the canary fails
     * @throws IllegalArgumentException if trafficPercent is outside 0-100
     */
    public ManagedModel deployCanary(ModelConfig config, int trafficPercent) throws DeploymentException {
        if (trafficPercent < 0 || trafficPercent > 100) {
            throw new IllegalArgumentException("trafficPercent must be between 0 and 100, was " + trafficPercent);
        }
        Objects.requireNonNull(config, "config cannot be null");
        String modelId = config.getModelId();
        ManagedModel managed = managedModel(modelId);

        EngineHandle handle = validateAndWarm(config, Map.of(), managed);
        handle.getVersion().setStatus(ModelVersion.Status.ACTIVE);

        EngineHandle previousCanary = managed.setCanary(handle, trafficPercent);
        LOG.info("Deployed canary for model '{}' version '{}' at {}% traffic",
                modelId, handle.getVersion().getVersion(), trafficPercent);
        fireActivated(modelId, handle.getVersion());

        if (previousCanary != null) {
            retire(modelId, previousCanary);
        }
        return managed;
    }

    /**
     * Promotes the current canary to primary: the canary's engine becomes the new primary
     * (no re-validation needed, it's already warm and serving traffic), the canary slot is
     * cleared, and the old primary is drained and retired.
     *
     * @param modelId the model id whose canary should be promoted
     * @throws IllegalStateException if no canary is currently deployed
     */
    public void promoteCanary(String modelId) {
        ManagedModel managed = getManagedModel(modelId);
        EngineHandle canaryHandle = managed.clearCanary();
        if (canaryHandle == null) {
            throw new IllegalStateException("No canary deployed for model '" + modelId + "'");
        }
        EngineHandle previousPrimary = managed.swapPrimary(canaryHandle);
        LOG.info("Promoted canary version '{}' to primary for model '{}'", canaryHandle.getVersion().getVersion(), modelId);
        fireActivated(modelId, canaryHandle.getVersion());
        if (previousPrimary != null) {
            retire(modelId, previousPrimary);
        }
    }

    /**
     * Discards the current canary without touching the primary: closes (after draining) the
     * canary's engine and clears the canary slot.
     *
     * @param modelId the model id whose canary should be discarded
     * @throws IllegalStateException if no canary is currently deployed
     */
    public void rollbackCanary(String modelId) {
        ManagedModel managed = getManagedModel(modelId);
        EngineHandle canaryHandle = managed.clearCanary();
        if (canaryHandle == null) {
            throw new IllegalStateException("No canary deployed for model '" + modelId + "'");
        }
        LOG.info("Discarded canary version '{}' for model '{}'", canaryHandle.getVersion().getVersion(), modelId);
        retire(modelId, canaryHandle);
    }

    // ------------------------------------------------------------------
    // Shadow deployments (Milestone 6)
    // ------------------------------------------------------------------

    /**
     * Deploys a shadow version: every call to {@link ManagedModel#infer(Map)} that's sampled in
     * (per {@code sampleRate}) is also, asynchronously and without affecting the caller, sent to
     * this engine. Results are reported to any registered {@link ShadowListener}.
     *
     * @param modelId    the model id to attach the shadow to (must already have a primary deployed)
     * @param config     the shadow model configuration
     * @param sampleRate fraction of traffic to mirror to the shadow, from 0.0 (off) to 1.0 (all)
     * @return the managed model, now shadowing traffic to this version
     * @throws DeploymentException     if validation/warmup of the shadow fails
     * @throws IllegalArgumentException if sampleRate is outside 0.0-1.0
     */
    public ManagedModel deployShadow(String modelId, ModelConfig config, double sampleRate) throws DeploymentException {
        if (sampleRate < 0.0 || sampleRate > 1.0) {
            throw new IllegalArgumentException("sampleRate must be between 0.0 and 1.0, was " + sampleRate);
        }
        Objects.requireNonNull(config, "config cannot be null");
        ManagedModel managed = managedModel(modelId);

        EngineHandle handle = validateAndWarm(config, Map.of(), managed);
        handle.getVersion().setStatus(ModelVersion.Status.ACTIVE);

        EngineHandle previousShadow = managed.setShadow(handle, sampleRate);
        LOG.info("Deployed shadow for model '{}' version '{}' at sample rate {}",
                modelId, handle.getVersion().getVersion(), sampleRate);
        fireActivated(modelId, handle.getVersion());

        if (previousShadow != null) {
            retire(modelId, previousShadow);
        }
        return managed;
    }

    /**
     * Stops shadowing traffic for a model: clears the shadow slot and retires (drains + closes)
     * its engine. No-op if no shadow is deployed.
     *
     * @param modelId the model id to stop shadowing
     */
    public void stopShadow(String modelId) {
        ManagedModel managed = getManagedModel(modelId);
        EngineHandle shadowHandle = managed.clearShadow();
        if (shadowHandle == null) {
            return;
        }
        LOG.info("Stopped shadow version '{}' for model '{}'", shadowHandle.getVersion().getVersion(), modelId);
        retire(modelId, shadowHandle);
    }

    // ------------------------------------------------------------------
    // Shared validate/warm + retire machinery
    // ------------------------------------------------------------------

    private EngineHandle validateAndWarm(ModelConfig config, Map<String, Object> warmupProbe, ManagedModel managed) throws DeploymentException {
        String modelId = config.getModelId();
        ModelVersion version = new ModelVersion(
                config.getModelVersion() != null ? config.getModelVersion() : "unversioned", config);

        InferenceEngine<?> engine;
        try {
            engine = providerRegistry.createEngine(config.getFormat());

            LOG.info("Validating model '{}' version '{}' (format={})", modelId, version.getVersion(), config.getFormat());
            fireValidating(modelId, version);
            engine.initialize(config);
            if (!engine.isReady()) {
                throw new DeploymentException(
                        "Engine reported not-ready after initialize() for model '" + modelId + "'");
            }

            version.setStatus(ModelVersion.Status.WARMING);
            LOG.info("Warming model '{}' version '{}'", modelId, version.getVersion());
            fireWarming(modelId, version);
            if (warmupProbe != null && !warmupProbe.isEmpty()) {
                engine.infer(warmupProbe);
            }
        } catch (DeploymentException e) {
            version.setStatus(ModelVersion.Status.FAILED);
            managed.recordFailedVersion(version);
            fireFailed(modelId, version, e);
            throw e;
        } catch (Exception e) {
            version.setStatus(ModelVersion.Status.FAILED);
            managed.recordFailedVersion(version);
            fireFailed(modelId, version, e);
            throw new DeploymentException(
                    "Failed to deploy model '" + modelId + "' version '" + version.getVersion() + "'", e);
        }

        return new EngineHandle(engine, version);
    }

    /**
     * Drains (waits for in-flight requests to finish, up to the configured timeout) and closes
     * a retiring engine handle, then fires {@link LifecycleListener#onRetired}.
     */
    private void retire(String modelId, EngineHandle handle) {
        long deadline = System.currentTimeMillis() + drainTimeoutMillis;
        while (handle.getInFlightCount() > 0 && System.currentTimeMillis() < deadline) {
            try {
                Thread.sleep(DRAIN_POLL_INTERVAL_MS);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        int remaining = handle.getInFlightCount();
        if (remaining > 0) {
            LOG.warn("Force-closing engine for model '{}' version '{}' with {} in-flight request(s) "
                            + "still active after {}ms drain timeout",
                    modelId, handle.getVersion().getVersion(), remaining, drainTimeoutMillis);
        }
        try {
            handle.getEngine().close();
        } catch (InferenceException e) {
            LOG.warn("Failed to cleanly close retired engine for model '{}': {}", modelId, e.getMessage(), e);
        }
        handle.getVersion().setStatus(ModelVersion.Status.RETIRED);
        fireRetired(modelId, handle.getVersion());
    }

    // ------------------------------------------------------------------
    // Accessors / bookkeeping
    // ------------------------------------------------------------------

    private ManagedModel managedModel(String modelId) {
        return managedModels.computeIfAbsent(modelId,
                id -> new ManagedModel(id, shadowExecutor, this::fireShadowResult));
    }

    /**
     * @throws IllegalStateException if nothing has ever been deployed under this id
     */
    public ManagedModel getManagedModel(String modelId) {
        ManagedModel managed = managedModels.get(modelId);
        if (managed == null) {
            throw new IllegalStateException("No model deployed under id '" + modelId + "'");
        }
        return managed;
    }

    public boolean isDeployed(String modelId) {
        return managedModels.containsKey(modelId);
    }

    /**
     * @return the logical ids of every model currently tracked by this manager, in no
     *         particular order
     */
    public Set<String> getDeployedModelIds() {
        return Collections.unmodifiableSet(managedModels.keySet());
    }

    /**
     * Removes a model entirely and closes (after draining) its primary, canary, and shadow
     * engines, if present. Best-effort: close failures are logged, not thrown.
     *
     * @param modelId the model id to undeploy
     */
    public void undeploy(String modelId) {
        ManagedModel managed = managedModels.remove(modelId);
        if (managed == null) {
            return;
        }
        EngineHandle canaryHandle = managed.clearCanary();
        if (canaryHandle != null) {
            retire(modelId, canaryHandle);
        }
        EngineHandle shadowHandle = managed.clearShadow();
        if (shadowHandle != null) {
            retire(modelId, shadowHandle);
        }
        EngineHandle primaryHandle = managed.getPrimaryHandle();
        if (primaryHandle != null) {
            retire(modelId, primaryHandle);
        }
    }

    /**
     * Shuts down the shadow-inference thread pool. Call once when the owning
     * {@link com.codedstream.otterstream.runtime.OtterRuntime} is closed; the manager should not
     * be reused afterward.
     */
    public void shutdown() {
        shadowExecutor.shutdown();
        try {
            if (!shadowExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                shadowExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            shadowExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // ------------------------------------------------------------------
    // Listener dispatch
    // ------------------------------------------------------------------

    private void fireValidating(String modelId, ModelVersion v) {
        for (LifecycleListener l : listeners) {
            l.onValidating(modelId, v);
        }
    }

    private void fireWarming(String modelId, ModelVersion v) {
        for (LifecycleListener l : listeners) {
            l.onWarming(modelId, v);
        }
    }

    private void fireActivated(String modelId, ModelVersion v) {
        for (LifecycleListener l : listeners) {
            l.onActivated(modelId, v);
        }
    }

    private void fireRetired(String modelId, ModelVersion v) {
        for (LifecycleListener l : listeners) {
            l.onRetired(modelId, v);
        }
    }

    private void fireFailed(String modelId, ModelVersion v, Throwable cause) {
        for (LifecycleListener l : listeners) {
            l.onFailed(modelId, v, cause);
        }
    }

    private void fireShadowResult(String modelId, String shadowVersion, InferenceResult primaryResult,
                                   InferenceResult shadowResult, Throwable error) {
        for (ShadowListener l : shadowListeners) {
            try {
                l.onShadowResult(modelId, shadowVersion, primaryResult, shadowResult, error);
            } catch (Exception e) {
                LOG.warn("ShadowListener threw while handling result for model '{}': {}", modelId, e.getMessage(), e);
            }
        }
    }
}
