package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A model identity (a stable {@code modelId}) whose underlying {@link InferenceEngine} can be
 * hot-swapped, canaried, and shadowed over time without callers ever holding a stale reference.
 *
 * <p>Three independent traffic slots are tracked:
 * <ul>
 *   <li><b>primary</b> — always present once a version is deployed; serves the rest of traffic</li>
 *   <li><b>canary</b> — optional; receives a configurable percentage of traffic
 *       (see {@link LifecycleManager#deployCanary})</li>
 *   <li><b>shadow</b> — optional; receives a sampled copy of traffic asynchronously, purely for
 *       comparison — never affects what's returned to the caller
 *       (see {@link LifecycleManager#deployShadow})</li>
 * </ul>
 *
 * <p>Callers should always go through {@link #infer(Map)} rather than caching an engine
 * themselves — the whole point of {@link ManagedModel} is that routing can change out from
 * under you between two calls, atomically and without ever exposing a partially-initialized
 * engine.
 *
 * @since 0.1.0
 * @see LifecycleManager
 */
public class ManagedModel {

    private static final Logger LOG = LoggerFactory.getLogger(ManagedModel.class);

    private final String modelId;
    private final ExecutorService shadowExecutor;
    private final ShadowResultHandler shadowResultHandler;

    private final AtomicReference<EngineHandle> primary = new AtomicReference<>();
    private final AtomicReference<EngineHandle> canary = new AtomicReference<>();
    private final AtomicInteger canaryTrafficPercent = new AtomicInteger(0);
    private final AtomicReference<EngineHandle> shadow = new AtomicReference<>();
    private volatile double shadowSampleRate = 0.0;

    private final List<ModelVersion> history = new CopyOnWriteArrayList<>();

    ManagedModel(String modelId, ExecutorService shadowExecutor, ShadowResultHandler shadowResultHandler) {
        this.modelId = Objects.requireNonNull(modelId, "modelId cannot be null");
        this.shadowExecutor = shadowExecutor;
        this.shadowResultHandler = shadowResultHandler;
    }

    public String getModelId() {
        return modelId;
    }

    /**
     * @return the currently active primary engine for this model
     * @throws IllegalStateException if no version has ever been successfully deployed
     */
    public InferenceEngine<?> getActiveEngine() {
        return requirePrimary().getEngine();
    }

    /**
     * @return metadata for the currently active primary version, or null if nothing has been
     *         deployed yet
     */
    public ModelVersion getActiveVersion() {
        EngineHandle handle = primary.get();
        return handle != null ? handle.getVersion() : null;
    }

    /**
     * @return metadata for the currently active canary version, or null if no canary is deployed
     */
    public ModelVersion getCanaryVersion() {
        EngineHandle handle = canary.get();
        return handle != null ? handle.getVersion() : null;
    }

    /**
     * @return the canary's traffic share (0-100); 0 if no canary is deployed
     */
    public int getCanaryTrafficPercent() {
        return canaryTrafficPercent.get();
    }

    /**
     * @return metadata for the currently active shadow version, or null if no shadow is deployed
     */
    public ModelVersion getShadowVersion() {
        EngineHandle handle = shadow.get();
        return handle != null ? handle.getVersion() : null;
    }

    /**
     * @return past versions that were once active and have since been retired (or failed),
     *         oldest first. Each retains the {@link com.codedstream.otterstream.inference.config.ModelConfig}
     *         it was deployed from, enabling {@link LifecycleManager#rollback(String)}.
     */
    public List<ModelVersion> getHistory() {
        return Collections.unmodifiableList(history);
    }

    /**
     * Runs inference, routed between primary and canary (if any) according to
     * {@link #getCanaryTrafficPercent()}, and — if a shadow deployment is active — fires a
     * sampled, asynchronous, non-blocking copy of the request at the shadow engine for
     * comparison. The shadow call never affects the value returned here.
     *
     * @param inputs inference inputs
     * @return inference result from whichever engine (primary or canary) served this call
     * @throws InferenceException if no version is deployed, or the serving engine call fails
     */
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        EngineHandle serving = selectServingHandle();
        serving.enter();
        InferenceResult result;
        try {
            result = serving.getEngine().infer(inputs);
        } finally {
            serving.exit();
        }
        maybeShadow(inputs, result);
        return result;
    }

    private EngineHandle selectServingHandle() {
        EngineHandle canaryHandle = canary.get();
        int percent = canaryTrafficPercent.get();
        if (canaryHandle != null && percent > 0
                && ThreadLocalRandom.current().nextInt(100) < percent) {
            return canaryHandle;
        }
        return requirePrimary();
    }

    private EngineHandle requirePrimary() {
        EngineHandle handle = primary.get();
        if (handle == null) {
            throw new IllegalStateException("No active engine deployed for model '" + modelId + "'");
        }
        return handle;
    }

    private void maybeShadow(Map<String, Object> inputs, InferenceResult primaryResult) {
        EngineHandle shadowHandle = shadow.get();
        if (shadowHandle == null || shadowExecutor == null) {
            return;
        }
        double rate = shadowSampleRate;
        if (rate <= 0.0) {
            return;
        }
        if (rate < 1.0 && ThreadLocalRandom.current().nextDouble() >= rate) {
            return;
        }
        try {
            shadowExecutor.submit(() -> runShadow(shadowHandle, inputs, primaryResult));
        } catch (Exception e) {
            // executor rejected the task (e.g. shutting down) — never let shadow scheduling
            // affect the primary/canary response path.
            LOG.debug("Failed to schedule shadow inference for model '{}': {}", modelId, e.getMessage());
        }
    }

    private void runShadow(EngineHandle shadowHandle, Map<String, Object> inputs, InferenceResult primaryResult) {
        shadowHandle.enter();
        try {
            InferenceResult shadowResult = shadowHandle.getEngine().infer(inputs);
            if (shadowResultHandler != null) {
                shadowResultHandler.handle(modelId, shadowHandle.getVersion().getVersion(), primaryResult, shadowResult, null);
            }
        } catch (Exception e) {
            if (shadowResultHandler != null) {
                shadowResultHandler.handle(modelId, shadowHandle.getVersion().getVersion(), primaryResult, null, e);
            }
        } finally {
            shadowHandle.exit();
        }
    }

    // ------------------------------------------------------------------
    // Package-private mutators — called only by LifecycleManager
    // ------------------------------------------------------------------

    /**
     * Atomically publishes a new engine/version as primary, moving the previous primary (if any)
     * into history.
     *
     * @return the previously-active primary handle, or null if this is the first deployment
     */
    EngineHandle swapPrimary(EngineHandle newHandle) {
        EngineHandle previous = primary.getAndSet(newHandle);
        if (previous != null) {
            previous.getVersion().setStatus(ModelVersion.Status.RETIRED);
            history.add(previous.getVersion());
        }
        return previous;
    }

    /**
     * Sets (or replaces) the canary engine/version and its traffic share.
     *
     * @return the previous canary handle, or null if none was set
     */
    EngineHandle setCanary(EngineHandle newHandle, int trafficPercent) {
        EngineHandle previous = canary.getAndSet(newHandle);
        canaryTrafficPercent.set(trafficPercent);
        return previous;
    }

    /**
     * Clears the canary slot (used by both promote and rollback-canary flows).
     *
     * @return the canary handle that was cleared, or null if none was set
     */
    EngineHandle clearCanary() {
        EngineHandle previous = canary.getAndSet(null);
        canaryTrafficPercent.set(0);
        return previous;
    }

    /**
     * Sets (or replaces) the shadow engine/version and its sample rate.
     *
     * @return the previous shadow handle, or null if none was set
     */
    EngineHandle setShadow(EngineHandle newHandle, double sampleRate) {
        EngineHandle previous = shadow.getAndSet(newHandle);
        this.shadowSampleRate = sampleRate;
        return previous;
    }

    /**
     * Clears the shadow slot.
     *
     * @return the shadow handle that was cleared, or null if none was set
     */
    EngineHandle clearShadow() {
        EngineHandle previous = shadow.getAndSet(null);
        this.shadowSampleRate = 0.0;
        return previous;
    }

    EngineHandle getPrimaryHandle() {
        return primary.get();
    }

    EngineHandle getCanaryHandle() {
        return canary.get();
    }

    EngineHandle getShadowHandle() {
        return shadow.get();
    }

    void recordFailedVersion(ModelVersion version) {
        history.add(version);
    }
}
