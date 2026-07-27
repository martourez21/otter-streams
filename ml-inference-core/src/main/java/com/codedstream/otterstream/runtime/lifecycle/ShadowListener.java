package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.model.InferenceResult;

/**
 * Observer callback fired after a shadow inference call completes (or fails), letting you wire
 * up comparison logging/metrics between a primary version and a shadow version without changing
 * {@link LifecycleManager} or {@link ManagedModel}.
 *
 * <p>Shadow calls run asynchronously and never affect what's returned to the caller of
 * {@link ManagedModel#infer(java.util.Map)} — this listener is purely observational.
 *
 * @since 0.1.0
 * @see LifecycleManager#deployShadow(String, com.codedstream.otterstream.inference.config.ModelConfig, double)
 */
public interface ShadowListener {

    /**
     * @param modelId       the model id the shadow deployment belongs to
     * @param shadowVersion the shadow version string that produced (or attempted) this result
     * @param primaryResult the result the caller actually received, from the primary/canary engine
     * @param shadowResult  the shadow engine's result, or null if the shadow call failed
     * @param error         the failure from the shadow call, or null if it succeeded
     */
    void onShadowResult(String modelId, String shadowVersion, InferenceResult primaryResult,
                         InferenceResult shadowResult, Throwable error);
}
