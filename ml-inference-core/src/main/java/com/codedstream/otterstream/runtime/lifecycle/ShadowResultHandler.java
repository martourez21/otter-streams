package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.model.InferenceResult;

/**
 * Package-private bridge from {@link ManagedModel}'s async shadow-call completion to
 * {@link LifecycleManager}'s public {@link ShadowListener} list, so {@link ManagedModel} doesn't
 * need to know about listener management.
 *
 * @since 0.1.0
 */
@FunctionalInterface
interface ShadowResultHandler {
    void handle(String modelId, String shadowVersion, InferenceResult primaryResult,
                InferenceResult shadowResult, Throwable error);
}
