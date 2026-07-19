package com.codedstream.otterstream.runtime.lifecycle;

/**
 * Observer callback for model deployment lifecycle events, useful for wiring up logging,
 * metrics, or alerting around deployments without changing {@link LifecycleManager} itself.
 *
 * <p>All methods have empty default implementations, so listeners only need to override the
 * events they care about.
 *
 * @since 0.1.0
 * @see LifecycleManager#addListener(LifecycleListener)
 */
public interface LifecycleListener {

    /** Called when a new version begins validation (engine construction + initialize()). */
    default void onValidating(String modelId, ModelVersion version) {
    }

    /** Called when validation succeeded and an optional warmup probe is about to run. */
    default void onWarming(String modelId, ModelVersion version) {
    }

    /** Called immediately after a version is atomically swapped in as active. */
    default void onActivated(String modelId, ModelVersion version) {
    }

    /** Called after a previously-active version's engine has been closed/released. */
    default void onRetired(String modelId, ModelVersion version) {
    }

    /** Called when validation or warmup fails; the previously-active version (if any) is untouched. */
    default void onFailed(String modelId, ModelVersion version, Throwable cause) {
    }
}
