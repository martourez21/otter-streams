package com.codedstream.otterstream.runtime.hardware;

/**
 * Implemented by an {@link com.codedstream.otterstream.inference.engine.InferenceEngine} that
 * supports running on more than one {@link ExecutionTarget} and wants
 * {@link ExecutionTargetManager} to automatically move it between them based on utilization.
 *
 * <p>This is an <em>opt-in</em> interface, not a requirement — engines that don't implement it
 * are simply never considered for auto-scaling. Only providers with a real CPU fallback path
 * (e.g. ONNX Runtime's CPU/CUDA execution providers) can meaningfully implement
 * {@link #switchTo}; providers without one should not implement this interface at all rather
 * than implementing it as a no-op, so {@link ExecutionTargetManager} doesn't report false
 * scale-down success.
 *
 * @since 0.1.0
 */
public interface ExecutionTargetAware {

    /** @return where this engine is currently executing */
    ExecutionTarget getCurrentExecutionTarget();

    /**
     * Attempts to move execution to the given target. Implementations should make this safe to
     * call while the engine may be actively serving requests — e.g. by completing in-flight
     * calls on the old target before releasing GPU resources, similar in spirit to
     * {@code LifecycleManager}'s graceful-drain approach to hot swaps.
     *
     * @param target the target to switch to
     * @return true if the switch succeeded; false if it was not possible right now (the caller
     *         should not assume the engine is broken — a false return means "try again later,"
     *         not "failed permanently")
     */
    boolean switchTo(ExecutionTarget target);

    /**
     * @return a recent utilization reading in [0.0, 1.0] for whatever {@link #getCurrentExecutionTarget()}
     *         currently is (GPU utilization while on GPU, otherwise a CPU-relative load
     *         figure). {@link ExecutionTargetManager} uses this, sampled on a fixed tick, to
     *         decide when an engine has been idle long enough to scale down.
     */
    double getRecentUtilization();
}
