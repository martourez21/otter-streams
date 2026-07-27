package com.codedstream.otterstream.inference.exception;

/**
 * Exception thrown when a model deployment (validate → warm → swap) fails inside the
 * {@link com.codedstream.otterstream.runtime.lifecycle.LifecycleManager}.
 *
 * <p>This can occur due to:
 * <ul>
 *   <li>No {@link com.codedstream.otterstream.runtime.spi.InferenceProvider} registered for
 *       the requested {@link com.codedstream.otterstream.inference.model.ModelFormat}</li>
 *   <li>Model failing to load/initialize</li>
 *   <li>The engine reporting not-ready after initialization</li>
 *   <li>A warmup probe inference failing</li>
 * </ul>
 *
 * <p>A failed deployment never disturbs a model version that is already active: the
 * lifecycle manager only performs the atomic swap after validation and warmup succeed.
 *
 * @since 0.1.0
 */
public class DeploymentException extends InferenceException {

    /**
     * Constructs a deployment exception with a message.
     *
     * @param message error description
     */
    public DeploymentException(String message) {
        super(message);
    }

    /**
     * Constructs a deployment exception with a message and cause.
     *
     * @param message error description
     * @param cause   underlying exception
     */
    public DeploymentException(String message, Throwable cause) {
        super(message, cause);
    }
}
