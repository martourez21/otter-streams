package com.codedstream.otterstream.inference.exception;

/**
 * Exception thrown when inference operations fail.
 *
 * <p>This can occur due to:
 * <ul>
 *   <li>Invalid input data</li>
 *   <li>Model execution errors</li>
 *   <li>Timeout exceeded</li>
 *   <li>Resource constraints</li>
 *   <li>Network issues (for remote models)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * try {
 *     InferenceResult result = engine.infer(inputs);
 * } catch (InferenceException e) {
 *     log.error("Inference failed: {}", e.getMessage(), e);
 *     // Handle error - maybe use default prediction
 * }
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class InferenceException extends Exception {
    /**
     * Constructs an inference exception with a message.
     *
     * @param message error description
     */
    public InferenceException(String message) {
        super(message);
    }

    /**
     * Constructs an inference exception with a message and cause.
     *
     * @param message error description
     * @param cause underlying exception
     */
    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}