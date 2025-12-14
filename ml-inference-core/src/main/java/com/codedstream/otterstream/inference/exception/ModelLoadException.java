package com.codedstream.otterstream.inference.exception;

/**
 * Exception thrown when model loading fails.
 *
 * <p>Common causes:
 * <ul>
 *   <li>Model file not found</li>
 *   <li>Corrupted model file</li>
 *   <li>Incompatible model format</li>
 *   <li>Insufficient memory</li>
 *   <li>Missing dependencies</li>
 * </ul>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class ModelLoadException extends Exception {
    /**
     * Constructs a model load exception with a message.
     *
     * @param message error description
     */
    public ModelLoadException(String message) {
        super(message);
    }

    /**
     * Constructs a model load exception with a message and cause.
     *
     * @param message error description
     * @param cause underlying exception
     */
    public ModelLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}