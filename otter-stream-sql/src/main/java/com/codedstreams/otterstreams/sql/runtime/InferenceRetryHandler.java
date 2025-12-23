package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.exception.InferenceException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Supplier;

/**
 * Handles retry logic for failed inference operations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class InferenceRetryHandler {
    private static final Logger LOG = LoggerFactory.getLogger(InferenceRetryHandler.class);

    private final int maxRetries;
    private final long baseBackoffMs;
    private final double backoffMultiplier;

    public InferenceRetryHandler(int maxRetries, long baseBackoffMs, double backoffMultiplier) {
        this.maxRetries = maxRetries;
        this.baseBackoffMs = baseBackoffMs;
        this.backoffMultiplier = backoffMultiplier;
    }

    /**
     * Executes operation with retry logic.
     */
    public <T> T executeWithRetry(Supplier<T> operation, String operationName) throws InferenceException {
        int attempt = 0;
        Exception lastException = null;

        while (attempt < maxRetries) {
            try {
                return operation.get();
            } catch (Exception e) {
                lastException = e;
                attempt++;

                if (attempt < maxRetries) {
                    long backoffMs = calculateBackoff(attempt);
                    LOG.warn("Operation '{}' failed (attempt {}/{}), retrying in {}ms: {}",
                            operationName, attempt, maxRetries, backoffMs, e.getMessage());

                    try {
                        Thread.sleep(backoffMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new InferenceException("Retry interrupted", ie);
                    }
                } else {
                    LOG.error("Operation '{}' failed after {} attempts", operationName, maxRetries, e);
                }
            }
        }

        throw new InferenceException(
                String.format("Operation '%s' failed after %d attempts", operationName, maxRetries),
                lastException
        );
    }

    private long calculateBackoff(int attempt) {
        return (long) (baseBackoffMs * Math.pow(backoffMultiplier, attempt - 1));
    }

    /**
     * Checks if exception is retryable.
     */
    public static boolean isRetryable(Exception e) {
        // Network errors, timeouts, temporary unavailability are retryable
        String message = e.getMessage();
        if (message == null) return false;

        String lowerMessage = message.toLowerCase();
        return lowerMessage.contains("timeout") ||
                lowerMessage.contains("connection refused") ||
                lowerMessage.contains("temporarily unavailable") ||
                lowerMessage.contains("service unavailable") ||
                lowerMessage.contains("503");
    }
}
