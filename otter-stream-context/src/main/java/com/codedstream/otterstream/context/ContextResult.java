package com.codedstream.otterstream.context;

import java.util.Map;
import java.util.Objects;

/**
 * One provider's contribution to an assembled {@link Context} — its data (if it succeeded), how
 * long it took, and its failure (if it didn't). {@link ContextEngine} never lets one provider's
 * failure sink the whole request; every provider's outcome, success or not, ends up as one of
 * these in the final {@link Context}.
 *
 * @param providerId     which provider produced this
 * @param data            the provider's contributed context, empty map on failure
 * @param latencyMicros   how long this provider's fetch took (including any timeout wait)
 * @param succeeded       whether the fetch completed without error or timeout
 * @param failureReason    human-readable failure description, null if succeeded
 * @since 0.1.0
 */
public record ContextResult(
        String providerId, Map<String, Object> data, long latencyMicros, boolean succeeded, String failureReason) {

    public ContextResult {
        Objects.requireNonNull(providerId, "providerId cannot be null");
        data = data == null ? Map.of() : Map.copyOf(data);
    }

    static ContextResult success(String providerId, Map<String, Object> data, long latencyMicros) {
        return new ContextResult(providerId, data, latencyMicros, true, null);
    }

    static ContextResult failure(String providerId, long latencyMicros, String reason) {
        return new ContextResult(providerId, Map.of(), latencyMicros, false, reason);
    }
}
