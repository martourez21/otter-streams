package com.codedstream.otterstream.inference.config;

import java.util.Map;
import java.util.Objects;

/**
 * Configuration for authentication with remote ML inference endpoints.
 *
 * <p>Supports various authentication methods including API keys, bearer tokens,
 * and custom headers. Use this when connecting to remote inference services
 * like AWS SageMaker, Google Vertex AI, or custom REST APIs.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // API Key authentication
 * AuthConfig auth = AuthConfig.builder()
 *     .apiKey("your-api-key-here")
 *     .build();
 *
 * // Bearer token authentication
 * AuthConfig auth = AuthConfig.builder()
 *     .token("Bearer eyJhbGc...")
 *     .build();
 *
 * // Custom headers
 * AuthConfig auth = AuthConfig.builder()
 *     .headers(Map.of(
 *         "X-API-Key", "key123",
 *         "X-Client-ID", "client456"
 *     ))
 *     .build();
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class AuthConfig {
    private final String apiKey;
    private final String token;
    private final Map<String, String> headers;

    /**
     * Constructs authentication configuration.
     *
     * @param apiKey API key for authentication (can be null)
     * @param token bearer token for authentication (can be null)
     * @param headers custom HTTP headers for authentication
     */
    public AuthConfig(String apiKey, String token, Map<String, String> headers) {
        this.apiKey = apiKey;
        this.token = token;
        this.headers = Map.copyOf(headers);
    }

    /**
     * Gets the API key.
     *
     * @return API key, or null if not configured
     */
    public String getApiKey() { return apiKey; }

    /**
     * Gets the bearer token.
     *
     * @return bearer token, or null if not configured
     */
    public String getToken() { return token; }

    /**
     * Gets custom authentication headers.
     *
     * @return immutable map of header name-value pairs
     */
    public Map<String, String> getHeaders() { return headers; }

    /**
     * Creates a new builder for AuthConfig.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for creating AuthConfig instances.
     */
    public static class Builder {
        private String apiKey;
        private String token;
        private Map<String, String> headers = Map.of();

        /**
         * Sets the API key.
         *
         * @param apiKey the API key
         * @return this builder
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * Sets the bearer token.
         *
         * @param token the bearer token
         * @return this builder
         */
        public Builder token(String token) {
            this.token = token;
            return this;
        }

        /**
         * Sets custom authentication headers.
         *
         * @param headers map of header name-value pairs
         * @return this builder
         */
        public Builder headers(Map<String, String> headers) {
            this.headers = Map.copyOf(headers);
            return this;
        }

        /**
         * Builds the AuthConfig instance.
         *
         * @return configured AuthConfig
         */
        public AuthConfig build() {
            return new AuthConfig(apiKey, token, headers);
        }
    }
}