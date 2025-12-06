package com.codedstream.otterstream.inference.config;

import java.util.Map;
import java.util.Objects;

public class AuthConfig {
    private final String apiKey;
    private final String token;
    private final Map<String, String> headers;

    public AuthConfig(String apiKey, String token, Map<String, String> headers) {
        this.apiKey = apiKey;
        this.token = token;
        this.headers = Map.copyOf(headers);
    }

    public String getApiKey() { return apiKey; }
    public String getToken() { return token; }
    public Map<String, String> getHeaders() { return headers; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String apiKey;
        private String token;
        private Map<String, String> headers = Map.of();

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder token(String token) {
            this.token = token;
            return this;
        }

        public Builder headers(Map<String, String> headers) {
            this.headers = Map.copyOf(headers);
            return this;
        }

        public AuthConfig build() {
            return new AuthConfig(apiKey, token, headers);
        }
    }
}
