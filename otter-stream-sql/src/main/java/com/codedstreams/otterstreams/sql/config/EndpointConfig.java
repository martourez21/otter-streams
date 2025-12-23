package com.codedstreams.otterstreams.sql.config;

import java.io.Serializable;
import java.util.Map;
import java.util.HashMap;

public class EndpointConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String url;
    private final String authToken;
    private final Map<String, String> headers;
    private final long connectionTimeoutMs;
    private final long readTimeoutMs;

    private EndpointConfig(Builder builder) {
        this.url = builder.url;
        this.authToken = builder.authToken;
        this.headers = Map.copyOf(builder.headers);
        this.connectionTimeoutMs = builder.connectionTimeoutMs;
        this.readTimeoutMs = builder.readTimeoutMs;
    }

    public String getUrl() { return url; }
    public String getAuthToken() { return authToken; }
    public Map<String, String> getHeaders() { return headers; }
    public long getConnectionTimeoutMs() { return connectionTimeoutMs; }
    public long getReadTimeoutMs() { return readTimeoutMs; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String url;
        private String authToken;
        private Map<String, String> headers = new HashMap<>();
        private long connectionTimeoutMs = 30000;
        private long readTimeoutMs = 60000;

        public Builder url(String url) { this.url = url; return this; }
        public Builder authToken(String token) { this.authToken = token; return this; }
        public Builder headers(Map<String, String> headers) { this.headers = new HashMap<>(headers); return this; }
        public Builder connectionTimeoutMs(long ms) { this.connectionTimeoutMs = ms; return this; }
        public Builder readTimeoutMs(long ms) { this.readTimeoutMs = ms; return this; }
        public EndpointConfig build() { return new EndpointConfig(this); }
    }
}
