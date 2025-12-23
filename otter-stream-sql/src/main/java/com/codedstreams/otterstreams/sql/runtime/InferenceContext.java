package com.codedstreams.otterstreams.sql.runtime;

import java.util.HashMap;
import java.util.Map;

/**
 * Context for inference execution with metadata.
 */
public class InferenceContext {
    private final Map<String, Object> metadata;
    private final long requestTime;

    public InferenceContext() {
        this.metadata = new HashMap<>();
        this.requestTime = System.currentTimeMillis();
    }

    public void put(String key, Object value) {
        metadata.put(key, value);
    }

    public Object get(String key) {
        return metadata.get(key);
    }

    public long getRequestTime() {
        return requestTime;
    }

    public long getElapsedTime() {
        return System.currentTimeMillis() - requestTime;
    }
}