package com.codedstreams.otterstreams.sql.runtime;

import java.io.Serializable;

/**
 * Manages stateful inference sessions.
 */
public class ModelSession implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String sessionId;
    private final String modelId;
    private long lastAccessTime;

    public ModelSession(String sessionId, String modelId) {
        this.sessionId = sessionId;
        this.modelId = modelId;
        this.lastAccessTime = System.currentTimeMillis();
    }

    public String getSessionId() { return sessionId; }
    public String getModelId() { return modelId; }
    public long getLastAccessTime() { return lastAccessTime; }

    public void updateAccessTime() {
        this.lastAccessTime = System.currentTimeMillis();
    }
}
