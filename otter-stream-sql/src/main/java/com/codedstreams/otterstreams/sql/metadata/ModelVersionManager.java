package com.codedstreams.otterstreams.sql.metadata;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages multiple versions of a model.
 */
public class ModelVersionManager {
    private final String modelName;
    private final Map<String, ModelDescriptor> versions;
    private String latestVersion;

    public ModelVersionManager(String modelName) {
        this.modelName = modelName;
        this.versions = new ConcurrentHashMap<>();
    }

    public void addVersion(String version, ModelDescriptor descriptor) {
        versions.put(version, descriptor);
        this.latestVersion = version;
    }

    public ModelDescriptor getVersion(String version) {
        if ("latest".equals(version)) {
            return versions.get(latestVersion);
        }
        return versions.get(version);
    }

    public ModelDescriptor getLatest() {
        return versions.get(latestVersion);
    }

    public String getModelName() {
        return modelName;
    }
}
