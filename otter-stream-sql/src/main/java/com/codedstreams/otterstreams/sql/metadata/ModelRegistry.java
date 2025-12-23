package com.codedstreams.otterstreams.sql.metadata;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for model metadata and lifecycle management.
 */
public class ModelRegistry {
    private static final ModelRegistry INSTANCE = new ModelRegistry();

    private final Map<String, ModelDescriptor> models;
    private final Map<String, ModelVersionManager> versionManagers;

    private ModelRegistry() {
        this.models = new ConcurrentHashMap<>();
        this.versionManagers = new ConcurrentHashMap<>();
    }

    public static ModelRegistry getInstance() {
        return INSTANCE;
    }

    public void registerModel(String name, ModelDescriptor descriptor) {
        models.put(name, descriptor);
    }

    public ModelDescriptor getModel(String name) {
        return models.get(name);
    }

    public ModelDescriptor getModel(String name, String version) {
        ModelVersionManager versionManager = versionManagers.get(name);
        if (versionManager != null) {
            return versionManager.getVersion(version);
        }
        return models.get(name);
    }

    public void unregisterModel(String name) {
        models.remove(name);
        versionManagers.remove(name);
    }
}

