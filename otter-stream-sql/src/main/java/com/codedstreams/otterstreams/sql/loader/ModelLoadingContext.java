package com.codedstreams.otterstreams.sql.loader;

import com.codedstreams.otterstreams.sql.config.ModelSourceConfig;

import java.util.HashMap;
import java.util.Map;

/**
 * Context information for model loading operations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ModelLoadingContext {
    private final String modelName;
    private final ModelSourceConfig sourceConfig;
    private final Map<String, Object> metadata;
    private final long startTime;

    public ModelLoadingContext(String modelName, ModelSourceConfig sourceConfig) {
        this.modelName = modelName;
        this.sourceConfig = sourceConfig;
        this.metadata = new HashMap<>();
        this.startTime = System.currentTimeMillis();
    }

    public String getModelName() {
        return modelName;
    }

    public ModelSourceConfig getSourceConfig() {
        return sourceConfig;
    }

    public void putMetadata(String key, Object value) {
        metadata.put(key, value);
    }

    public Object getMetadata(String key) {
        return metadata.get(key);
    }

    public long getElapsedTime() {
        return System.currentTimeMillis() - startTime;
    }
}
