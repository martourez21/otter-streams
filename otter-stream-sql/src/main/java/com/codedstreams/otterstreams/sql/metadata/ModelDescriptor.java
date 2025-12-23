package com.codedstreams.otterstreams.sql.metadata;

import com.codedstream.otterstream.inference.model.ModelFormat;

import java.io.Serializable;
import java.util.Map;

/**
 * Metadata descriptor for a registered model.
 */
public class ModelDescriptor implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String modelId;
    private final String modelName;
    private final String version;
    private final ModelFormat format;
    private final String path;
    private final InputOutputSchema schema;
    private final Map<String, String> tags;
    private final long registrationTime;

    public ModelDescriptor(String modelId, String modelName, String version,
                           ModelFormat format, String path, InputOutputSchema schema,
                           Map<String, String> tags) {
        this.modelId = modelId;
        this.modelName = modelName;
        this.version = version;
        this.format = format;
        this.path = path;
        this.schema = schema;
        this.tags = Map.copyOf(tags);
        this.registrationTime = System.currentTimeMillis();
    }

    public String getModelId() { return modelId; }
    public String getModelName() { return modelName; }
    public String getVersion() { return version; }
    public ModelFormat getFormat() { return format; }
    public String getPath() { return path; }
    public InputOutputSchema getSchema() { return schema; }
    public Map<String, String> getTags() { return tags; }
    public long getRegistrationTime() { return registrationTime; }
}
