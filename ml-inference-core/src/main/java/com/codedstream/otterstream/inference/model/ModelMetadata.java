package com.codedstream.otterstream.inference.model;

import java.util.Map;
import java.util.Objects;

public class ModelMetadata {
    private final String modelName;
    private final String modelVersion;
    private final ModelFormat format;
    private final Map<String, Object> inputSchema;
    private final Map<String, Object> outputSchema;
    private final long modelSize;
    private final long loadTimestamp;

    public ModelMetadata(String modelName, String modelVersion, ModelFormat format,
                         Map<String, Object> inputSchema, Map<String, Object> outputSchema,
                         long modelSize, long loadTimestamp) {
        this.modelName = Objects.requireNonNull(modelName);
        this.modelVersion = Objects.requireNonNull(modelVersion);
        this.format = Objects.requireNonNull(format);
        this.inputSchema = Map.copyOf(inputSchema);
        this.outputSchema = Map.copyOf(outputSchema);
        this.modelSize = modelSize;
        this.loadTimestamp = loadTimestamp;
    }

    public String getModelName() { return modelName; }
    public String getModelVersion() { return modelVersion; }
    public ModelFormat getFormat() { return format; }
    public Map<String, Object> getInputSchema() { return inputSchema; }
    public Map<String, Object> getOutputSchema() { return outputSchema; }
    public long getModelSize() { return modelSize; }
    public long getLoadTimestamp() { return loadTimestamp; }
}