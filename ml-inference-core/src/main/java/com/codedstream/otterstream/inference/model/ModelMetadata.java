package com.codedstream.otterstream.inference.model;

import java.util.Map;
import java.util.HashMap;
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
        this.modelName = Objects.requireNonNull(modelName, "modelName cannot be null");
        this.modelVersion = Objects.requireNonNull(modelVersion, "modelVersion cannot be null");
        this.format = Objects.requireNonNull(format, "format cannot be null");
        this.inputSchema = inputSchema != null ? Map.copyOf(inputSchema) : Map.of();
        this.outputSchema = outputSchema != null ? Map.copyOf(outputSchema) : Map.of();
        this.modelSize = modelSize;
        this.loadTimestamp = loadTimestamp;
    }

    // Getters
    public String getModelName() { return modelName; }
    public String getModelVersion() { return modelVersion; }
    public ModelFormat getFormat() { return format; }
    public Map<String, Object> getInputSchema() { return inputSchema; }
    public Map<String, Object> getOutputSchema() { return outputSchema; }
    public long getModelSize() { return modelSize; }
    public long getLoadTimestamp() { return loadTimestamp; }

    // Builder pattern
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelName;
        private String modelVersion = "unknown";
        private ModelFormat format;
        private Map<String, Object> inputSchema = new HashMap<>();
        private Map<String, Object> outputSchema = new HashMap<>();
        private long modelSize = 0L;
        private long loadTimestamp = System.currentTimeMillis();

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder modelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public Builder inputSchema(Map<String, Object> inputSchema) {
            this.inputSchema = inputSchema != null ? new HashMap<>(inputSchema) : new HashMap<>();
            return this;
        }

        public Builder outputSchema(Map<String, Object> outputSchema) {
            this.outputSchema = outputSchema != null ? new HashMap<>(outputSchema) : new HashMap<>();
            return this;
        }

        public Builder modelSize(long modelSize) {
            this.modelSize = modelSize;
            return this;
        }

        public Builder loadTimestamp(long loadTimestamp) {
            this.loadTimestamp = loadTimestamp;
            return this;
        }

        public ModelMetadata build() {
            return new ModelMetadata(
                    modelName,
                    modelVersion,
                    format,
                    inputSchema,
                    outputSchema,
                    modelSize,
                    loadTimestamp
            );
        }
    }

    @Override
    public String toString() {
        return "ModelMetadata{" +
                "modelName='" + modelName + '\'' +
                ", modelVersion='" + modelVersion + '\'' +
                ", format=" + format +
                ", inputSchema=" + inputSchema +
                ", outputSchema=" + outputSchema +
                ", modelSize=" + modelSize +
                ", loadTimestamp=" + loadTimestamp +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ModelMetadata that = (ModelMetadata) o;
        return modelSize == that.modelSize &&
                loadTimestamp == that.loadTimestamp &&
                Objects.equals(modelName, that.modelName) &&
                Objects.equals(modelVersion, that.modelVersion) &&
                format == that.format &&
                Objects.equals(inputSchema, that.inputSchema) &&
                Objects.equals(outputSchema, that.outputSchema);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelName, modelVersion, format, inputSchema,
                outputSchema, modelSize, loadTimestamp);
    }
}