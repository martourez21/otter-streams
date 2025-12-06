package com.codedstream.otterstream.inference.model;

import java.util.Map;
import java.util.Objects;

public class InferenceResult {
    private final Map<String, Object> outputs;
    private final long inferenceTimeMs;
    private final String modelId;
    private final boolean success;
    private final String errorMessage;

    public InferenceResult(Map<String, Object> outputs, long inferenceTimeMs, String modelId) {
        this.outputs = Map.copyOf(Objects.requireNonNull(outputs));
        this.inferenceTimeMs = inferenceTimeMs;
        this.modelId = Objects.requireNonNull(modelId);
        this.success = true;
        this.errorMessage = null;
    }

    public InferenceResult(String modelId, String errorMessage, long inferenceTimeMs) {
        this.outputs = Map.of();
        this.inferenceTimeMs = inferenceTimeMs;
        this.modelId = Objects.requireNonNull(modelId);
        this.success = false;
        this.errorMessage = errorMessage;
    }

    public Map<String, Object> getOutputs() { return outputs; }
    public long getInferenceTimeMs() { return inferenceTimeMs; }
    public String getModelId() { return modelId; }
    public boolean isSuccess() { return success; }
    public String getErrorMessage() { return errorMessage; }

    @SuppressWarnings("unchecked")
    public <T> T getOutput(String key) {
        return (T) outputs.get(key);
    }
}
