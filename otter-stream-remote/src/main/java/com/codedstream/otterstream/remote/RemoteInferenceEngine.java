package com.codedstream.otterstream.remote;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;

import java.util.Map;

public abstract class RemoteInferenceEngine implements InferenceEngine<Void> {
    protected ModelConfig modelConfig;
    protected boolean initialized = false;
    protected String endpointUrl;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        this.modelConfig = config;
        this.endpointUrl = config.getEndpointUrl();
        this.initialized = true;
    }

    @Override
    public abstract InferenceResult infer(Map<String, Object> inputs) throws InferenceException;

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        // Default implementation - process sequentially
        // Override for batch-optimized remote endpoints
        Map<String, Object> batchOutputs = new java.util.HashMap<>();
        long totalTime = 0;

        for (int i = 0; i < batchInputs.length; i++) {
            InferenceResult result = infer(batchInputs[i]);
            totalTime += result.getInferenceTimeMs();

            // Store results with index
            for (Map.Entry<String, Object> entry : result.getOutputs().entrySet()) {
                String batchKey = entry.getKey() + "_" + i;
                batchOutputs.put(batchKey, entry.getValue());
            }
        }

        return new InferenceResult(batchOutputs, totalTime, modelConfig.getModelId());
    }

//    @Override
//    public Void getModelConfig() {
//        return null; // Remote engines don't have loaded models
//    }

    @Override
    public boolean isReady() {
        return initialized && endpointUrl != null;
    }

//    @Override
//    public ModelConfig getModelConfig() {
//        return modelConfig;
//    }

    @Override
    public void close() throws InferenceException {
        this.initialized = false;
        this.endpointUrl = null;
    }

    /**
     * Validate connection to remote endpoint
     */
    public abstract boolean validateConnection() throws InferenceException;

    /**
     * Get engine capabilities for remote endpoints
     */
    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(false, false, 1, true);
    }
}
