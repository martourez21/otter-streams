package com.codedstream.otterstream.inference.engine;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelLoader;

import java.util.Map;

public abstract class LocalInferenceEngine<T> implements InferenceEngine<T> {
    protected ModelConfig modelConfig;
    protected T loadedModel;
    protected ModelLoader<T> modelLoader;
    protected boolean initialized = false;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        this.modelConfig = config;
        try {
            if (modelLoader != null) {
                this.loadedModel = modelLoader.loadModel(config);
            } else {
                // For engines that handle their own loading
                loadModelDirectly(config);
            }
            this.initialized = true;
        } catch (Exception e) {
            throw new InferenceException("Failed to initialize inference engine", e);
        }
    }

    /**
     * Override this method for engines that handle their own model loading
     */
    protected void loadModelDirectly(ModelConfig config) throws InferenceException {
        throw new InferenceException("Model loader not provided and loadModelDirectly not implemented");
    }

    @Override
    public abstract InferenceResult infer(Map<String, Object> inputs) throws InferenceException;

    @Override
    public abstract InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException;

    @Override
    public boolean isReady() {
        return initialized && loadedModel != null;
    }

    @Override
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    @Override
    public void close() throws InferenceException {
        this.initialized = false;
        this.loadedModel = null;
    }

    @Override
    public abstract EngineCapabilities getCapabilities();
}