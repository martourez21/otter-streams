package com.codedstream.otterstream.inference.engine;


import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;

import java.util.Map;

public interface InferenceEngine<T> {

    void initialize(ModelConfig config) throws InferenceException;

    InferenceResult infer(Map<String, Object> inputs) throws InferenceException;

    InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException;

    EngineCapabilities getCapabilities();

    void close() throws InferenceException;

    boolean isReady();

    ModelConfig getModelConfig();

    class EngineCapabilities {
        private final boolean supportsBatching;
        private final boolean supportsGPU;
        private final int maxBatchSize;
        private final boolean supportsStreaming;

        public EngineCapabilities(boolean supportsBatching, boolean supportsGPU,
                                  int maxBatchSize, boolean supportsStreaming) {
            this.supportsBatching = supportsBatching;
            this.supportsGPU = supportsGPU;
            this.maxBatchSize = maxBatchSize;
            this.supportsStreaming = supportsStreaming;
        }

        public boolean supportsBatching() { return supportsBatching; }
        public boolean supportsGPU() { return supportsGPU; }
        public int getMaxBatchSize() { return maxBatchSize; }
        public boolean supportsStreaming() { return supportsStreaming; }
    }
}