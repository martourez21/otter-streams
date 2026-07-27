package com.codedstream.otterstream.benchmarks.support;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import java.util.Map;

/**
 * An {@link InferenceEngine} that does no actual inference — returns a fixed result
 * immediately. Used only in benchmarks to isolate Otter's own routing/tracking/lifecycle
 * overhead from real model computation time, which obviously varies enormously by model and
 * isn't something a generic benchmark suite can meaningfully measure on your behalf. See
 * {@code README.md}'s "What these benchmarks measure" section.
 *
 * @since 0.1.0 (benchmark support code only — not part of the public API)
 */
public class NoOpInferenceEngine implements InferenceEngine<Void> {

    private volatile boolean ready = false;
    private volatile ModelConfig modelConfig;

    @Override
    public void initialize(ModelConfig config) {
        this.modelConfig = config;
        this.ready = true;
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        if (!ready) {
            throw new InferenceException("NoOpInferenceEngine not initialized");
        }
        return new InferenceResult(Map.of("result", 0.42), 0, "noop-model");
    }

    @Override
    @SuppressWarnings("unchecked")
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        return infer(batchInputs.length > 0 ? batchInputs[0] : Map.of());
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, false, 0, false);
    }

    @Override
    public void close() {
        ready = false;
    }

    @Override
    public boolean isReady() {
        return ready;
    }

    @Override
    public ModelMetadata getMetadata() {
        return ModelMetadata.builder()
                .modelName(modelConfig != null ? modelConfig.getModelId() : "noop-model")
                .format(modelConfig != null ? modelConfig.getFormat() : com.codedstream.otterstream.inference.model.ModelFormat.ONNX)
                .build();
    }

    @Override
    public ModelConfig getModelConfig() {
        return modelConfig;
    }
}
