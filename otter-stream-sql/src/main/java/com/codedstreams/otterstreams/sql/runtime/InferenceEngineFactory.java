package com.codedstreams.otterstreams.sql.runtime;


import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.ModelFormat;

/**
 * Factory for creating inference engines based on model format.
 */
public class InferenceEngineFactory {

    public static InferenceEngine<?> createEngine(ModelConfig config) throws InferenceException {
        ModelFormat format = config.getFormat();

        switch (format) {
            case TENSORFLOW_SAVEDMODEL:
                return new TensorFlowSavedModelEngine();
            case TENSORFLOW_GRAPHDEF:
                return new TensorFlowGraphDefEngine();
            default:
                throw new InferenceException("Unsupported model format: " + format);
        }
    }
}
