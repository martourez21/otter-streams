package com.codedstream.otterstream.onnx;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.Set;

/**
 * {@link InferenceProvider} for ONNX models, discovered automatically by
 * {@code OtterRuntime} via {@link java.util.ServiceLoader} when this module is on the classpath.
 *
 * <p>Purely additive: {@link OnnxInferenceEngine} and {@link OnnxModelLoader} are unchanged and
 * remain fully usable directly, without going through {@code OtterRuntime}.
 *
 * @since 0.1.0
 */
public class OnnxProvider implements InferenceProvider {

    @Override
    public String getProviderId() {
        return "onnx";
    }

    @Override
    public Set<ModelFormat> getSupportedFormats() {
        return Set.of(ModelFormat.ONNX);
    }

    @Override
    public InferenceEngine<?> createEngine() {
        return new OnnxInferenceEngine();
    }
}
