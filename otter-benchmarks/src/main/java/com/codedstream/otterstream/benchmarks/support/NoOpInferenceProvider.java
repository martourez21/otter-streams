package com.codedstream.otterstream.benchmarks.support;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.Set;

/**
 * Registers {@link NoOpInferenceEngine} under {@link ModelFormat#ONNX} — chosen arbitrarily,
 * just needs to be some real enum value {@code LifecycleManager.deploy} can resolve. This
 * provider is constructed and registered locally within each benchmark's own
 * {@code ProviderRegistry} instance — never registered globally, so it can't shadow a real ONNX
 * provider if this jar somehow ended up on the same classpath as one (it won't, in normal use;
 * this module isn't meant to be a runtime dependency of anything).
 *
 * @since 0.1.0 (benchmark support code only — not part of the public API)
 */
public class NoOpInferenceProvider implements InferenceProvider {

    @Override
    public String getProviderId() {
        return "noop-benchmark";
    }

    @Override
    public Set<ModelFormat> getSupportedFormats() {
        return Set.of(ModelFormat.ONNX);
    }

    @Override
    public InferenceEngine<?> createEngine() {
        return new NoOpInferenceEngine();
    }
}
