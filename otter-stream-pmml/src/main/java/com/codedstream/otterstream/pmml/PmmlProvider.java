package com.codedstream.otterstream.pmml;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.Set;

/**
 * {@link InferenceProvider} for PMML models, discovered automatically by
 * {@code OtterRuntime} via {@link java.util.ServiceLoader} when this module is on the classpath.
 *
 * @since 0.1.0
 */
public class PmmlProvider implements InferenceProvider {

    @Override
    public String getProviderId() {
        return "pmml";
    }

    @Override
    public Set<ModelFormat> getSupportedFormats() {
        return Set.of(ModelFormat.PMML);
    }

    @Override
    public InferenceEngine<?> createEngine() {
        return new PmmlInferenceEngine();
    }
}
