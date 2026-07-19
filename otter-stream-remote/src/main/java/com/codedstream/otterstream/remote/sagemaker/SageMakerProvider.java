package com.codedstream.otterstream.remote.sagemaker;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.Set;

/**
 * {@link InferenceProvider} for AWS SageMaker endpoints, discovered automatically by
 * {@code OtterRuntime} via {@link java.util.ServiceLoader} when this module is on the classpath.
 *
 * @since 0.1.0
 */
public class SageMakerProvider implements InferenceProvider {

    @Override
    public String getProviderId() {
        return "sagemaker";
    }

    @Override
    public Set<ModelFormat> getSupportedFormats() {
        return Set.of(ModelFormat.SAGEMAKER);
    }

    @Override
    public InferenceEngine<?> createEngine() {
        return new SageMakerInferenceClient();
    }
}
