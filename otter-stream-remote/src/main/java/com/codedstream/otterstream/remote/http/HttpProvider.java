package com.codedstream.otterstream.remote.http;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.Set;

/**
 * {@link InferenceProvider} for generic remote HTTP/REST inference endpoints, discovered
 * automatically by {@code OtterRuntime} via {@link java.util.ServiceLoader} when this module is
 * on the classpath.
 *
 * @since 0.1.0
 */
public class HttpProvider implements InferenceProvider {

    @Override
    public String getProviderId() {
        return "remote-http";
    }

    @Override
    public Set<ModelFormat> getSupportedFormats() {
        return Set.of(ModelFormat.REMOTE_HTTP);
    }

    @Override
    public InferenceEngine<?> createEngine() {
        return new HttpInferenceClient();
    }
}
