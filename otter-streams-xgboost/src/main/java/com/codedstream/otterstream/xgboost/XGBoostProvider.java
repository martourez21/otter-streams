package com.codedstream.otterstream.xgboost;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.Set;

/**
 * {@link InferenceProvider} for XGBoost models (binary and JSON booster formats), discovered
 * automatically by {@code OtterRuntime} via {@link java.util.ServiceLoader} when this module is
 * on the classpath.
 *
 * @since 0.1.0
 */
public class XGBoostProvider implements InferenceProvider {

    @Override
    public String getProviderId() {
        return "xgboost";
    }

    @Override
    public Set<ModelFormat> getSupportedFormats() {
        return Set.of(ModelFormat.XGBOOST_BINARY, ModelFormat.XGBOOST_JSON);
    }

    @Override
    public InferenceEngine<?> createEngine() {
        return new XGBoostInferenceEngine();
    }
}
