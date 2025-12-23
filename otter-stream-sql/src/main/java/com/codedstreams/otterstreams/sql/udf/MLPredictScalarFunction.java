package com.codedstreams.otterstreams.sql.udf;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import com.codedstreams.otterstreams.sql.util.JsonFeatureExtractor;
import org.apache.flink.table.functions.ScalarFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Flink SQL Scalar Function for ML model inference.
 *
 * Usage: SELECT ML_PREDICT('model-name', '{"feature1": 1.0}') FROM table
 */
public class MLPredictScalarFunction extends ScalarFunction {
    private static final Logger LOG = LoggerFactory.getLogger(MLPredictScalarFunction.class);
    private static final long serialVersionUID = 1L;

    private transient ModelCache modelCache;

    public Double eval(String modelName, String featuresJson) {
        try {
            if (modelCache == null) {
                modelCache = ModelCache.getInstance();
            }

            // Get or load engine
            InferenceEngine<?> engine = modelCache.getEngine(modelName);
            if (engine == null) {
                LOG.warn("Model not found in cache: {}", modelName);
                return null;
            }

            // Parse features
            Map<String, Object> features = JsonFeatureExtractor.extractFeatures(featuresJson);

            // Perform inference
            InferenceResult result = engine.infer(features);

            if (result.isSuccess()) {
                Object prediction = result.getOutputs().values().iterator().next();
                return ((Number) prediction).doubleValue();
            }

            return null;
        } catch (Exception e) {
            LOG.error("Inference failed for model: {}", modelName, e);
            return null;
        }
    }

    @Override
    public void close() {
        // Cleanup handled by ModelCache singleton
    }
}
