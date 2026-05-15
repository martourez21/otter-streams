package com.codedstreams.otterstreams.sql.udf;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import com.codedstreams.otterstreams.sql.util.JsonFeatureExtractor;
import org.apache.flink.table.functions.ScalarFunction;

import java.util.Map;

/**
 * Flink SQL scalar function for ML inference.
 *
 * Usage:
 * SELECT OTTER_XGBOOST_PREDICT(
 *   '{"amt":226.32,"distance":123.0}',
 *   'fraud-detection-xgboost'
 * );
 */
public class MLInferenceFunction extends ScalarFunction {

    private transient ModelCache modelCache;

    public Double eval(String featuresJson, String modelName) throws Exception {

        if (modelCache == null) {
            modelCache = ModelCache.getInstance();
        }

        InferenceEngine<?> engine = modelCache.getEngine(modelName);

        if (engine == null) {
            return null; // model not loaded
        }

        Map<String, Object> features =
                JsonFeatureExtractor.extractFeatures(featuresJson);

        InferenceResult result = engine.infer(features);

        if (result == null || !result.isSuccess() || result.getOutputs().isEmpty()) {
            return null;
        }

        Object output = result.getOutputs().values().iterator().next();

        return (output instanceof Number)
                ? ((Number) output).doubleValue()
                : null;
    }
}