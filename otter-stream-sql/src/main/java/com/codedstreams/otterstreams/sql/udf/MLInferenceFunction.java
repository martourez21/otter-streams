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
 * This function performs online inference against a preloaded/cached model
 * using JSON feature payloads passed from Flink SQL.
 *
 * Usage:
 *
 * SELECT OTTER_XGBOOST_PREDICT(
 *   JSON_OBJECT(
 *     'amt' VALUE 226.32,
 *     'distance' VALUE 123.0,
 *     'hour' VALUE 14
 *   ),
 *   'fraud-detection-xgboost'
 * );
 *
 * Notes:
 * - Models are loaded and cached internally using ModelCache.
 * - Features are passed as JSON strings for Flink SQL compatibility.
 * - The function returns the first numeric prediction output.
 */
public class MLInferenceFunction extends ScalarFunction {

    /**
     * Shared model cache instance.
     *
     * Lazily initialized to avoid serialization/runtime issues
     * inside distributed Flink task managers.
     */
    private transient ModelCache modelCache;

    /**
     * Executes ML inference using the provided feature payload.
     *
     * @param featuresJson JSON feature payload from Flink SQL
     * @param modelName registered/cached model name
     * @return prediction score/probability or null if inference fails
     * @throws InferenceException inference engine exception
     */
    public Double eval(String featuresJson, String modelName)
            throws Exception {

        // Lazily initialize cache
        if (modelCache == null) {
            modelCache = ModelCache.getInstance();
        }

        // Retrieve cached inference engine/model
        InferenceEngine<?> engine = modelCache.getEngine(modelName);

        if (engine == null) {
            return null;
        }

        // Convert JSON payload into feature map
        Map<String, Object> features =
                JsonFeatureExtractor.extractFeatures(featuresJson);

        // Execute inference
        InferenceResult result = engine.infer(features);

        // Validate inference result
        if (!result.isSuccess() || result.getOutputs().isEmpty()) {
            return null;
        }

        // Return first numeric output
        Object output = result.getOutputs()
                .values()
                .iterator()
                .next();

        return output instanceof Number
                ? ((Number) output).doubleValue()
                : null;
    }
}