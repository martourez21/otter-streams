package com.codedstreams.otterstreams.sql.udf;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.table.functions.ScalarFunction;

import java.util.Map;

/**
 * Flink SQL scalar function for ML inference.
 *
 * Usage:
 *   SELECT ml_score(features, 'fraud-model') FROM transactions;
 */
public class MLInferenceFunction extends ScalarFunction {

    private transient ModelCache modelCache;

    public Double eval(Map<String, Object> features, String modelName) throws InferenceException {
        if (modelCache == null) {
            modelCache = ModelCache.getInstance();
        }

        InferenceEngine<?> engine = modelCache.getEngine(modelName);
        if (engine == null) {
            return null;
        }

        InferenceResult result = engine.infer(features);
        if (!result.isSuccess() || result.getOutputs().isEmpty()) {
            return null;
        }

        Object output = result.getOutputs().values().iterator().next();
        return output instanceof Number ? ((Number) output).doubleValue() : null;
    }
}
