package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility for warming up models to avoid cold start latency.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ModelWarmupUtility {
    private static final Logger LOG = LoggerFactory.getLogger(ModelWarmupUtility.class);

    /**
     * Warms up a model by running a dummy inference.
     */
    public static void warmupModel(String modelName) {
        try {
            ModelCache cache = ModelCache.getInstance();
            InferenceEngine<?> engine = cache.getEngine(modelName);

            if (engine == null) {
                LOG.warn("Cannot warmup model {}: not found in cache", modelName);
                return;
            }

            LOG.info("Warming up model: {}", modelName);

            // Create dummy input
            Map<String, Object> dummyInput = createDummyInput();

            // Run inference
            long startTime = System.currentTimeMillis();
            engine.infer(dummyInput);
            long endTime = System.currentTimeMillis();

            LOG.info("Model {} warmed up in {}ms", modelName, endTime - startTime);
        } catch (Exception e) {
            LOG.warn("Failed to warmup model: {}", modelName, e);
        }
    }

    /**
     * Warms up multiple models.
     */
    public static void warmupModels(String... modelNames) {
        for (String modelName : modelNames) {
            warmupModel(modelName);
        }
    }

    private static Map<String, Object> createDummyInput() {
        Map<String, Object> input = new HashMap<>();
        input.put("dummy_feature", 1.0);
        return input;
    }
}
