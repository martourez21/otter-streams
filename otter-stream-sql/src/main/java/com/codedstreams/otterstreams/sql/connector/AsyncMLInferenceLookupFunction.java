package com.codedstreams.otterstreams.sql.connector;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.functions.AsyncLookupFunction;
import org.apache.flink.table.functions.FunctionContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Async lookup function for ML inference.
 * Uses modern Flink 2.0 AsyncLookupFunction API.
 */
public class AsyncMLInferenceLookupFunction extends AsyncLookupFunction {

    private static final Logger LOG = LoggerFactory.getLogger(AsyncMLInferenceLookupFunction.class);

    private final SqlInferenceConfig config;
    private transient ModelCache modelCache;

    public AsyncMLInferenceLookupFunction(SqlInferenceConfig config) {
        this.config = config;
    }

    @Override
    public void open(FunctionContext context) throws Exception {
        this.modelCache = ModelCache.getInstance();
        LOG.info("AsyncMLInferenceLookupFunction initialized for model: {}", config.getModelName());
    }

    @Override
    public CompletableFuture<Collection<RowData>> asyncLookup(RowData keyRow) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                // Get engine from cache
                InferenceEngine<?> engine = modelCache.getEngine(config.getModelName());
                if (engine == null) {
                    LOG.warn("Model engine not found in cache for: {}", config.getModelName());
                    return Collections.emptyList();
                }

                // Extract features from RowData
                Map<String, Object> features = extractFeatures(keyRow);

                // Perform inference
                InferenceResult result = engine.infer(features);

                if (result.isSuccess()) {
                    GenericRowData row = new GenericRowData(2);
                    Object prediction = result.getOutputs().values().iterator().next();
                    row.setField(0, ((Number) prediction).doubleValue());
                    row.setField(1, 1.0); // confidence
                    return Collections.singletonList(row);
                } else {
                    LOG.warn("Inference failed for model: {}, error: {}",
                            config.getModelName(), result.getErrorMessage());
                    return Collections.emptyList();
                }
            } catch (Exception e) {
                LOG.error("Error during async inference for model: {}", config.getModelName(), e);
                return Collections.emptyList();
            }
        });
    }

    /**
     * Extract features from RowData.
     */
    protected Map<String, Object> extractFeatures(RowData keyRow) {
        Map<String, Object> features = new HashMap<>();
        int arity = keyRow.getArity();

        for (int i = 0; i < arity; i++) {
            if (keyRow.isNullAt(i)) {
                features.put("feature_" + i, null);
            } else {
                try {
                    Object value = getFieldValue(keyRow, i);
                    features.put("feature_" + i, value);
                } catch (Exception e) {
                    LOG.warn("Failed to extract field {} from RowData", i, e);
                    features.put("feature_" + i, null);
                }
            }
        }
        return features;
    }

    /**
     * Get field value from RowData based on type.
     */
    private Object getFieldValue(RowData row, int pos) {
        if (row.isNullAt(pos)) {
            return null;
        }
        try {
            return row.getString(pos).toString();
        } catch (Exception e1) {
            try {
                return row.getInt(pos);
            } catch (Exception e2) {
                try {
                    return row.getDouble(pos);
                } catch (Exception e3) {
                    try {
                        return row.getBoolean(pos);
                    } catch (Exception e4) {
                        return row.getRawValue(pos);
                    }
                }
            }
        }
    }
}