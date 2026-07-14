package com.codedstreams.otterstreams.sql.connector;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.LookupFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Synchronous lookup function for ML inference.
 * Uses modern Flink 2.0 LookupFunction API.
 */
public class MLInferenceLookupFunction extends LookupFunction {

    private static final Logger LOG = LoggerFactory.getLogger(MLInferenceLookupFunction.class);

    private final SqlInferenceConfig config;
    private transient ModelCache modelCache;

    public MLInferenceLookupFunction(SqlInferenceConfig config) {
        this.config = config;
    }

    @Override
    public void open(FunctionContext context) throws Exception {
        this.modelCache = ModelCache.getInstance();
        LOG.info("MLInferenceLookupFunction initialized for model: {}", config.getModelName());
    }

    @Override
    public Collection<RowData> lookup(RowData keyRow) throws IOException {
        try {
            InferenceEngine<?> engine = modelCache.getEngine(config.getModelName());
            if (engine == null) {
                LOG.warn("Model engine not found in cache for: {}", config.getModelName());
                return Collections.emptyList();
            }

            Map<String, Object> features = extractFeatures(keyRow);
            InferenceResult result = engine.infer(features);

            if (result.isSuccess()) {
                GenericRowData row = new GenericRowData(2);
                Object prediction = result.getOutputs().values().iterator().next();
                row.setField(0, ((Number) prediction).doubleValue());
                row.setField(1, 1.0);
                return Collections.singletonList(row);
            } else {
                LOG.warn("Inference failed for model: {}, error: {}",
                        config.getModelName(), result.getErrorMessage());
                return Collections.emptyList();
            }
        } catch (Exception e) {
            LOG.error("Error during inference for model: {}", config.getModelName(), e);
            throw new IOException("Failed to perform inference", e);
        }
    }

    protected Map<String, Object> extractFeatures(RowData keyRow) {
        Map<String, Object> features = new HashMap<>();
        int arity = keyRow.getArity();

        for (int i = 0; i < arity; i++) {
            if (keyRow.isNullAt(i)) {
                features.put("feature_" + i, null);
            } else {
                try {
                    features.put("feature_" + i, getFieldValue(keyRow, i));
                } catch (Exception e) {
                    LOG.warn("Failed to extract field {} from RowData", i, e);
                    features.put("feature_" + i, null);
                }
            }
        }
        return features;
    }

    private Object getFieldValue(RowData row, int pos) {
        if (row.isNullAt(pos)) return null;
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