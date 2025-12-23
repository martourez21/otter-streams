package com.codedstreams.otterstreams.sql.connector;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.TableFunction;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Lookup function for temporal joins with ML predictions.
 */
public class MLInferenceLookupFunction extends TableFunction<RowData> {

    private final SqlInferenceConfig config;
    private transient ModelCache modelCache;

    public MLInferenceLookupFunction(SqlInferenceConfig config) {
        this.config = config;
    }

    @Override
    public void open(FunctionContext context) throws Exception {
        super.open(context);
        this.modelCache = ModelCache.getInstance();
    }

    public void eval(Object... keys) {
        try {
            InferenceEngine<?> engine = modelCache.getEngine(config.getModelName());
            if (engine == null) return;

            // Convert keys to features
            Map<String, Object> features = new HashMap<>();
            for (int i = 0; i < keys.length; i++) {
                features.put("feature_" + i, keys[i]);
            }

            InferenceResult result = engine.infer(features);

            if (result.isSuccess()) {
                GenericRowData row = new GenericRowData(2);
                Object prediction = result.getOutputs().values().iterator().next();
                row.setField(0, ((Number) prediction).doubleValue());
                row.setField(1, 1.0); // confidence
                collect(row);
            }
        } catch (Exception e) {
            // Log and skip
        }
    }
}
