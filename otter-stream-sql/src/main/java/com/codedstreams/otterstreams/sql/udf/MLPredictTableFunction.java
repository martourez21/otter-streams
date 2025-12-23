package com.codedstreams.otterstreams.sql.udf;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import com.codedstreams.otterstreams.sql.util.JsonFeatureExtractor;
import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.annotation.FunctionHint;
import org.apache.flink.table.functions.TableFunction;
import org.apache.flink.types.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

/**
 * Table function that returns multiple rows.
 *
 * Usage: SELECT * FROM table, LATERAL TABLE(ML_PREDICT_TABLE('model', features))
 */
@FunctionHint(output = @DataTypeHint("ROW<item_id STRING, score DOUBLE>"))
public class MLPredictTableFunction extends TableFunction<Row> {
    private static final Logger LOG = LoggerFactory.getLogger(MLPredictTableFunction.class);
    private transient ModelCache modelCache;

    public void eval(String modelName, String featuresJson) {
        try {
            if (modelCache == null) {
                modelCache = ModelCache.getInstance();
            }

            InferenceEngine<?> engine = modelCache.getEngine(modelName);
            if (engine == null) return;

            Map<String, Object> features = JsonFeatureExtractor.extractFeatures(featuresJson);
            InferenceResult result = engine.infer(features);

            if (result.isSuccess()) {
                // Assuming result contains array of items
                Map<String, Object> outputs = result.getOutputs();
                for (Map.Entry<String, Object> entry : outputs.entrySet()) {
                    Row row = Row.of(entry.getKey(), ((Number) entry.getValue()).doubleValue());
                    collect(row);
                }
            }
        } catch (Exception e) {
            LOG.error("Table function inference failed", e);
        }
    }
}
