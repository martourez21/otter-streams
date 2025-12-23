package com.codedstreams.otterstreams.sql.metadata;


import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Extracts input/output schema from loaded models.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class InputOutputSchemaExtractor {
    private static final Logger LOG = LoggerFactory.getLogger(InputOutputSchemaExtractor.class);

    /**
     * Extracts schema from inference engine.
     */
    public static InputOutputSchema extractSchema(InferenceEngine<?> engine) {
        try {
            ModelMetadata metadata = engine.getMetadata();

            List<InputOutputSchema.FieldSchema> inputs = new ArrayList<>();
            List<InputOutputSchema.FieldSchema> outputs = new ArrayList<>();

            // Extract from metadata if available
            if (metadata != null && metadata.getInputSchema() != null) {
                for (Map.Entry<String, Object> entry : metadata.getInputSchema().entrySet()) {
                    inputs.add(new InputOutputSchema.FieldSchema(
                            entry.getKey(),
                            inferType(entry.getValue()),
                            inferShape(entry.getValue())
                    ));
                }
            }

            if (metadata != null && metadata.getOutputSchema() != null) {
                for (Map.Entry<String, Object> entry : metadata.getOutputSchema().entrySet()) {
                    outputs.add(new InputOutputSchema.FieldSchema(
                            entry.getKey(),
                            inferType(entry.getValue()),
                            inferShape(entry.getValue())
                    ));
                }
            }

            return new InputOutputSchema(inputs, outputs);
        } catch (Exception e) {
            LOG.warn("Could not extract schema from engine", e);
            return new InputOutputSchema(List.of(), List.of());
        }
    }

    private static String inferType(Object value) {
        if (value instanceof String) {
            String str = (String) value;
            if (str.contains("float")) return "float32";
            if (str.contains("int")) return "int32";
            if (str.contains("double")) return "float64";
        }
        return "unknown";
    }

    private static List<Integer> inferShape(Object value) {
        // Default shape
        return List.of(-1);
    }
}
