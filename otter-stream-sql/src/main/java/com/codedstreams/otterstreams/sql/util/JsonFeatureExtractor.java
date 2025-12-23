package com.codedstreams.otterstreams.sql.util;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;


/**
 * Extracts features from JSON strings for model input.
 */
public class JsonFeatureExtractor {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static Map<String, Object> extractFeatures(String json) throws Exception {
        JsonNode root = MAPPER.readTree(json);
        Map<String, Object> features = new HashMap<>();

        root.fields().forEachRemaining(entry -> {
            features.put(entry.getKey(), convertNode(entry.getValue()));
        });

        return features;
    }

    private static Object convertNode(JsonNode node) {
        if (node.isDouble()) return node.asDouble();
        if (node.isInt()) return node.asInt();
        if (node.isLong()) return node.asLong();
        if (node.isBoolean()) return node.asBoolean();
        if (node.isArray()) {
            double[] array = new double[node.size()];
            for (int i = 0; i < node.size(); i++) {
                array[i] = node.get(i).asDouble();
            }
            return array;
        }
        return node.asText();
    }
}
