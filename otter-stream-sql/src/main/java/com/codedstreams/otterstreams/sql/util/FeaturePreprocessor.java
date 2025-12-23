package com.codedstreams.otterstreams.sql.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Preprocesses features before inference (normalization, encoding, etc.).
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class FeaturePreprocessor {
    private static final Logger LOG = LoggerFactory.getLogger(FeaturePreprocessor.class);

    /**
     * Normalizes numerical features to [0, 1] range.
     */
    public static Map<String, Object> normalize(Map<String, Object> features,
                                                Map<String, Double> minValues,
                                                Map<String, Double> maxValues) {
        Map<String, Object> normalized = new HashMap<>(features);

        for (Map.Entry<String, Object> entry : features.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            if (value instanceof Number && minValues.containsKey(key) && maxValues.containsKey(key)) {
                double val = ((Number) value).doubleValue();
                double min = minValues.get(key);
                double max = maxValues.get(key);

                if (max > min) {
                    double normalizedValue = (val - min) / (max - min);
                    normalized.put(key, normalizedValue);
                }
            }
        }

        return normalized;
    }

    /**
     * Standardizes features to zero mean and unit variance.
     */
    public static Map<String, Object> standardize(Map<String, Object> features,
                                                  Map<String, Double> means,
                                                  Map<String, Double> stdDevs) {
        Map<String, Object> standardized = new HashMap<>(features);

        for (Map.Entry<String, Object> entry : features.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            if (value instanceof Number && means.containsKey(key) && stdDevs.containsKey(key)) {
                double val = ((Number) value).doubleValue();
                double mean = means.get(key);
                double stdDev = stdDevs.get(key);

                if (stdDev > 0) {
                    double standardizedValue = (val - mean) / stdDev;
                    standardized.put(key, standardizedValue);
                }
            }
        }

        return standardized;
    }

    /**
     * One-hot encodes categorical features.
     */
    public static Map<String, Object> oneHotEncode(Map<String, Object> features,
                                                   Map<String, String[]> categoricalMappings) {
        Map<String, Object> encoded = new HashMap<>(features);

        for (Map.Entry<String, String[]> entry : categoricalMappings.entrySet()) {
            String featureName = entry.getKey();
            String[] categories = entry.getValue();
            Object value = features.get(featureName);

            if (value != null) {
                String stringValue = value.toString();
                for (int i = 0; i < categories.length; i++) {
                    String encodedKey = featureName + "_" + categories[i];
                    encoded.put(encodedKey, categories[i].equals(stringValue) ? 1.0 : 0.0);
                }
                encoded.remove(featureName);  // Remove original categorical
            }
        }

        return encoded;
    }

    /**
     * Fills missing values with defaults.
     */
    public static Map<String, Object> fillMissingValues(Map<String, Object> features,
                                                        Map<String, Object> defaults) {
        Map<String, Object> filled = new HashMap<>(features);

        for (Map.Entry<String, Object> entry : defaults.entrySet()) {
            if (!filled.containsKey(entry.getKey()) || filled.get(entry.getKey()) == null) {
                filled.put(entry.getKey(), entry.getValue());
            }
        }

        return filled;
    }
}
