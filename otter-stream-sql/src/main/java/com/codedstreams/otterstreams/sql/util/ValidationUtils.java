package com.codedstreams.otterstreams.sql.util;

import java.util.Map;

/**
 * Validation utilities for input data.
 */
public class ValidationUtils {

    public static boolean validateFeatures(Map<String, Object> features) {
        if (features == null || features.isEmpty()) {
            return false;
        }

        for (Map.Entry<String, Object> entry : features.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null) {
                return false;
            }
        }

        return true;
    }

    public static boolean validateModelPath(String path) {
        if (path == null || path.trim().isEmpty()) {
            return false;
        }

        return path.startsWith("file://") ||
                path.startsWith("s3://") ||
                path.startsWith("hdfs://") ||
                path.startsWith("http://") ||
                path.startsWith("https://") ||
                path.startsWith("minio://");
    }
}
