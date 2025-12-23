package com.codedstreams.otterstreams.sql.util;

import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.types.DataType;

/**
 * Utilities for Flink type conversions.
 */
public class TypeUtils {

    public static DataType javaToFlinkType(Class<?> javaType) {
        if (javaType == Integer.class || javaType == int.class) {
            return DataTypes.INT();
        } else if (javaType == Long.class || javaType == long.class) {
            return DataTypes.BIGINT();
        } else if (javaType == Double.class || javaType == double.class) {
            return DataTypes.DOUBLE();
        } else if (javaType == Float.class || javaType == float.class) {
            return DataTypes.FLOAT();
        } else if (javaType == String.class) {
            return DataTypes.STRING();
        } else if (javaType == Boolean.class || javaType == boolean.class) {
            return DataTypes.BOOLEAN();
        } else if (javaType.isArray()) {
            Class<?> componentType = javaType.getComponentType();
            return DataTypes.ARRAY(javaToFlinkType(componentType));
        }
        return DataTypes.STRING();
    }

    public static Object convertToJavaType(Object value, Class<?> targetType) {
        if (value == null || targetType.isInstance(value)) {
            return value;
        }

        if (targetType == Double.class || targetType == double.class) {
            return ((Number) value).doubleValue();
        } else if (targetType == Float.class || targetType == float.class) {
            return ((Number) value).floatValue();
        } else if (targetType == Integer.class || targetType == int.class) {
            return ((Number) value).intValue();
        } else if (targetType == Long.class || targetType == long.class) {
            return ((Number) value).longValue();
        } else if (targetType == String.class) {
            return value.toString();
        }

        return value;
    }
}
