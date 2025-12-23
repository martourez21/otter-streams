package com.codedstreams.otterstreams.sql.util;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;

/**
 * Converts between Java objects and TensorFlow tensors.
 */
public class TensorConverter {

    /** Converts a Java object to a TensorFlow tensor. */
    public static Tensor toTensor(Object value) {
        if (value instanceof Double || value instanceof Float) {
            float floatValue = ((Number) value).floatValue();
            return TFloat32.scalarOf(floatValue);
        } else if (value instanceof Integer) {
            return TInt32.scalarOf((Integer) value);
        } else if (value instanceof Long) {
            return TInt64.scalarOf((Long) value);
        } else if (value instanceof double[]) {
            double[] array = (double[]) value;
            float[] floatArray = new float[array.length];
            for (int i = 0; i < array.length; i++) {
                floatArray[i] = (float) array[i];
            }
            TFloat32 tensor = TFloat32.tensorOf(Shape.of(array.length));
            for (int i = 0; i < floatArray.length; i++) {
                tensor.setFloat(floatArray[i], i);
            }
            return tensor;
        } else if (value instanceof float[]) {
            float[] array = (float[]) value;
            TFloat32 tensor = TFloat32.tensorOf(Shape.of(array.length));
            for (int i = 0; i < array.length; i++) {
                tensor.setFloat(array[i], i);
            }
            return tensor;
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            TInt32 tensor = TInt32.tensorOf(Shape.of(array.length));
            for (int i = 0; i < array.length; i++) {
                tensor.setInt(array[i], i);
            }
            return tensor;
        } else if (value instanceof long[]) {
            long[] array = (long[]) value;
            TInt64 tensor = TInt64.tensorOf(Shape.of(array.length));
            for (int i = 0; i < array.length; i++) {
                tensor.setLong(array[i], i);
            }
            return tensor;
        }

        throw new IllegalArgumentException("Unsupported type: " + value.getClass());
    }

    /** Converts a TensorFlow tensor to a Java object. */
    public static Object fromTensor(Tensor tensor) {
        Shape shape = tensor.shape();
        long numElements = shape.size();

        if (tensor instanceof TFloat32) {
            TFloat32 t = (TFloat32) tensor;
            if (shape.numDimensions() == 0) {
                return t.getFloat();
            } else {
                float[] array = new float[(int) numElements];
                for (int i = 0; i < numElements; i++) {
                    array[i] = t.getFloat(i);
                }
                return array;
            }
        } else if (tensor instanceof TInt32) {
            TInt32 t = (TInt32) tensor;
            if (shape.numDimensions() == 0) {
                return t.getInt();
            } else {
                int[] array = new int[(int) numElements];
                for (int i = 0; i < numElements; i++) {
                    array[i] = t.getInt(i);
                }
                return array;
            }
        } else if (tensor instanceof TInt64) {
            TInt64 t = (TInt64) tensor;
            if (shape.numDimensions() == 0) {
                return t.getLong();
            } else {
                long[] array = new long[(int) numElements];
                for (int i = 0; i < numElements; i++) {
                    array[i] = t.getLong(i);
                }
                return array;
            }
        }

        throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getClass());
    }
}
