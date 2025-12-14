package com.codedstream.otterstream.onnx;

import ai.onnxruntime.*;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

public class OnnxInferenceEngine extends LocalInferenceEngine<OrtSession> {

    private OrtEnvironment environment;
    private OrtSession session;

    private static final Logger LOG = LoggerFactory.getLogger(OnnxInferenceEngine.class);
    private static final long serialVersionUID = 1L;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;
            this.environment = OrtEnvironment.getEnvironment();

            // Configure session options
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

            // Set optimization level
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

            // Set thread options
            sessionOptions.setInterOpNumThreads(2);
            sessionOptions.setIntraOpNumThreads(4);

            // Load model
            this.session = environment.createSession(config.getModelPath(), sessionOptions);
            this.initialized = true;

        } catch (Exception e) {
            throw new InferenceException("Failed to initialize ONNX runtime", e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try {
            long startTime = System.currentTimeMillis();

            Map<String, OnnxTensor> inputTensors = createInputTensors(inputs);

            // Run inference - use the correct run method that takes Map
            OrtSession.Result results = session.run(inputTensors);

            Map<String, Object> outputs = extractOutputs(results);
            long endTime = System.currentTimeMillis();

            // Clean up input tensors
            for (OnnxTensor tensor : inputTensors.values()) {
                tensor.close();
            }

            // Clean up output tensors from results
            results.close();

            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("ONNX inference failed", e);
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        try {
            long startTime = System.currentTimeMillis();

            // For batch processing, we assume all inputs have same structure
            int batchSize = batchInputs.length;
            if (batchSize == 0) {
                return new InferenceResult(Map.of(), 0, modelConfig.getModelId());
            }

            // Get first input to understand structure
            Map<String, Object> firstInput = batchInputs[0];
            Map<String, OnnxTensor> batchTensors = new HashMap<>();

            for (String inputName : firstInput.keySet()) {
                // Combine all batch values for this input
                Object[] batchValues = new Object[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    batchValues[i] = batchInputs[i].get(inputName);
                }
                OnnxTensor batchTensor = createBatchTensor(inputName, batchValues);
                batchTensors.put(inputName, batchTensor);
            }

            OrtSession.Result results = session.run(batchTensors);
            Map<String, Object> outputs = extractOutputs(results);

            long endTime = System.currentTimeMillis();

            // Clean up
            for (OnnxTensor tensor : batchTensors.values()) {
                tensor.close();
            }
            results.close();

            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("ONNX batch inference failed", e);
        }
    }

    private Map<String, OnnxTensor> createInputTensors(Map<String, Object> inputs) throws Exception {
        Map<String, OnnxTensor> tensors = new HashMap<>();
        for (Map.Entry<String, Object> entry : inputs.entrySet()) {
            OnnxTensor tensor = createTensor(entry.getValue());
            tensors.put(entry.getKey(), tensor);
        }
        return tensors;
    }

    private OnnxTensor createTensor(Object value) throws Exception {
        if (value instanceof float[]) {
            float[] array = (float[]) value;
            // Create tensor with correct shape [1, array_length]
            long[] shape = new long[]{1, array.length};
            return OnnxTensor.createTensor(environment, FloatBuffer.wrap(array), shape);
        } else if (value instanceof float[][]) {
            float[][] array = (float[][]) value;
            // For 2D arrays, use the array directly
            return OnnxTensor.createTensor(environment, array);
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            long[] shape = new long[]{1, array.length};
            return OnnxTensor.createTensor(environment, IntBuffer.wrap(array), shape);
        } else if (value instanceof long[]) {
            long[] array = (long[]) value;
            long[] shape = new long[]{1, array.length};
            return OnnxTensor.createTensor(environment, LongBuffer.wrap(array), shape);
        } else {
            throw new InferenceException("Unsupported input type: " + value.getClass());
        }
    }

    // Fixed helper methods for OnnxInferenceEngine

    /**
     * Creates an ONNX tensor from batch values with proper type handling
     */
    private OnnxTensor createBatchTensor(String inputName, Object[] batchValues) throws OrtException {
        if (batchValues == null || batchValues.length == 0) {
            throw new IllegalArgumentException("Batch values cannot be null or empty for input: " + inputName);
        }

        Object firstValue = batchValues[0];
        int batchSize = batchValues.length;

        // Handle float array batches
        if (firstValue instanceof float[]) {
            int featureSize = ((float[]) firstValue).length;
            float[][] batchArray = new float[batchSize][featureSize];

            for (int i = 0; i < batchSize; i++) {
                if (batchValues[i] instanceof float[]) {
                    batchArray[i] = (float[]) batchValues[i];
                } else {
                    throw new IllegalArgumentException(
                            String.format("Inconsistent batch types at index %d for input: %s", i, inputName)
                    );
                }
            }

            long[] shape = {batchSize, featureSize};
            return OnnxTensor.createTensor(environment, batchArray);
        }

        // Handle int array batches
        else if (firstValue instanceof int[]) {
            int featureSize = ((int[]) firstValue).length;
            int[][] batchArray = new int[batchSize][featureSize];

            for (int i = 0; i < batchSize; i++) {
                if (batchValues[i] instanceof int[]) {
                    batchArray[i] = (int[]) batchValues[i];
                } else {
                    throw new IllegalArgumentException(
                            String.format("Inconsistent batch types at index %d for input: %s", i, inputName)
                    );
                }
            }

            long[] shape = {batchSize, featureSize};
            return OnnxTensor.createTensor(environment, batchArray);
        }

        // Handle long array batches
        else if (firstValue instanceof long[]) {
            int featureSize = ((long[]) firstValue).length;
            long[][] batchArray = new long[batchSize][featureSize];

            for (int i = 0; i < batchSize; i++) {
                if (batchValues[i] instanceof long[]) {
                    batchArray[i] = (long[]) batchValues[i];
                } else {
                    throw new IllegalArgumentException(
                            String.format("Inconsistent batch types at index %d for input: %s", i, inputName)
                    );
                }
            }

            long[] shape = {batchSize, featureSize};
            return OnnxTensor.createTensor(environment, batchArray);
        }

        // Handle double array batches
        else if (firstValue instanceof double[]) {
            int featureSize = ((double[]) firstValue).length;
            double[][] batchArray = new double[batchSize][featureSize];

            for (int i = 0; i < batchSize; i++) {
                if (batchValues[i] instanceof double[]) {
                    batchArray[i] = (double[]) batchValues[i];
                } else {
                    throw new IllegalArgumentException(
                            String.format("Inconsistent batch types at index %d for input: %s", i, inputName)
                    );
                }
            }

            long[] shape = {batchSize, featureSize};
            return OnnxTensor.createTensor(environment, batchArray);
        }

        // Handle String array batches
        else if (firstValue instanceof String[]) {
            int featureSize = ((String[]) firstValue).length;
            String[][] batchArray = new String[batchSize][featureSize];

            for (int i = 0; i < batchSize; i++) {
                if (batchValues[i] instanceof String[]) {
                    batchArray[i] = (String[]) batchValues[i];
                } else {
                    throw new IllegalArgumentException(
                            String.format("Inconsistent batch types at index %d for input: %s", i, inputName)
                    );
                }
            }

            long[] shape = {batchSize, featureSize};
            return OnnxTensor.createTensor(environment, batchArray);
        }

        throw new IllegalArgumentException(
                String.format("Unsupported batch input type for '%s': %s",
                        inputName,
                        firstValue.getClass().getName())
        );
    }

    /**
     * Extracts outputs from ONNX Runtime results with proper type handling and cleanup
     */
    private Map<String, Object> extractOutputs(OrtSession.Result results) throws OrtException {
        Map<String, Object> outputs = new HashMap<>();

        try {
            // Iterate through all output values
            for (Map.Entry<String, OnnxValue> entry : results) {
                String outputName = entry.getKey();
                OnnxValue value = entry.getValue();

                if (value == null) {
                    LOG.warn("Null value for output: {}", outputName);
                    continue;
                }

                // Handle OnnxTensor (most common case)
                if (value instanceof OnnxTensor) {
                    OnnxTensor tensor = (OnnxTensor) value;
                    Object tensorValue = extractTensorValue(tensor, outputName);
                    outputs.put(outputName, tensorValue);
                }
                // Handle OnnxSequence
                else if (value instanceof OnnxSequence) {
                    OnnxSequence sequence = (OnnxSequence) value;
                    List<Object> sequenceValues = new ArrayList<>();
                    for (OnnxValue seqValue : sequence.getValue()) {
                        if (seqValue instanceof OnnxTensor) {
                            sequenceValues.add(extractTensorValue((OnnxTensor) seqValue, outputName));
                        }
                    }
                    outputs.put(outputName, sequenceValues);
                }
                // Handle OnnxMap
                else if (value instanceof OnnxMap) {
                    OnnxMap map = (OnnxMap) value;
                    outputs.put(outputName, map.getValue());
                }
                // Handle unknown types
                else {
                    LOG.warn("Unsupported ONNX value type for output '{}': {}",
                            outputName, value.getClass().getName());
                    outputs.put(outputName, value.toString());
                }
            }
        } catch (OrtException e) {
            LOG.error("Error extracting outputs from ONNX results", e);
            throw e;
        }

        return outputs;
    }

    /**
     * Extracts the actual value from an OnnxTensor based on its type
     */
    private Object extractTensorValue(OnnxTensor tensor, String outputName) throws OrtException {
        TensorInfo info = tensor.getInfo();
        OnnxJavaType type = info.type;
        long[] shape = info.getShape();

        try {
            switch (type) {
                case FLOAT:
                    return extractFloatTensor(tensor, shape);

                case DOUBLE:
                    return extractDoubleTensor(tensor, shape);

                case INT8:
                case INT16:
                case INT32:
                    return extractIntTensor(tensor, shape);

                case INT64:
                    return extractLongTensor(tensor, shape);

                case STRING:
                    return extractStringTensor(tensor, shape);

                case BOOL:
                    return extractBooleanTensor(tensor, shape);

                default:
                    LOG.warn("Unsupported tensor type for output '{}': {}", outputName, type);
                    return tensor.getValue();
            }
        } catch (Exception e) {
            LOG.error("Error extracting tensor value for output: {}", outputName, e);
            throw new RuntimeException("Failed to extract tensor value: " + e.getMessage(), e);        }
    }

    /**
     * Extract float tensor with shape handling
     */
    private Object extractFloatTensor(OnnxTensor tensor, long[] shape) throws OrtException {
        if (shape.length == 1) {
            // 1D array
            float[] result = new float[(int) shape[0]];
            tensor.getFloatBuffer().get(result);
            return result;
        } else if (shape.length == 2) {
            // 2D array (batch)
            float[][] result = (float[][]) tensor.getValue();
            return result;
        } else {
            // Multi-dimensional array
            return tensor.getValue();
        }
    }

    /**
     * Extract double tensor with shape handling
     */
    private Object extractDoubleTensor(OnnxTensor tensor, long[] shape) throws OrtException {
        if (shape.length == 1) {
            double[] result = new double[(int) shape[0]];
            tensor.getDoubleBuffer().get(result);
            return result;
        } else if (shape.length == 2) {
            double[][] result = (double[][]) tensor.getValue();
            return result;
        } else {
            return tensor.getValue();
        }
    }

    /**
     * Extract int tensor with shape handling
     */
    private Object extractIntTensor(OnnxTensor tensor, long[] shape) throws OrtException {
        if (shape.length == 1) {
            int[] result = new int[(int) shape[0]];
            tensor.getIntBuffer().get(result);
            return result;
        } else if (shape.length == 2) {
            int[][] result = (int[][]) tensor.getValue();
            return result;
        } else {
            return tensor.getValue();
        }
    }

    /**
     * Extract long tensor with shape handling
     */
    private Object extractLongTensor(OnnxTensor tensor, long[] shape) throws OrtException {
        if (shape.length == 1) {
            long[] result = new long[(int) shape[0]];
            tensor.getLongBuffer().get(result);
            return result;
        } else if (shape.length == 2) {
            long[][] result = (long[][]) tensor.getValue();
            return result;
        } else {
            return tensor.getValue();
        }
    }

    /**
     * Extract String tensor
     */
    private Object extractStringTensor(OnnxTensor tensor, long[] shape) throws OrtException {
        if (shape.length == 1) {
            String[] result = (String[]) tensor.getValue();
            return result;
        } else if (shape.length == 2) {
            String[][] result = (String[][]) tensor.getValue();
            return result;
        } else {
            return tensor.getValue();
        }
    }

    /**
     * Extract boolean tensor
     */
    private Object extractBooleanTensor(OnnxTensor tensor, long[] shape) throws OrtException {
        if (shape.length == 1) {
            boolean[] result = (boolean[]) tensor.getValue();
            return result;
        } else if (shape.length == 2) {
            boolean[][] result = (boolean[][]) tensor.getValue();
            return result;
        } else {
            return tensor.getValue();
        }
    }

    /**
     * Helper method to validate tensor shapes match expectations
     */
    private void validateTensorShape(long[] actualShape, long[] expectedShape, String tensorName) {
        if (expectedShape == null) {
            return; // Skip validation if no expected shape
        }

        if (actualShape.length != expectedShape.length) {
            throw new IllegalArgumentException(
                    String.format("Shape mismatch for tensor '%s': expected rank %d but got %d",
                            tensorName, expectedShape.length, actualShape.length)
            );
        }

        for (int i = 0; i < actualShape.length; i++) {
            // -1 in expected shape means dynamic dimension (any size allowed)
            if (expectedShape[i] != -1 && actualShape[i] != expectedShape[i]) {
                throw new IllegalArgumentException(
                        String.format("Shape mismatch for tensor '%s' at dimension %d: expected %d but got %d",
                                tensorName, i, expectedShape[i], actualShape[i])
                );
            }
        }
    }

    /**
     * Helper to get total element count from shape
     */
    private long getElementCount(long[] shape) {
        long count = 1;
        for (long dim : shape) {
            count *= dim;
        }
        return count;
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, true, 256, true);
    }

    @Override
    public void close() throws InferenceException {
        try {
            if (session != null) {
                session.close();
            }
            if (environment != null) {
                environment.close();
            }
        } catch (Exception e) {
            throw new InferenceException("Failed to close ONNX resources", e);
        } finally {
            super.close();
        }
    }

    @Override
    public ModelMetadata getMetadata() {
        return null;
    }
}