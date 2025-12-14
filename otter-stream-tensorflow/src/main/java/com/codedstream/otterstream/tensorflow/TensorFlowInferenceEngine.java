package com.codedstream.otterstream.tensorflow;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.SignatureDef;
import org.tensorflow.proto.TensorInfo;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

import java.util.Map;
import java.util.HashMap;
import java.util.List;

public class TensorFlowInferenceEngine extends LocalInferenceEngine<SavedModelBundle> {

    private List<String> cachedInputNames;
    private List<String> cachedOutputNames;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;
            this.loadedModel = SavedModelBundle.load(config.getModelPath(), "serve");

            // Parse and cache signature information
            this.cachedInputNames = parseInputNames();
            this.cachedOutputNames = parseOutputNames();

            this.initialized = true;
        } catch (Exception e) {
            throw new InferenceException("Failed to load TensorFlow model from: " + config.getModelPath(), e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        Session.Runner runner = null;
        Result result = null;
        try {
            runner = loadedModel.session().runner();
            long startTime = System.currentTimeMillis();

            // Prepare inputs
            for (Map.Entry<String, Object> entry : inputs.entrySet()) {
                try (Tensor tensor = createTensor(entry.getValue())) {
                    runner.feed(entry.getKey(), tensor);
                }
            }

            // Get output tensor names
            List<String> outputNames = cachedOutputNames;
            for (String outputName : outputNames) {
                runner.fetch(outputName);
            }

            // Run inference
            result = runner.run();

            // Extract outputs
            Map<String, Object> outputs = new HashMap<>();
            for (int i = 0; i < result.size() && i < outputNames.size(); i++) {
                Tensor tensor = result.get(i);
                String outputName = outputNames.get(i);
                outputs.put(outputName, extractTensorValue(tensor));
            }

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("TensorFlow inference failed", e);
        } finally {
            // Close all output tensors
            if (result != null) {
                for (int i = 0; i < result.size(); i++) {
                    Tensor tensor = result.get(i);
                    if (tensor != null) {
                        tensor.close();
                    }
                }
            }
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        // TensorFlow handles batching through tensor shapes
        // For simplicity, we'll process the first input
        // In production, you'd create proper batch tensors
        if (batchInputs.length == 0) {
            return new InferenceResult(Map.of(), 0, modelConfig.getModelId());
        }
        return infer(batchInputs[0]);
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, true, 128, true);
    }

    @Override
    public ModelMetadata getMetadata() {
        Map<String, Object> inputSchemaMap = new HashMap<>();
        if (cachedInputNames != null) {
            for (int i = 0; i < cachedInputNames.size(); i++) {
                inputSchemaMap.put("input_" + i, cachedInputNames.get(i));
            }
        }

        Map<String, Object> outputSchemaMap = new HashMap<>();
        if (cachedOutputNames != null) {
            for (int i = 0; i < cachedOutputNames.size(); i++) {
                outputSchemaMap.put("output_" + i, cachedOutputNames.get(i));
            }
        }

        // Use modelId from config as modelName, or use a descriptive name
        String modelName = modelConfig.getModelId() != null ?
                modelConfig.getModelId() : "tensorflow-model";

        return ModelMetadata.builder()
                .modelName(modelName)
                .modelVersion(modelConfig.getModelVersion() != null ?
                        modelConfig.getModelVersion() : "unknown")
                .format(ModelFormat.TENSORFLOW_SAVEDMODEL)
                .inputSchema(inputSchemaMap)
                .outputSchema(outputSchemaMap)
                .modelSize(0L)
                .loadTimestamp(System.currentTimeMillis())
                .build();
    }

    @Override
    public void close() throws InferenceException {
        if (loadedModel != null) {
            loadedModel.close();
        }
        super.close();
    }

    private Tensor createTensor(Object value) {
        if (value instanceof float[]) {
            float[] array = (float[]) value;
            return TFloat32.tensorOf(Shape.of(1, array.length), data -> {
                long[] indices = new long[2];
                for (int i = 0; i < array.length; i++) {
                    indices[0] = 0;
                    indices[1] = i;
                    data.setFloat(array[i], indices);
                }
            });
        } else if (value instanceof float[][]) {
            float[][] array = (float[][]) value;
            int rows = array.length;
            int cols = array[0].length;
            return TFloat32.tensorOf(Shape.of(rows, cols), data -> {
                long[] indices = new long[2];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        indices[0] = i;
                        indices[1] = j;
                        data.setFloat(array[i][j], indices);
                    }
                }
            });
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            return TInt32.tensorOf(Shape.of(1, array.length), data -> {
                long[] indices = new long[2];
                for (int i = 0; i < array.length; i++) {
                    indices[0] = 0;
                    indices[1] = i;
                    data.setInt(array[i], indices);
                }
            });
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + value.getClass());
        }
    }

    private Object extractTensorValue(Tensor tensor) {
        // Extract value based on tensor type
        if (tensor instanceof TFloat32) {
            TFloat32 floatTensor = (TFloat32) tensor;
            long totalSize = tensor.shape().size();
            float[] values = new float[(int) totalSize];

            // Read data using direct ndarray access
            if (tensor.shape().numDimensions() == 1) {
                for (int i = 0; i < totalSize; i++) {
                    values[i] = floatTensor.getFloat(i);
                }
            } else if (tensor.shape().numDimensions() == 2) {
                int idx = 0;
                for (int i = 0; i < tensor.shape().size(0); i++) {
                    for (int j = 0; j < tensor.shape().size(1); j++) {
                        values[idx++] = floatTensor.getFloat(i, j);
                    }
                }
            } else {
                // For higher dimensions, flatten and read
                long[] indices = new long[tensor.shape().numDimensions()];
                for (int i = 0; i < totalSize; i++) {
                    // Convert flat index to multi-dimensional indices
                    long remaining = i;
                    for (int d = tensor.shape().numDimensions() - 1; d >= 0; d--) {
                        indices[d] = remaining % tensor.shape().size(d);
                        remaining /= tensor.shape().size(d);
                    }
                    values[i] = floatTensor.getFloat(indices);
                }
            }

            if (tensor.shape().numDimensions() == 1 && values.length == 1) {
                return values[0];
            } else if (tensor.shape().numDimensions() == 2 && tensor.shape().size(0) == 1) {
                // Return first row for 2D tensor with single row
                int cols = (int) tensor.shape().size(1);
                float[] firstRow = new float[cols];
                System.arraycopy(values, 0, firstRow, 0, cols);
                return firstRow;
            }
            return values;
        } else if (tensor instanceof TInt32) {
            TInt32 intTensor = (TInt32) tensor;
            long totalSize = tensor.shape().size();
            int[] values = new int[(int) totalSize];

            // Read data using direct ndarray access
            if (tensor.shape().numDimensions() == 1) {
                for (int i = 0; i < totalSize; i++) {
                    values[i] = intTensor.getInt(i);
                }
            } else if (tensor.shape().numDimensions() == 2) {
                int idx = 0;
                for (int i = 0; i < tensor.shape().size(0); i++) {
                    for (int j = 0; j < tensor.shape().size(1); j++) {
                        values[idx++] = intTensor.getInt(i, j);
                    }
                }
            } else {
                // For higher dimensions, flatten and read
                long[] indices = new long[tensor.shape().numDimensions()];
                for (int i = 0; i < totalSize; i++) {
                    long remaining = i;
                    for (int d = tensor.shape().numDimensions() - 1; d >= 0; d--) {
                        indices[d] = remaining % tensor.shape().size(d);
                        remaining /= tensor.shape().size(d);
                    }
                    values[i] = intTensor.getInt(indices);
                }
            }

            if (tensor.shape().numDimensions() == 1 && values.length == 1) {
                return values[0];
            }
            return values;
        } else {
            // Fallback for other types
            return tensor.toString();
        }
    }

    private List<String> parseInputNames() {
        try {
            SignatureDef signatureDef = getSignatureDef();

            if (signatureDef != null) {
                List<String> inputNames = new java.util.ArrayList<>();
                Map<String, TensorInfo> inputs = signatureDef.getInputsMap();

                for (Map.Entry<String, TensorInfo> entry : inputs.entrySet()) {
                    inputNames.add(entry.getValue().getName());
                }

                if (!inputNames.isEmpty()) {
                    return inputNames;
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to parse input names from signature: " + e.getMessage());
        }

        return new java.util.ArrayList<>();
    }

    private List<String> parseOutputNames() {
        try {
            SignatureDef signatureDef = getSignatureDef();

            if (signatureDef != null) {
                List<String> outputNames = new java.util.ArrayList<>();
                Map<String, TensorInfo> outputs = signatureDef.getOutputsMap();

                for (Map.Entry<String, TensorInfo> entry : outputs.entrySet()) {
                    outputNames.add(entry.getValue().getName());
                }

                if (!outputNames.isEmpty()) {
                    return outputNames;
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to parse output names from signature: " + e.getMessage());
        }

        // Fallback: return common output names
        return java.util.Arrays.asList("output", "predictions", "scores");
    }

    private SignatureDef getSignatureDef() {
        try {
            // Try to get the default serving signature
            SignatureDef signatureDef = loadedModel.metaGraphDef()
                    .getSignatureDefOrDefault("serving_default", null);

            if (signatureDef == null) {
                // Fallback to any available signature
                Map<String, SignatureDef> signatureMap = loadedModel.metaGraphDef()
                        .getSignatureDefMap();
                if (!signatureMap.isEmpty()) {
                    signatureDef = signatureMap.values().iterator().next();
                }
            }

            return signatureDef;
        } catch (Exception e) {
            System.err.println("Failed to retrieve model signature: " + e.getMessage());
            return null;
        }
    }
}