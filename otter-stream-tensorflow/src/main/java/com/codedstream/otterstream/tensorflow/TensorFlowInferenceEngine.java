package com.codedstream.otterstream.tensorflow;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

import java.util.Map;
import java.util.HashMap;
import java.util.List;

public class TensorFlowInferenceEngine extends LocalInferenceEngine<SavedModelBundle> {

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;
            this.loadedModel = SavedModelBundle.load(config.getModelPath(), "serve");
            this.initialized = true;
        } catch (Exception e) {
            throw new InferenceException("Failed to load TensorFlow model from: " + config.getModelPath(), e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try (Session.Runner runner = loadedModel.session().runner()) {
            long startTime = System.currentTimeMillis();

            // Prepare inputs
            for (Map.Entry<String, Object> entry : inputs.entrySet()) {
                Tensor tensor = createTensor(entry.getValue());
                runner.feed(entry.getKey(), tensor);
                // Note: Tensor is managed by the session and will be closed automatically
            }

            // Get output tensor names (simplified - in production, parse signature)
            List<String> outputNames = getOutputNames();
            for (String outputName : outputNames) {
                runner.fetch(outputName);
            }

            // Run inference
            Result results = runner.run();

            // Extract outputs
            Map<String, Object> outputs = new HashMap<>();
            for (int i = 0; i < results.size(); i++) {
                Tensor tensor = results.get(i);
                String outputName = outputNames.get(i);
                outputs.put(outputName, extractTensorValue(tensor));
                tensor.close(); // Close the output tensor
            }

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("TensorFlow inference failed", e);
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
    public void close() throws InferenceException {
        if (loadedModel != null) {
            loadedModel.close();
        }
        super.close();
    }

    private Tensor createTensor(Object value) {
        if (value instanceof float[]) {
            float[] array = (float[]) value;
            return TFloat32.tensorOf(Shape.of(1, array.length),
                    data -> data.put(array));
        } else if (value instanceof float[][]) {
            float[][] array = (float[][]) value;
            return TFloat32.tensorOf(Shape.of(array.length, array[0].length),
                    data -> {
                        for (float[] row : array) {
                            data.put(row);
                        }
                    });
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            return TInt32.tensorOf(Shape.of(1, array.length),
                    data -> data.put(array));
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + value.getClass());
        }
    }

    private Object extractTensorValue(Tensor tensor) {
        // Extract value based on tensor type
        if (tensor instanceof TFloat32) {
            TFloat32 floatTensor = (TFloat32) tensor;
            float[] values = new float[(int) tensor.shape().size()];
            floatTensor.copyTo(values, 0, values.length);

            if (tensor.shape().numDimensions() == 1 && values.length == 1) {
                return values[0];
            } else if (tensor.shape().numDimensions() == 2 && tensor.shape().size(0) == 1) {
                // Return first row for 2D tensor with single row
                float[] firstRow = new float[(int) tensor.shape().size(1)];
                System.arraycopy(values, 0, firstRow, 0, firstRow.length);
                return firstRow;
            }
            return values;
        } else if (tensor instanceof TInt32) {
            TInt32 intTensor = (TInt32) tensor;
            int[] values = new int[(int) tensor.shape().size()];
            intTensor.copyTo(values, 0, values.length);

            if (tensor.shape().numDimensions() == 1 && values.length == 1) {
                return values[0];
            }
            return values;
        } else {
            // Fallback for other types
            return tensor.toString();
        }
    }

    private List<String> getOutputNames() {
        // In a real implementation, parse the model signature
        // For now, return common output names
        return java.util.Arrays.asList("output", "predictions", "scores");
    }
}