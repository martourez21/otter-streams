package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.Result;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TensorFlow GraphDef (frozen graph) inference engine.
 *
 * <p>Unlike {@code SavedModelBundle} (used by {@code TensorFlowInferenceEngine} in
 * {@code otter-stream-tensorflow}), a raw GraphDef carries no signature metadata — there is no
 * built-in way to discover input/output tensor names from the file itself. This engine requires
 * them to be supplied explicitly via {@link ModelConfig#getModelOptions()}:
 *
 * <ul>
 *   <li>{@code inputTensorName} (required) — the name of the input placeholder/tensor</li>
 *   <li>{@code outputTensorNames} (required) — comma-separated list of output tensor names to fetch</li>
 * </ul>
 *
 * <h2>Example:</h2>
 * <pre>{@code
 * ModelConfig.builder()
 *     .modelId("fraud-graphdef")
 *     .modelPath("file:///models/fraud.pb")
 *     .format(ModelFormat.TENSORFLOW_GRAPHDEF)
 *     .modelOptions(Map.of(
 *         "inputTensorName", "input_1",
 *         "outputTensorNames", "output_1,output_2"))
 *     .build();
 * }</pre>
 */
public class TensorFlowGraphDefEngine extends LocalInferenceEngine<Graph> {
    private static final Logger LOG = LoggerFactory.getLogger(TensorFlowGraphDefEngine.class);

    private Session session;
    private String inputTensorName;
    private List<String> outputTensorNames;

    @Override
    protected void loadModelDirectly(ModelConfig config) throws InferenceException {
        try {
            String modelPath = config.getModelPath();
            LOG.info("Loading TensorFlow GraphDef from: {}", modelPath);

            byte[] graphDef = Files.readAllBytes(Paths.get(modelPath));

            this.loadedModel = new Graph();
            this.loadedModel.importGraphDef(GraphDef.parseFrom(graphDef));
            this.session = new Session(loadedModel);

            Map<String, Object> options = config.getModelOptions();
            Object inputNameOpt = options != null ? options.get("inputTensorName") : null;
            Object outputNamesOpt = options != null ? options.get("outputTensorNames") : null;
            if (inputNameOpt == null || outputNamesOpt == null) {
                throw new InferenceException(
                        "TensorFlowGraphDefEngine requires 'inputTensorName' and 'outputTensorNames' "
                                + "to be set in ModelConfig.modelOptions() — a raw GraphDef has no "
                                + "signature metadata to infer them from. See class Javadoc for an example.");
            }
            this.inputTensorName = inputNameOpt.toString();
            this.outputTensorNames = new ArrayList<>();
            for (String name : outputNamesOpt.toString().split(",")) {
                String trimmed = name.trim();
                if (!trimmed.isEmpty()) {
                    this.outputTensorNames.add(trimmed);
                }
            }
            if (this.outputTensorNames.isEmpty()) {
                throw new InferenceException("'outputTensorNames' resolved to an empty list");
            }

            LOG.info("GraphDef loaded successfully (input='{}', outputs={})", inputTensorName, outputTensorNames);
        } catch (InferenceException e) {
            throw e;
        } catch (Exception e) {
            throw new InferenceException("Failed to load GraphDef", e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        long startTime = System.currentTimeMillis();
        Session.Runner runner = session.runner();
        Result result = null;
        try {
            Object rawInput = inputs.get(inputTensorName);
            if (rawInput == null) {
                throw new InferenceException(
                        "No input provided under key '" + inputTensorName + "' (see 'inputTensorName' in modelOptions)");
            }
            try (Tensor tensor = createTensor(rawInput)) {
                runner.feed(inputTensorName, tensor);
            }
            for (String outputName : outputTensorNames) {
                runner.fetch(outputName);
            }

            result = runner.run();

            Map<String, Object> outputs = new HashMap<>();
            for (int i = 0; i < result.size() && i < outputTensorNames.size(); i++) {
                outputs.put(outputTensorNames.get(i), extractTensorValue(result.get(i)));
            }

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (InferenceException e) {
            throw e;
        } catch (Exception e) {
            throw new InferenceException("GraphDef inference failed", e);
        } finally {
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
        // Same documented simplification as TensorFlowInferenceEngine (otter-stream-tensorflow):
        // TensorFlow batches via tensor shape, not via repeated calls. Proper batching means
        // stacking batchInputs into one higher-rank tensor before a single runner.run() — left
        // as a follow-up since it requires knowing the batch axis convention per model, but
        // this at least runs a REAL inference per call rather than fabricating a result.
        if (batchInputs.length == 0) {
            return new InferenceResult(Map.of(), 0, modelConfig.getModelId());
        }
        return infer(batchInputs[0]);
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, false, 128, false);
    }

    @Override
    public ModelMetadata getMetadata() {
        return ModelMetadata.builder()
                .modelName(modelConfig.getModelId())
                .format(modelConfig.getFormat())
                .build();
    }

    @Override
    public void close() throws InferenceException {
        if (session != null) session.close();
        if (loadedModel != null) loadedModel.close();
        super.close();
    }

    /** Minimal float[]/int[] → Tensor conversion, matching TensorFlowInferenceEngine's conventions. */
    private Tensor createTensor(Object value) {
        if (value instanceof float[]) {
            float[] array = (float[]) value;
            return TFloat32.tensorOf(Shape.of(1, array.length), data -> {
                for (int i = 0; i < array.length; i++) {
                    data.setFloat(array[i], 0, i);
                }
            });
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            return TInt32.tensorOf(Shape.of(1, array.length), data -> {
                for (int i = 0; i < array.length; i++) {
                    data.setInt(array[i], 0, i);
                }
            });
        }
        throw new IllegalArgumentException("Unsupported input type for GraphDef inference: "
                + (value != null ? value.getClass() : "null") + " (expected float[] or int[])");
    }

    /** Minimal Tensor → Java value extraction (float outputs), matching TensorFlowInferenceEngine's conventions. */
    private Object extractTensorValue(Tensor tensor) {
        if (tensor instanceof TFloat32) {
            TFloat32 floatTensor = (TFloat32) tensor;
            long totalSize = tensor.shape().size();
            float[] values = new float[(int) totalSize];
            if (tensor.shape().numDimensions() <= 1) {
                for (int i = 0; i < totalSize; i++) {
                    values[i] = floatTensor.getFloat(i);
                }
            } else {
                int idx = 0;
                for (int i = 0; i < tensor.shape().size(0); i++) {
                    for (int j = 0; j < tensor.shape().size(1); j++) {
                        values[idx++] = floatTensor.getFloat(i, j);
                    }
                }
            }
            return values.length == 1 ? values[0] : values;
        }
        return tensor.toString();
    }
}
