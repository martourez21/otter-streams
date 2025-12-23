package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import com.codedstreams.otterstreams.sql.util.TensorConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.SignatureDef;

import java.util.*;

/**
 * TensorFlow SavedModel inference engine.
 */
public class TensorFlowSavedModelEngine extends LocalInferenceEngine<SavedModelBundle> {

    private static final Logger LOG = LoggerFactory.getLogger(TensorFlowSavedModelEngine.class);

    private Session session;
    private List<String> inputNames;
    private List<String> outputNames;

    @Override
    protected void loadModelDirectly(ModelConfig config) throws InferenceException {
        try {
            String modelPath = config.getModelPath();
            LOG.info("Loading TensorFlow SavedModel from: {}", modelPath);

            this.loadedModel = SavedModelBundle.load(modelPath, "serve");
            this.session = loadedModel.session();

            // Extract input/output names from signature def
            this.inputNames = extractInputNames();
            this.outputNames = extractOutputNames();

            LOG.info("Model loaded successfully. Inputs: {}, Outputs: {}", inputNames, outputNames);
        } catch (Exception e) {
            throw new InferenceException("Failed to load SavedModel", e);
        }
    }

    private List<String> extractInputNames() {
        List<String> names = new ArrayList<>();
        Map<String, SignatureDef> signatures = loadedModel.metaGraphDef().getSignatureDefMap();
        SignatureDef sig = signatures.get("serving_default");
        if (sig != null) {
            names.addAll(sig.getInputsMap().keySet());
        }
        if (names.isEmpty()) {
            names.add("serving_default_input"); // fallback
        }
        return names;
    }

    private List<String> extractOutputNames() {
        List<String> names = new ArrayList<>();
        Map<String, SignatureDef> signatures = loadedModel.metaGraphDef().getSignatureDefMap();
        SignatureDef sig = signatures.get("serving_default");
        if (sig != null) {
            names.addAll(sig.getOutputsMap().keySet());
        }
        if (names.isEmpty()) {
            names.add("StatefulPartitionedCall"); // fallback
        }
        return names;
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        long startTime = System.currentTimeMillis();
        List<Tensor> inputTensors = new ArrayList<>();

        try {
            Session.Runner runner = session.runner();

            // Convert inputs to tensors and feed
            int index = 0;
            for (Map.Entry<String, Object> entry : inputs.entrySet()) {
                Tensor tensor = TensorConverter.toTensor(entry.getValue());
                inputTensors.add(tensor);
                String inputName = index < inputNames.size() ? inputNames.get(index) : entry.getKey();
                runner.feed(inputName, tensor);
                index++;
            }

            // Fetch outputs
            for (String outputName : outputNames) {
                runner.fetch(outputName);
            }

            // Run inference and get results
            Result outputs = runner.run();
            Map<String, Object> results = new HashMap<>();

            for (int i = 0; i < outputs.size(); i++) {
                Tensor t = outputs.get(i);
                try {
                    String key = i < outputNames.size() ? outputNames.get(i) : "output_" + i;
                    results.put(key, TensorConverter.fromTensor(t));
                } finally {
                    t.close(); // safely close each output tensor
                }
            }

            long endTime = System.currentTimeMillis();
            return new InferenceResult(results, endTime - startTime, modelConfig.getModelId());

        } catch (Exception e) {
            throw new InferenceException("Inference execution failed", e);
        } finally {
            // Close input tensors
            for (Tensor t : inputTensors) {
                t.close();
            }
        }
    }

    /**
     * Performs batch inference by running each input separately.
     * Returns a list of InferenceResult objects, one per input.
     */
    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        long startTime = System.currentTimeMillis();
        Map<String, Object> combinedOutputs = new HashMap<>();

        for (int i = 0; i < batchInputs.length; i++) {
            InferenceResult result = infer(batchInputs[i]);
            // Use a prefix for each input in the batch
            combinedOutputs.put("input_" + i, result.getOutputs());
        }

        long endTime = System.currentTimeMillis();
        return new InferenceResult(combinedOutputs, endTime - startTime, modelConfig.getModelId());
    }


    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, false, 128, false);
    }

    @Override
    public ModelMetadata getMetadata() {
        return ModelMetadata.builder()
                .modelName(modelConfig.getModelId())
                .modelVersion(modelConfig.getModelVersion())
                .format(modelConfig.getFormat())
                .build();
    }

    @Override
    public void close() throws InferenceException {
        try {
            if (session != null) session.close();
            if (loadedModel != null) loadedModel.close();
        } catch (Exception e) {
            throw new InferenceException("Failed to close model", e);
        } finally {
            super.close();
        }
    }
}
