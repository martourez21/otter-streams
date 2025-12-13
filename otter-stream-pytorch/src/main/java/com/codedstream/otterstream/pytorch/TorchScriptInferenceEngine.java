package com.codedstream.otterstream.pytorch;


import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;

import java.util.Map;
import java.util.HashMap;

public class TorchScriptInferenceEngine extends LocalInferenceEngine<ZooModel<Map<String, Object>, Map<String, Object>>> {

    private Predictor<Map<String, Object>, Map<String, Object>> predictor;
    private NDManager ndManager;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;
            this.ndManager = NDManager.newBaseManager();

            Criteria<Map<String, Object>, Map<String, Object>> criteria =
                    Criteria.<Map<String, Object>, Map<String, Object>>builder()
                            .setTypes(
                                    (Class<Map<String, Object>>) (Class<?>) Map.class,
                                    (Class<Map<String, Object>>) (Class<?>) Map.class
                            )
                            .optModelPath(java.nio.file.Paths.get(config.getModelPath()))
                            .optTranslator(new MapTranslator())
                            .build();

            this.loadedModel = criteria.loadModel();
            this.predictor = loadedModel.newPredictor();
            this.initialized = true;

        } catch (Exception e) {
            throw new InferenceException("Failed to load PyTorch model", e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try {
            long startTime = System.currentTimeMillis();
            Map<String, Object> outputs = predictor.predict(inputs);
            long endTime = System.currentTimeMillis();

            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("PyTorch inference failed", e);
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        return null;
    }

    @Override
    public void close() throws InferenceException {
        if (predictor != null) {
            predictor.close();
        }
        if (loadedModel != null) {
            loadedModel.close();
        }
        if (ndManager != null) {
            ndManager.close();
        }
        super.close();
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, true, 64, true);
    }

    private static class MapTranslator implements Translator<Map<String, Object>, Map<String, Object>> {

        @Override
        public NDList processInput(TranslatorContext ctx, Map<String, Object> input) {
            NDManager manager = ctx.getNDManager();
            NDList list = new NDList();

            for (Object value : input.values()) {
                if (value instanceof float[]) {
                    float[] array = (float[]) value;
                    list.add(manager.create(array).expandDims(0));
                } else if (value instanceof int[]) {
                    int[] array = (int[]) value;
                    list.add(manager.create(array).expandDims(0));
                }
            }

            return list;
        }

        @Override
        public Map<String, Object> processOutput(TranslatorContext ctx, NDList list) {
            Map<String, Object> output = new HashMap<>();
            for (int i = 0; i < list.size(); i++) {
                output.put("output_" + i, list.get(i).toFloatArray());
            }
            return output;
        }
    }
}