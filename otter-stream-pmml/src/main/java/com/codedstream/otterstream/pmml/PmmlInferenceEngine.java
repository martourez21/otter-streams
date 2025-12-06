package com.codedstream.otterstream.pmml;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;
import org.dmg.pmml.PMML;


import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Map;
import java.util.HashMap;
import java.util.List;

public class PmmlInferenceEngine extends LocalInferenceEngine<Evaluator> {

    private Evaluator evaluator;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;

            File pmmlFile = new File(config.getModelPath());
            try (InputStream is = new FileInputStream(pmmlFile)) {

                PMML pmml = PMMLUtil.unmarshal(is);

                ModelEvaluatorFactory factory = ModelEvaluatorFactory.newInstance();

                // This method exists ONLY if pmml-evaluator is included
                this.evaluator = factory.newModelEvaluator(pmml);

                this.evaluator.verify();
                this.initialized = true;
            }
        } catch (Exception e) {
            throw new InferenceException(
                    "Failed to load PMML model from: " + config.getModelPath(), e
            );
        }
    }



    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try {
            long startTime = System.currentTimeMillis();

            Map<FieldName, FieldValue> arguments = new HashMap<>();

            for (InputField inputField : evaluator.getInputFields()) {
                FieldName name = inputField.getName();

                if (inputs.containsKey(name.getValue())) {
                    Object raw = inputs.get(name.getValue());
                    FieldValue prepared = inputField.prepare(raw);
                    arguments.put(name, prepared);
                }w
            }

            Map<FieldName, ?> results = evaluator.evaluate(arguments);

            Map<String, Object> outputs = new HashMap<>();

            // Output fields
            for (OutputField outputField : evaluator.getOutputFields()) {
                FieldName name = outputField.getName();
                Object value = results.get(name);

                if (value instanceof Computable) {
                    value = ((Computable) value).getResult();
                }

                outputs.put(name.getValue(), value);
            }

            // Target fields
            for (TargetField targetField : evaluator.getTargetFields()) {
                FieldName name = targetField.getName();
                Object value = results.get(name);

                if (value instanceof Computable) {
                    value = ((Computable) value).getResult();
                }

                outputs.put(name.getValue(), value);
            }

            long endTime = System.currentTimeMillis();

            return new InferenceResult(
                    outputs,
                    endTime - startTime,
                    modelConfig.getModelId()
            );

        } catch (Exception e) {
            throw new InferenceException("PMML inference failed", e);
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        Map<String, Object> batchOutputs = new HashMap<>();
        long totalTime = 0;

        for (int i = 0; i < batchInputs.length; i++) {
            InferenceResult result = infer(batchInputs[i]);
            totalTime += result.getInferenceTimeMs();

            int finalI = i;
            result.getOutputs().forEach((k, v) -> {
                batchOutputs.put(k + "_" + finalI, v);
            });
        }

        return new InferenceResult(batchOutputs, totalTime, modelConfig.getModelId());
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(false, false, 1, true);
    }

    @Override
    public void close() throws InferenceException {
        this.evaluator = null;
        super.close();
    }
}
