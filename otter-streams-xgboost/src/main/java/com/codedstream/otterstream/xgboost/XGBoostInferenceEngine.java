package com.codedstream.otterstream.xgboost;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.Map;
import java.util.HashMap;

public class XGBoostInferenceEngine extends LocalInferenceEngine<Booster> {

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;
            this.loadedModel = XGBoost.loadModel(config.getModelPath());
            this.initialized = true;
        } catch (XGBoostError e) {
            throw new InferenceException("Failed to load XGBoost model from: " + config.getModelPath(), e);
        } catch (Exception e) {
            throw new InferenceException("Unexpected error loading XGBoost model", e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        DMatrix data = null;
        try {
            long startTime = System.currentTimeMillis();

            // Convert inputs to feature array
            float[] features = extractFeatures(inputs);

            // Create DMatrix for single prediction
            data = new DMatrix(features, 1, features.length, Float.NaN);

            // Perform prediction
            float[][] predictions = loadedModel.predict(data);

            // Extract results
            Map<String, Object> outputs = new HashMap<>();

            if (predictions.length > 0 && predictions[0].length == 1) {
                // Single output (regression or binary classification)
                outputs.put("prediction", predictions[0][0]);
            } else if (predictions.length > 0 && predictions[0].length > 1) {
                // Multi-class classification
                outputs.put("probabilities", predictions[0]);
                outputs.put("prediction", getMaxIndex(predictions[0]));
            } else {
                outputs.put("predictions", predictions[0]);
            }

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());

        } catch (Exception e) {
            throw new InferenceException("XGBoost inference failed", e);
        } finally {
            if (data != null) {
                try {
                    data.dispose();
                } catch (Exception e) {
                    // Log disposal error but don't fail inference
                }
            }
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        DMatrix data = null;
        try {
            long startTime = System.currentTimeMillis();

            int batchSize = batchInputs.length;
            if (batchSize == 0) {
                return new InferenceResult(Map.of("batch_predictions", new float[0][]), 0, modelConfig.getModelId());
            }

            int featureSize = extractFeatures(batchInputs[0]).length;
            float[] batchFeatures = new float[batchSize * featureSize];

            // Flatten batch inputs
            for (int i = 0; i < batchSize; i++) {
                float[] features = extractFeatures(batchInputs[i]);
                System.arraycopy(features, 0, batchFeatures, i * featureSize, featureSize);
            }

            // Create DMatrix for batch prediction
            data = new DMatrix(batchFeatures, batchSize, featureSize, Float.NaN);

            // Perform batch prediction
            float[][] predictions = loadedModel.predict(data);

            Map<String, Object> outputs = new HashMap<>();
            outputs.put("batch_predictions", predictions);

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());

        } catch (Exception e) {
            throw new InferenceException("XGBoost batch inference failed", e);
        } finally {
            if (data != null) {
                try {
                    data.dispose();
                } catch (Exception e) {
                    // Log disposal error
                }
            }
        }
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, false, 1000, true);
    }

    @Override
    public void close() throws InferenceException {
        if (loadedModel != null) {
            try {
                loadedModel.dispose();
            } catch (Exception e) {
                throw new InferenceException("Failed to dispose XGBoost model", e);
            }
        }
        super.close();
    }

    @Override
    public ModelMetadata getMetadata() {
        return null;
    }

    private float[] extractFeatures(Map<String, Object> inputs) {
        // Extract and order features properly for XGBoost
        // This implementation assumes inputs are already in correct order
        // In production, you might want to sort by feature names

        float[] features = new float[inputs.size()];
        int i = 0;
        for (Object value : inputs.values()) {
            if (value instanceof Number) {
                features[i++] = ((Number) value).floatValue();
            } else if (value instanceof float[]) {
                float[] array = (float[]) value;
                if (array.length == 1) {
                    features[i++] = array[0];
                } else {
                    throw new IllegalArgumentException("Multi-value features not supported in this implementation");
                }
            } else {
                throw new IllegalArgumentException("Unsupported feature type: " + value.getClass());
            }
        }
        return features;
    }

    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}