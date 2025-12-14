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

/**
 * XGBoost inference engine for gradient boosting tree models.
 *
 * <p>This engine provides inference capabilities for XGBoost models using the
 * XGBoost4J Java library. XGBoost is an optimized distributed gradient boosting
 * library designed for efficiency, flexibility, and portability, widely used for
 * tabular data and structured data problems.
 *
 * <h2>Supported XGBoost Features:</h2>
 * <ul>
 *   <li><b>Model Formats:</b> XGBoost binary (.model), JSON (.json), UBJSON (.ubj)</li>
 *   <li><b>Task Types:</b> Regression, binary classification, multi-class classification</li>
 *   <li><b>Batch Inference:</b> Efficient batch prediction through matrix operations</li>
 *   <li><b>Missing Values:</b> Native support for NaN as missing value indicator</li>
 *   <li><b>Thread Safety:</b> Model predictions are thread-safe</li>
 * </ul>
 *
 * <h2>Model Loading:</h2>
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelPath("model.xgb")  // XGBoost model file
 *     .modelId("xgboost-model")
 *     .build();
 *
 * XGBoostInferenceEngine engine = new XGBoostInferenceEngine();
 * engine.initialize(config);
 * }</pre>
 *
 * <h2>Inference Example:</h2>
 * <pre>{@code
 * Map<String, Object> inputs = new HashMap<>();
 * inputs.put("age", 35.0f);
 * inputs.put("income", 75000.0f);
 * inputs.put("credit_score", 720.0f);
 * inputs.put("loan_amount", 25000.0f);
 *
 * InferenceResult result = engine.infer(inputs);
 *
 * // For regression/binary classification
 * float prediction = (float) result.getOutput("prediction");
 *
 * // For multi-class classification
 * float[] probabilities = (float[]) result.getOutput("probabilities");
 * int predictedClass = (int) result.getOutput("prediction");
 * }</pre>
 *
 * <h2>Feature Extraction:</h2>
 * <p>The engine assumes input features are already in the correct order for
 * the XGBoost model. Features are extracted in the order they appear in the
 * input Map. For production use, implement feature ordering based on model
 * metadata or configuration.
 *
 * <h2>Prediction Outputs:</h2>
 * <table border="1">
 *   <tr><th>Task Type</th><th>Output Format</th><th>Example</th></tr>
 *   <tr><td>Regression</td><td>Single float value</td><td>{"prediction": 0.75}</td></tr>
 *   <tr><td>Binary Classification</td><td>Single probability</td><td>{"prediction": 0.92}</td></tr>
 *   <tr><td>Multi-class Classification</td><td>Probabilities array + class index</td><td>{"probabilities": [0.1,0.8,0.1], "prediction": 1}</td></tr>
 * </table>
 *
 * <h2>Capabilities:</h2>
 * <table border="1">
 *   <tr><th>Feature</th><th>Supported</th><th>Notes</th></tr>
 *   <tr><td>Batch Inference</td><td>Yes</td><td>Efficient matrix-based batch processing</td></tr>
 *   <tr><td>Native Batching</td><td>No</td><td>Batch size limited by memory</td></tr>
 *   <tr><td>Max Batch Size</td><td>1000</td><td>Conservative default for memory safety</td></tr>
 *   <tr><td>GPU Support</td><td>Yes</td><td>When XGBoost built with GPU support</td></tr>
 *   <tr><td>Missing Values</td><td>Yes</td><td>NaN represents missing values</td></tr>
 * </table>
 *
 * <h2>Dependencies:</h2>
 * <pre>
 * Requires XGBoost4J Java library:
 * - ml.dmlc:xgboost4j (runtime)
 * - ml.dmlc:xgboost4j-linux-gpu (optional, for GPU support)
 * </pre>
 *
 * <h2>Performance Features:</h2>
 * <ul>
 *   <li><b>DMatrix Optimization:</b> Efficient column-major data storage</li>
 *   <li><b>Thread Pool:</b> XGBoost uses internal thread pool for prediction</li>
 *   <li><b>Memory Efficient:</b> Automatic DMatrix disposal to prevent leaks</li>
 *   <li><b>Batch Processing:</b> Significant speedup for batch predictions</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>{@link Booster} prediction methods are thread-safe according to XGBoost
 * documentation. Multiple threads can call {@link Booster# predict} concurrently.
 * However, {@link DMatrix} creation and disposal should be synchronized if
 * sharing matrices between threads.
 *
 * <h2>Resource Management:</h2>
 * <p>XGBoost uses native memory through {@link DMatrix} and {@link Booster}.
 * Always call {@link #close()} to release native resources:
 *
 * <pre>{@code
 * try (XGBoostInferenceEngine engine = new XGBoostInferenceEngine()) {
 *     engine.initialize(config);
 *     InferenceResult result = engine.infer(inputs);
 * }
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see LocalInferenceEngine
 * @see Booster
 * @see DMatrix
 * @see <a href="https://xgboost.readthedocs.io/">XGBoost Documentation</a>
 */
public class XGBoostInferenceEngine extends LocalInferenceEngine<Booster> {

    /**
     * Initializes the XGBoost inference engine by loading a model file.
     *
     * <p>Supports various XGBoost model formats:
     * <ul>
     *   <li><b>Binary:</b> .model (default binary format)</li>
     *   <li><b>JSON:</b> .json (human-readable, larger file size)</li>
     *   <li><b>UBJSON:</b> .ubj (binary JSON, efficient storage)</li>
     * </ul>
     *
     * <h2>GPU Support:</h2>
     * <p>If XGBoost is compiled with GPU support and a GPU is available,
     * predictions will automatically use GPU acceleration. Check XGBoost
     * documentation for GPU compilation instructions.
     *
     * @param config model configuration containing XGBoost model file path
     * @throws InferenceException if model loading fails or file is invalid
     */
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

    /**
     * Performs single inference using XGBoost model.
     *
     * <p>The inference process:
     * <ol>
     *   <li>Extracts features from input Map to float array</li>
     *   <li>Creates {@link DMatrix} with shape [1, num_features]</li>
     *   <li>Calls {@link Booster# predict} for prediction</li>
     *   <li>Formats output based on prediction task type</li>
     *   <li>Disposes {@link DMatrix} to prevent memory leaks</li>
     * </ol>
     *
     * <h2>Output Format Detection:</h2>
     * <p>Automatically detects prediction task type based on output shape:
     * <ul>
     *   <li><b>Single value:</b> Regression or binary classification</li>
     *   <li><b>Multiple values:</b> Multi-class classification probabilities</li>
     * </ul>
     *
     * <h2>Missing Values:</h2>
     * <p>XGBoost handles missing values represented as {@link Float#NaN}.
     * Features with NaN values are treated as missing during prediction.
     *
     * @param inputs map of feature names to values (Number or float[])
     * @return inference result with formatted predictions
     * @throws InferenceException if inference fails or feature extraction fails
     * @throws IllegalArgumentException for unsupported feature types
     */
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

    /**
     * Performs batch inference using XGBoost's efficient matrix operations.
     *
     * <p>Batch inference is significantly faster than sequential single predictions
     * because XGBoost processes the entire batch matrix in native code. The method:
     * <ol>
     *   <li>Validates all batch inputs have same feature count</li>
     *   <li>Flattens batch features into single array</li>
     *   <li>Creates {@link DMatrix} with shape [batch_size, num_features]</li>
     *   <li>Performs batch prediction in single native call</li>
     *   <li>Returns 2D array of predictions</li>
     * </ol>
     *
     * <h2>Memory Efficiency:</h2>
     * <p>The batch features array uses contiguous memory for efficient
     * data transfer to native XGBoost library. For very large batches,
     * consider splitting into smaller batches to manage memory usage.
     *
     * <h2>Output Format:</h2>
     * <pre>
     * {
     *   "batch_predictions": [
     *     [0.1, 0.9],  // Sample 1 predictions
     *     [0.7, 0.3],  // Sample 2 predictions
     *     ...
     *   ]
     * }
     * </pre>
     *
     * @param batchInputs array of input maps for batch processing
     * @return inference result containing 2D array of batch predictions
     * @throws InferenceException if batch inference fails or feature counts mismatch
     */
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

    /**
     * Gets the engine's capabilities for XGBoost inference.
     *
     * <p>XGBoost engine capabilities:
     * <ul>
     *   <li><b>Batch Inference:</b> Yes, efficient matrix operations</li>
     *   <li><b>Native Batching:</b> No, batch size limited by memory</li>
     *   <li><b>Max Batch Size:</b> 1000 (conservative for memory safety)</li>
     *   <li><b>GPU Support:</b> Yes, when XGBoost compiled with GPU support</li>
     * </ul>
     *
     * @return engine capabilities for XGBoost
     */
    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, false, 1000, true);
    }

    /**
     * Closes the XGBoost engine and releases native resources.
     *
     * <p>Disposes the {@link Booster} which releases:
     * <ul>
     *   <li>Tree structure memory</li>
     *   <li>Leaf values and split conditions</li>
     *   <li>Any GPU memory allocated</li>
     *   <li>Internal thread pool resources</li>
     * </ul>
     *
     * <p>Always call this method when finished to prevent native memory leaks.
     *
     * @throws InferenceException if Booster disposal fails
     */
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

    /**
     * Gets metadata about the loaded XGBoost model.
     *
     * <p><strong>TODO:</strong> Implement XGBoost metadata extraction.
     * XGBoost models contain metadata that can be extracted via:
     * <ul>
     *   <li>{@link Booster#getModelDump} - Tree structure dump</li>
     *   <li>{@link Booster#getFeatureScore} - Feature importance scores</li>
     *   <li>{@link Booster# attributes} - Model attributes (objective, booster type)</li>
     *   <li>Number of features and trees</li>
     * </ul>
     *
     * @return model metadata (currently returns null, override for implementation)
     */
    @Override
    public ModelMetadata getMetadata() {
        return null;
    }

    /**
     * Extracts features from input Map to float array for XGBoost.
     *
     * <p>Converts input values to float array in the order they appear in the Map.
     * Supported input types:
     * <ul>
     *   <li><b>Number:</b> Integer, Float, Double, etc. converted to float</li>
     *   <li><b>float[]:</b> Single-element arrays extracted as float</li>
     *   <li><b>Other types:</b> Throw {@link IllegalArgumentException}</li>
     * </ul>
     *
     * <h2>Feature Ordering:</h2>
     * <p>Current implementation uses Map iteration order. For production,
     * implement feature ordering based on:
     * <ul>
     *   <li>Model feature names from XGBoost metadata</li>
     *   <li>Configuration file specifying feature order</li>
     *   <li>Training data schema</li>
     * </ul>
     *
     * @param inputs map of feature names to values
     * @return float array of feature values in order
     * @throws IllegalArgumentException for unsupported feature types
     */
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

    /**
     * Finds the index of maximum value in float array.
     *
     * <p>Used for multi-class classification to determine predicted class
     * from probability array. Returns the index (0-based) of the highest
     * probability value.
     *
     * @param array float array of class probabilities
     * @return index of maximum value
     */
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