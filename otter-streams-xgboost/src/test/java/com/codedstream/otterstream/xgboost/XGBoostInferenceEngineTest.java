package com.codedstream.otterstream.xgboost;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for XGBoostInferenceEngine.
 *
 * <p>Tests cover:
 * <ul>
 *   <li>Model initialization and configuration</li>
 *   <li>Single inference operations</li>
 *   <li>Batch inference operations</li>
 *   <li>Feature extraction and validation</li>
 *   <li>Error handling and edge cases</li>
 *   <li>Resource management and cleanup</li>
 *   <li>Engine capabilities</li>
 * </ul>
 *
 * @author Nestor Martourez
 * @since 1.0.0
 */
class XGBoostInferenceEngineTest {

    private XGBoostInferenceEngine engine;
    private ModelConfig testModelConfig;
    private static String testModelPath;
    private static boolean modelCreated = false;

    @TempDir
    static Path tempDir;

    /**
     * Creates a simple XGBoost model for testing purposes.
     * This creates a binary classification model with 4 features.
     */
    @BeforeAll
    static void setupTestModel() throws Exception {
        // Create training data for a simple binary classification model
        float[] labels = new float[]{0, 0, 1, 1, 0, 1};
        float[] data = new float[]{
                // feature1, feature2, feature3, feature4
                1.0f, 2.0f, 3.0f, 4.0f,    // sample 0
                1.5f, 2.5f, 3.5f, 4.5f,    // sample 1
                5.0f, 6.0f, 7.0f, 8.0f,    // sample 2
                5.5f, 6.5f, 7.5f, 8.5f,    // sample 3
                2.0f, 3.0f, 4.0f, 5.0f,    // sample 4
                6.0f, 7.0f, 8.0f, 9.0f     // sample 5
        };

        DMatrix trainMatrix = new DMatrix(data, 6, 4);
        trainMatrix.setLabel(labels);

        // Train a simple model
        Map<String, Object> params = new HashMap<>();
        params.put("max_depth", 2);
        params.put("eta", 0.3);
        params.put("objective", "binary:logistic");
        params.put("eval_metric", "logloss");

        Map<String, DMatrix> watches = new HashMap<>();
        watches.put("train", trainMatrix);

        Booster booster = XGBoost.train(trainMatrix, params, 10, watches, null, null);

        // Save model to temp directory with .json extension to avoid warnings
        testModelPath = tempDir.resolve("test_model.json").toString();
        booster.saveModel(testModelPath);

        // Cleanup
        booster.dispose();
        trainMatrix.dispose();

        modelCreated = true;
    }

    @BeforeEach
    void setUp() {
        engine = new XGBoostInferenceEngine();

        // Create model config for each test
        testModelConfig = ModelConfig.builder()
                .modelId("test-xgboost-model")
                .modelPath(testModelPath)
                .format(ModelFormat.XGBOOST_JSON)  // Use JSON format
                .modelVersion("1.0.0")
                .build();
    }

    @AfterEach
    void tearDown() throws InferenceException {
        if (engine != null && engine.isReady()) {
            engine.close();
        }
    }

    // ========================================================================
    // Initialization Tests
    // ========================================================================

    @Test
    @DisplayName("Should initialize engine with valid model config")
    void testInitializeWithValidConfig() throws InferenceException {
        engine.initialize(testModelConfig);

        assertTrue(engine.isReady(), "Engine should be ready after initialization");
        assertEquals(testModelConfig, engine.getModelConfig(),
                "Engine should store the provided model config");
    }

    @Test
    @DisplayName("Should throw exception when initializing with non-existent model")
    void testInitializeWithNonExistentModel() {
        ModelConfig invalidConfig = ModelConfig.builder()
                .modelId("invalid-model")
                .modelPath("/nonexistent/path/model.json")  // Use .json extension
                .format(ModelFormat.XGBOOST_JSON)
                .build();

        InferenceException exception = assertThrows(InferenceException.class,
                () -> engine.initialize(invalidConfig),
                "Should throw InferenceException for non-existent model");

        assertTrue(exception.getMessage().contains("Failed to load XGBoost model"),
                "Exception message should indicate model loading failure");
    }

    @Test
    @DisplayName("Should throw exception when initializing with null config")
    void testInitializeWithNullConfig() {
        InferenceException exception = assertThrows(InferenceException.class,
                () -> engine.initialize(null),
                "Should throw InferenceException for null config");

        assertTrue(exception.getCause() instanceof NullPointerException,
                "Root cause should be NullPointerException");
    }

    @Test
    @DisplayName("Should not be ready before initialization")
    void testNotReadyBeforeInitialization() {
        assertFalse(engine.isReady(), "Engine should not be ready before initialization");
    }

    // ========================================================================
    // Single Inference Tests
    // ========================================================================

    @Test
    @DisplayName("Should perform single inference successfully")
    void testSingleInference() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", 1.0f);
        inputs.put("feature2", 2.0f);
        inputs.put("feature3", 3.0f);
        inputs.put("feature4", 4.0f);

        InferenceResult result = engine.infer(inputs);

        assertNotNull(result, "Result should not be null");
        assertTrue(result.isSuccess(), "Inference should be successful");
        assertNotNull(result.getOutputs(), "Outputs should not be null");
        assertTrue(result.getOutputs().containsKey("prediction"),
                "Result should contain 'prediction' key");
    }

    @Test
    @DisplayName("Should handle regression output format")
    void testRegressionOutputFormat() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);
        InferenceResult result = engine.infer(inputs);

        Object prediction = result.getOutput("prediction");
        assertNotNull(prediction, "Prediction should not be null");
        assertTrue(prediction instanceof Float, "Prediction should be Float type");

        float predValue = (Float) prediction;
        assertTrue(predValue >= 0.0f && predValue <= 1.0f,
                "Binary classification probability should be between 0 and 1");
    }

    @Test
    @DisplayName("Should handle inference with different numeric types")
    void testInferenceWithDifferentNumericTypes() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", 1);        // Integer
        inputs.put("feature2", 2.0);      // Double
        inputs.put("feature3", 3.0f);     // Float
        inputs.put("feature4", 4L);       // Long

        InferenceResult result = engine.infer(inputs);

        assertTrue(result.isSuccess(), "Should handle different numeric types");
        assertNotNull(result.getOutput("prediction"));
    }

    @Test
    @DisplayName("Should handle inference with NaN (missing values)")
    void testInferenceWithNaN() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", Float.NaN);  // Missing value
        inputs.put("feature2", 2.0f);
        inputs.put("feature3", 3.0f);
        inputs.put("feature4", 4.0f);

        InferenceResult result = engine.infer(inputs);

        assertTrue(result.isSuccess(), "Should handle NaN values gracefully");
        assertNotNull(result.getOutput("prediction"));
    }

    @Test
    @DisplayName("Should record inference timing")
    void testInferenceTimingRecorded() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);
        InferenceResult result = engine.infer(inputs);

        assertTrue(result.getInferenceTimeMs() >= 0,
                "Inference time should be non-negative");
        assertEquals(testModelConfig.getModelId(), result.getModelId(),
                "Result should contain correct model ID");
    }

    @Test
    @DisplayName("Should throw exception for invalid input types")
    void testInferenceWithInvalidInputType() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", "invalid_string");  // String not supported
        inputs.put("feature2", 2.0f);
        inputs.put("feature3", 3.0f);
        inputs.put("feature4", 4.0f);

        assertThrows(InferenceException.class, () -> engine.infer(inputs),
                "Should throw exception for unsupported input types");
    }

    @Test
    @DisplayName("Should throw exception when inferring before initialization")
    void testInferBeforeInitialization() {
        Map<String, Object> inputs = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);

        assertThrows(Exception.class, () -> engine.infer(inputs),
                "Should throw exception when inferring before initialization");
    }

    @Test
    @DisplayName("Should handle empty input map")
    void testInferenceWithEmptyInputs() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();

        // XGBoost will create a DMatrix with 0 features, which should work
        // but may produce unexpected results. We test that it doesn't crash.
        assertDoesNotThrow(() -> engine.infer(inputs),
                "Engine should handle empty inputs without crashing");

        // Alternative: If you want to enforce validation, update the engine
    }

    // ========================================================================
    // Batch Inference Tests
    // ========================================================================

    @Test
    @DisplayName("Should perform batch inference successfully")
    void testBatchInference() throws InferenceException {
        engine.initialize(testModelConfig);

        @SuppressWarnings("unchecked")
        Map<String, Object>[] batchInputs = new Map[3];
        batchInputs[0] = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);
        batchInputs[1] = createTestInputs(5.0f, 6.0f, 7.0f, 8.0f);
        batchInputs[2] = createTestInputs(2.0f, 3.0f, 4.0f, 5.0f);

        InferenceResult result = engine.inferBatch(batchInputs);

        assertNotNull(result, "Batch result should not be null");
        assertTrue(result.isSuccess(), "Batch inference should be successful");
        assertTrue(result.getOutputs().containsKey("batch_predictions"),
                "Result should contain 'batch_predictions' key");

        float[][] predictions = (float[][]) result.getOutput("batch_predictions");
        assertNotNull(predictions, "Batch predictions should not be null");
        assertEquals(3, predictions.length, "Should return predictions for all 3 samples");
    }

    @Test
    @DisplayName("Should handle empty batch")
    void testBatchInferenceWithEmptyBatch() throws InferenceException {
        engine.initialize(testModelConfig);

        @SuppressWarnings("unchecked")
        Map<String, Object>[] emptyBatch = new Map[0];

        InferenceResult result = engine.inferBatch(emptyBatch);

        assertNotNull(result, "Result should not be null for empty batch");
        float[][] predictions = (float[][]) result.getOutput("batch_predictions");
        assertEquals(0, predictions.length, "Empty batch should return empty predictions");
    }

    @Test
    @DisplayName("Should handle single-item batch")
    void testBatchInferenceWithSingleItem() throws InferenceException {
        engine.initialize(testModelConfig);

        @SuppressWarnings("unchecked")
        Map<String, Object>[] batchInputs = new Map[1];
        batchInputs[0] = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);

        InferenceResult result = engine.inferBatch(batchInputs);

        float[][] predictions = (float[][]) result.getOutput("batch_predictions");
        assertEquals(1, predictions.length, "Single-item batch should return one prediction");
    }

    @Test
    @DisplayName("Should handle large batch efficiently")
    void testBatchInferenceWithLargeBatch() throws InferenceException {
        engine.initialize(testModelConfig);

        int batchSize = 100;
        @SuppressWarnings("unchecked")
        Map<String, Object>[] batchInputs = new Map[batchSize];
        for (int i = 0; i < batchSize; i++) {
            batchInputs[i] = createTestInputs(
                    (float) i, (float) i + 1, (float) i + 2, (float) i + 3);
        }

        long startTime = System.currentTimeMillis();
        InferenceResult result = engine.inferBatch(batchInputs);
        long duration = System.currentTimeMillis() - startTime;

        float[][] predictions = (float[][]) result.getOutput("batch_predictions");
        assertEquals(batchSize, predictions.length,
                "Should return predictions for all samples");

        // Batch should be reasonably fast (arbitrary threshold)
        assertTrue(duration < 5000,
                "Batch inference should complete in reasonable time");
    }

    @Test
    @DisplayName("Should throw exception for inconsistent batch feature sizes")
    void testBatchInferenceWithInconsistentFeatures() throws InferenceException {
        engine.initialize(testModelConfig);

        @SuppressWarnings("unchecked")
        Map<String, Object>[] batchInputs = new Map[2];
        batchInputs[0] = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);  // 4 features

        Map<String, Object> inconsistentInput = new HashMap<>();
        inconsistentInput.put("feature1", 1.0f);
        inconsistentInput.put("feature2", 2.0f);  // Only 2 features
        batchInputs[1] = inconsistentInput;

        assertThrows(Exception.class, () -> engine.inferBatch(batchInputs),
                "Should throw exception for inconsistent feature sizes");
    }

    // ========================================================================
    // Feature Extraction Tests
    // ========================================================================

    @Test
    @DisplayName("Should extract float array features correctly")
    void testExtractFloatArrayFeature() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", new float[]{1.5f});  // Single-element float array
        inputs.put("feature2", 2.0f);
        inputs.put("feature3", 3.0f);
        inputs.put("feature4", 4.0f);

        InferenceResult result = engine.infer(inputs);

        assertTrue(result.isSuccess(), "Should handle single-element float arrays");
    }

    @Test
    @DisplayName("Should reject multi-value float arrays")
    void testRejectMultiValueFloatArray() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", new float[]{1.0f, 2.0f});  // Multi-element array
        inputs.put("feature2", 2.0f);
        inputs.put("feature3", 3.0f);
        inputs.put("feature4", 4.0f);

        assertThrows(InferenceException.class, () -> engine.infer(inputs),
                "Should throw exception for multi-value float arrays");
    }

    // ========================================================================
    // Engine Capabilities Tests
    // ========================================================================

    @Test
    @DisplayName("Should return correct engine capabilities")
    void testGetCapabilities() throws InferenceException {
        engine.initialize(testModelConfig);

        InferenceEngine.EngineCapabilities capabilities = engine.getCapabilities();

        assertNotNull(capabilities, "Capabilities should not be null");
        assertTrue(capabilities.supportsBatching(),
                "XGBoost engine should support batching");
        assertEquals(1000, capabilities.getMaxBatchSize(),
                "Max batch size should be 1000");
        assertTrue(capabilities.supportsStreaming(),
                "XGBoost engine should support streaming");
    }

    // ========================================================================
    // Resource Management Tests
    // ========================================================================

    @Test
    @DisplayName("Should close engine and release resources")
    void testCloseEngine() throws InferenceException {
        engine.initialize(testModelConfig);
        assertTrue(engine.isReady(), "Engine should be ready after initialization");

        engine.close();

        // After closing, engine should not be ready
        assertFalse(engine.isReady(), "Engine should not be ready after closing");
    }

    @Test
    @DisplayName("Should handle close on uninitialized engine")
    void testCloseUninitializedEngine() {
        assertDoesNotThrow(() -> engine.close(),
                "Closing uninitialized engine should not throw exception");
    }

    @Test
    @DisplayName("Should handle multiple close calls")
    void testMultipleCloseCalls() throws InferenceException {
        engine.initialize(testModelConfig);

        assertDoesNotThrow(() -> {
            engine.close();
            engine.close();  // Second close
        }, "Multiple close calls should not throw exception");
    }

    @Test
    @DisplayName("Should not allow inference after close")
    void testInferenceAfterClose() throws InferenceException {
        engine.initialize(testModelConfig);
        engine.close();

        Map<String, Object> inputs = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);

        assertThrows(Exception.class, () -> engine.infer(inputs),
                "Should not allow inference after engine is closed");
    }

    // ========================================================================
    // Metadata Tests
    // ========================================================================

    @Test
    @DisplayName("Should return null metadata (not yet implemented)")
    void testGetMetadata() throws InferenceException {
        engine.initialize(testModelConfig);

        assertNull(engine.getMetadata(),
                "Metadata should be null as it's not yet implemented");
    }

    // ========================================================================
    // Edge Cases and Error Handling Tests
    // ========================================================================

    @Test
    @DisplayName("Should handle concurrent inference calls")
    void testConcurrentInference() throws Exception {
        engine.initialize(testModelConfig);

        int numThreads = 10;
        Thread[] threads = new Thread[numThreads];
        boolean[] results = new boolean[numThreads];

        for (int i = 0; i < numThreads; i++) {
            final int index = i;
            threads[i] = new Thread(() -> {
                try {
                    Map<String, Object> inputs = createTestInputs(
                            (float) index, (float) index + 1,
                            (float) index + 2, (float) index + 3);
                    InferenceResult result = engine.infer(inputs);
                    results[index] = result.isSuccess();
                } catch (Exception e) {
                    results[index] = false;
                }
            });
            threads[i].start();
        }

        // Wait for all threads to complete
        for (Thread thread : threads) {
            thread.join();
        }

        // Check all inferences succeeded
        for (boolean result : results) {
            assertTrue(result, "All concurrent inferences should succeed");
        }
    }

    @Test
    @DisplayName("Should handle extreme feature values")
    void testExtremeFeatureValues() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", Float.MAX_VALUE);
        inputs.put("feature2", Float.MIN_VALUE);
        inputs.put("feature3", 0.0f);
        inputs.put("feature4", -1000000.0f);

        assertDoesNotThrow(() -> engine.infer(inputs),
                "Should handle extreme feature values");
    }

    @Test
    @DisplayName("Should handle all NaN inputs")
    void testAllNaNInputs() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", Float.NaN);
        inputs.put("feature2", Float.NaN);
        inputs.put("feature3", Float.NaN);
        inputs.put("feature4", Float.NaN);

        InferenceResult result = engine.infer(inputs);

        assertTrue(result.isSuccess(), "Should handle all NaN inputs");
        assertNotNull(result.getOutput("prediction"));
    }

    // ========================================================================
    // Performance Tests
    // ========================================================================

    @Test
    @DisplayName("Should perform single inference within reasonable time")
    void testSingleInferencePerformance() throws InferenceException {
        engine.initialize(testModelConfig);

        Map<String, Object> inputs = createTestInputs(1.0f, 2.0f, 3.0f, 4.0f);

        long startTime = System.currentTimeMillis();
        InferenceResult result = engine.infer(inputs);
        long duration = System.currentTimeMillis() - startTime;

        assertTrue(result.isSuccess(), "Inference should succeed");
        assertTrue(duration < 1000,
                "Single inference should complete within 1 second");
    }

    @Test
    @DisplayName("Should show batch inference is faster than sequential")
    void testBatchVsSequentialPerformance() throws InferenceException {
        engine.initialize(testModelConfig);

        int batchSize = 50;
        @SuppressWarnings("unchecked")
        Map<String, Object>[] batchInputs = new Map[batchSize];
        for (int i = 0; i < batchSize; i++) {
            batchInputs[i] = createTestInputs(
                    (float) i, (float) i + 1, (float) i + 2, (float) i + 3);
        }

        // Sequential inference
        long seqStart = System.currentTimeMillis();
        for (Map<String, Object> input : batchInputs) {
            engine.infer(input);
        }
        long seqDuration = System.currentTimeMillis() - seqStart;

        // Batch inference
        long batchStart = System.currentTimeMillis();
        engine.inferBatch(batchInputs);
        long batchDuration = System.currentTimeMillis() - batchStart;

        assertTrue(batchDuration < seqDuration,
                "Batch inference should be faster than sequential inference");
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Creates test inputs with specified feature values.
     */
    private Map<String, Object> createTestInputs(float f1, float f2, float f3, float f4) {
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("feature1", f1);
        inputs.put("feature2", f2);
        inputs.put("feature3", f3);
        inputs.put("feature4", f4);
        return inputs;
    }
}