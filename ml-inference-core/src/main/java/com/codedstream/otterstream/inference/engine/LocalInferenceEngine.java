package com.codedstream.otterstream.inference.engine;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelLoader;
import java.util.Map;

/**
 * Abstract base class for local inference engines that load models from files.
 *
 * <p>Provides common functionality for engines that run models locally on the
 * Flink TaskManager, as opposed to remote inference via API calls.
 *
 * <h2>For Engine Implementers:</h2>
 * <p>Extend this class and implement:
 * <ul>
 *   <li>{@link #infer(Map)} - single inference logic</li>
 *   <li>{@link #inferBatch(Map[])} - batch inference logic</li>
 *   <li>{@link #getCapabilities()} - engine capabilities</li>
 *   <li>{@link #getMetadata()} - model metadata</li>
 * </ul>
 *
 * <p>You can either:
 * <ul>
 *   <li>Provide a {@link ModelLoader} via constructor, OR</li>
 *   <li>Override {@link #loadModelDirectly(ModelConfig)} to handle loading yourself</li>
 * </ul>
 *
 * <h2>Example Implementation:</h2>
 * <pre>{@code
 * public class MyEngine extends LocalInferenceEngine<MyModel> {
 *     @Override
 *     protected void loadModelDirectly(ModelConfig config) {
 *         this.loadedModel = MyModel.load(config.getModelPath());
 *     }
 *
 *     @Override
 *     public InferenceResult infer(Map<String, Object> inputs) {
 *         // Use this.loadedModel to make predictions
 *     }
 * }
 * }</pre>
 *
 * @param <T> the type of the loaded model
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see InferenceEngine
 */
public abstract class LocalInferenceEngine<T> implements InferenceEngine<T> {
    protected ModelConfig modelConfig;
    protected T loadedModel;
    protected ModelLoader<T> modelLoader;
    protected boolean initialized = false;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        this.modelConfig = config;
        try {
            if (modelLoader != null) {
                this.loadedModel = modelLoader.loadModel(config);
            } else {
                loadModelDirectly(config);
            }
            this.initialized = true;
        } catch (Exception e) {
            throw new InferenceException("Failed to initialize inference engine", e);
        }
    }

    /**
     * Override this method for engines that handle their own model loading.
     * <p>Called during {@link #initialize(ModelConfig)} if no ModelLoader is provided.
     *
     * @param config model configuration containing path and options
     * @throws InferenceException if model loading fails
     */
    protected void loadModelDirectly(ModelConfig config) throws InferenceException {
        throw new InferenceException("Model loader not provided and loadModelDirectly not implemented");
    }

    @Override
    public abstract InferenceResult infer(Map<String, Object> inputs) throws InferenceException;

    @Override
    public abstract InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException;

    @Override
    public boolean isReady() {
        return initialized && loadedModel != null;
    }

    @Override
    public ModelConfig getModelConfig() {
        return modelConfig;
    }

    @Override
    public void close() throws InferenceException {
        this.initialized = false;
        this.loadedModel = null;
    }

    @Override
    public abstract EngineCapabilities getCapabilities();
}