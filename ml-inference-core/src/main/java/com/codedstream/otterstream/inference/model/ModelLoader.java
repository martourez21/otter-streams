package com.codedstream.otterstream.inference.model;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.ModelLoadException;
import java.io.InputStream;
import java.util.Map;

/**
 * Interface for loading ML models from various sources.
 *
 * <p>Implementations handle the specifics of loading different model formats
 * (ONNX, TensorFlow, PyTorch, etc.) from various storage systems (file system,
 * HDFS, S3, HTTP, etc.).
 *
 * <h2>Responsibilities:</h2>
 * <ul>
 *   <li>Load models from configured paths</li>
 *   <li>Validate model format and integrity</li>
 *   <li>Extract model metadata</li>
 *   <li>Handle different storage backends</li>
 * </ul>
 *
 * <h2>Implementation Example:</h2>
 * <pre>{@code
 * public class OnnxModelLoader implements ModelLoader<OrtSession> {
 *     @Override
 *     public OrtSession loadModel(ModelConfig config) throws ModelLoadException {
 *         try {
 *             OrtEnvironment env = OrtEnvironment.getEnvironment();
 *             return env.createSession(config.getModelPath());
 *         } catch (Exception e) {
 *             throw new ModelLoadException("Failed to load ONNX model", e);
 *         }
 *     }
 *
 *     @Override
 *     public boolean validateModel(OrtSession model, ModelConfig config) {
 *         return model != null && model.getNumInputs() > 0;
 *     }
 *
 *     @Override
 *     public ModelFormat[] getSupportedFormats() {
 *         return new ModelFormat[]{ModelFormat.ONNX};
 *     }
 *
 *     @Override
 *     public ModelMetadata getModelMetadata(OrtSession model) {
 *         // Extract metadata from loaded model
 *     }
 * }
 * }</pre>
 *
 * @param <T> the type of loaded model (e.g., OrtSession, SavedModelBundle)
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see ModelConfig
 * @see ModelMetadata
 */
public interface ModelLoader<T> {

    /**
     * Loads a model from the path specified in configuration.
     *
     * @param config model configuration containing path and options
     * @return loaded model instance
     * @throws ModelLoadException if loading fails
     */
    T loadModel(ModelConfig config) throws ModelLoadException;

    /**
     * Loads a model from an input stream.
     * <p>Useful for loading models from non-file sources.
     *
     * @param inputStream stream containing model data
     * @param config model configuration
     * @return loaded model instance
     * @throws ModelLoadException if loading fails
     */
    T loadModel(InputStream inputStream, ModelConfig config) throws ModelLoadException;

    /**
     * Validates that a loaded model matches the configuration.
     *
     * @param model the loaded model
     * @param config model configuration
     * @return true if model is valid
     */
    boolean validateModel(T model, ModelConfig config);

    /**
     * Gets the model formats supported by this loader.
     *
     * @return array of supported formats
     */
    ModelFormat[] getSupportedFormats();

    /**
     * Extracts metadata from a loaded model.
     *
     * @param model the loaded model
     * @return model metadata including inputs, outputs, and format
     */
    ModelMetadata getModelMetadata(T model);
}