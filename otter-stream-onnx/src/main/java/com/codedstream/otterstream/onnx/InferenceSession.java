package com.codedstream.otterstream.onnx;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Map;

/**
 * Wrapper class for ONNX Runtime sessions providing simplified access to inference capabilities.
 *
 * <p>This class encapsulates the ONNX Runtime {@link OrtSession} and {@link OrtEnvironment}
 * to provide a cleaner API for loading and managing ONNX models. It supports loading models
 * from both file paths and byte arrays, and provides access to model metadata.
 *
 * <h2>Key Responsibilities:</h2>
 * <ul>
 *   <li>Manage ONNX Runtime environment and session lifecycle</li>
 *   <li>Provide access to model input/output metadata</li>
 *   <li>Handle resource cleanup through {@link #close()} method</li>
 *   <li>Support multiple model loading strategies</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Load from file
 * InferenceSession session = new InferenceSession(
 *     "model.onnx",
 *     new OrtSession.SessionOptions(),
 *     OrtEnvironment.getEnvironment()
 * );
 *
 * // Get input metadata
 * Map<String, NodeInfo> inputs = session.getInputMetadata();
 *
 * // Use session for inference
 * OrtSession ortSession = session.getSession();
 *
 * // Clean up
 * session.close();
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>ONNX Runtime sessions are not thread-safe for concurrent inference calls.
 * For multi-threaded scenarios, create separate {@code InferenceSession} instances
 * or synchronize access to the {@link #getSession()} method.
 *
 * <h2>Resource Management:</h2>
 * <p>Always call {@link #close()} when finished with the session to release
 * native resources. Consider using try-with-resources pattern:
 *
 * <pre>{@code
 * try (InferenceSession session = new InferenceSession(...)) {
 *     // Use session
 * }
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see OrtSession
 * @see OrtEnvironment
 * @see OnnxInferenceEngine
 */
public class InferenceSession {

    private final OrtEnvironment environment;
    private final OrtSession session;

    /**
     * Loads an ONNX model from a file path.
     *
     * @param modelPath path to the ONNX model file
     * @param options session configuration options
     * @param environment ONNX Runtime environment
     * @throws Exception if model loading fails
     */
    public InferenceSession(String modelPath, OrtSession.SessionOptions options, OrtEnvironment environment) throws Exception {
        this.environment = environment;
        this.session = environment.createSession(modelPath, options);
    }

    /**
     * Loads an ONNX model from a byte array.
     * <p>Useful for loading models from memory or network streams.
     *
     * @param modelBytes byte array containing the ONNX model
     * @param options session configuration options
     * @param environment ONNX Runtime environment
     * @throws Exception if model loading fails
     */
    public InferenceSession(byte[] modelBytes, OrtSession.SessionOptions options, OrtEnvironment environment) throws Exception {
        this.environment = environment;
        this.session = environment.createSession(modelBytes, options);
    }

    /**
     * Gets metadata about model inputs.
     *
     * @return map of input names to {@link NodeInfo} describing input tensors
     * @throws OrtException if metadata retrieval fails
     */
    public Map<String, NodeInfo> getInputMetadata() throws OrtException {
        return session.getInputInfo();
    }

    /**
     * Gets metadata about model outputs.
     *
     * @return map of output names to {@link NodeInfo} describing output tensors
     * @throws OrtException if metadata retrieval fails
     */
    public Map<String, NodeInfo> getOutputMetadata() throws OrtException {
        return session.getOutputInfo();
    }

    /**
     * Gets the underlying ONNX Runtime session.
     * <p>Provides direct access to ONNX Runtime API for advanced use cases.
     *
     * @return the {@link OrtSession} instance
     */
    public OrtSession getSession() {
        return session;
    }

    /**
     * Closes the session and releases native resources.
     * <p>This method is idempotent and can be called multiple times.
     * Always call this method when finished with the session to prevent
     * native memory leaks.
     */
    public void close() {
        try {
            session.close();
            environment.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}