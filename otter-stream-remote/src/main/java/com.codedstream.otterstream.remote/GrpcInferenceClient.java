package com.codedstream.otterstream.remote;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * gRPC inference client for TensorFlow Serving, NVIDIA Triton, and other gRPC-based model servers.
 *
 * <p><strong>Note:</strong> This is a template class. Actual gRPC stubs must be generated
 * from protobuf definition files (.proto) for specific model serving frameworks.
 *
 * <h2>Supported Frameworks:</h2>
 * <ul>
 *   <li><b>TensorFlow Serving:</b> tensorflow/serving protobuf definitions</li>
 *   <li><b>NVIDIA Triton:</b> nvidia/triton protobuf definitions</li>
 *   <li><b>Custom gRPC:</b> Any gRPC-based model serving endpoint</li>
 * </ul>
 *
 * <h2>Prerequisites:</h2>
 * <p>Before using this client, generate gRPC Java stubs from your .proto files:
 * <pre>{@code
 * // Example for TensorFlow Serving
 * protoc --java_out=src/main/java \
 *        --grpc-java_out=src/main/java \
 *        tensorflow_serving/apis/*.proto
 * }</pre>
 *
 * <h2>Configuration Example:</h2>
 * <pre>{@code
 * InferenceConfig inferenceConfig = InferenceConfig.builder()
 *     .modelConfig(ModelConfig.builder()
 *         .modelName("tensorflow-model")
 *         .modelVersion("1")
 *         .build())
 *     .engineOption("endpoint", "localhost:8500") // gRPC server address
 *     .timeout(Duration.ofSeconds(30))
 *     .build();
 *
 * GrpcInferenceClient client = new GrpcInferenceClient(inferenceConfig);
 * client.initialize(inferenceConfig.getModelConfig());
 * }</pre>
 *
 * <h2>Security Note:</h2>
 * <p>This template uses plaintext gRPC ({@code .usePlaintext()}). In production:
 * <ul>
 *   <li>Use TLS: {@code .useTransportSecurity()}</li>
 *   <li>Add authentication interceptors</li>
 *   <li>Implement certificate validation</li>
 * </ul>
 *
 * <h2>gRPC Advantages:</h2>
 * <ul>
 *   <li><b>Performance:</b> Binary protocol, HTTP/2, multiplexing</li>
 *   <li><b>Streaming:</b> Supports client/server/bidirectional streaming</li>
 *   <li><b>Code Generation:</b> Type-safe from protobuf definitions</li>
 *   <li><b>Interceptors:</b> Middleware for logging, auth, metrics</li>
 * </ul>
 *
 * <h2>Implementation Steps:</h2>
 * <ol>
 *   <li>Add protobuf definitions to project</li>
 *   <li>Generate Java stubs with protoc</li>
 *   <li>Replace placeholder inference logic with actual gRPC calls</li>
 *   <li>Configure TLS/authentication for production</li>
 *   <li>Add interceptors for observability</li>
 * </ol>
 *
 * <h2>Error Handling:</h2>
 * <p>gRPC uses {@link StatusRuntimeException} for all errors. Status codes include:
 * <ul>
 *   <li><b>DEADLINE_EXCEEDED:</b> Request timeout</li>
 *   <li><b>UNAVAILABLE:</b> Server unavailable</li>
 *   <li><b>INVALID_ARGUMENT:</b> Bad request</li>
 *   <li><b>INTERNAL:</b> Server error</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>{@link ManagedChannel} is thread-safe and should be reused. Generated
 * gRPC stubs are not thread-safe; create separate stubs per thread or use
 * {@code .newStub(channel)} for each request.
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see InferenceEngine
 * @see ManagedChannel
 * @see <a href="https://grpc.io/docs/languages/java/basics/">gRPC Java Tutorial</a>
 * @see <a href="https://www.tensorflow.org/tfx/serving/api_rest">TensorFlow Serving gRPC API</a>
 */
public class GrpcInferenceClient
        implements InferenceEngine<ManagedChannel> {

    private static final Logger LOG =
            LoggerFactory.getLogger(GrpcInferenceClient.class);

    private final InferenceConfig inferenceConfig;

    private ManagedChannel channel;
    private ModelMetadata metadata;
    private boolean initialized = false;

    /**
     * Constructs a new gRPC inference client with the provided configuration.
     *
     * @param inferenceConfig inference configuration containing gRPC endpoint options
     */
    public GrpcInferenceClient(InferenceConfig inferenceConfig) {
        this.inferenceConfig = inferenceConfig;
    }

    // ---------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------

    /**
     * Initializes the gRPC inference client and establishes connection to server.
     *
     * <p>Initialization process:
     * <ol>
     *   <li>Parses endpoint from engine options (format: "host:port")</li>
     *   <li>Creates {@link ManagedChannel} with plaintext connection</li>
     *   <li>Initializes {@link ModelMetadata} with REMOTE_GRPC format</li>
     *   <li>Marks client as initialized</li>
     * </ol>
     *
     * <h2>Security Warning:</h2>
     * <p>This template uses {@code .usePlaintext()} for simplicity. In production:
     * <pre>{@code
     * ManagedChannelBuilder.forAddress(host, port)
     *     .useTransportSecurity()  // Enable TLS
     *     .build();
     * }</pre>
     *
     * <h2>Endpoint Format:</h2>
     * <ul>
     *   <li><b>With port:</b> "localhost:8500"</li>
     *   <li><b>Default port:</b> "localhost" (uses port 8500)</li>
     * </ul>
     *
     * @param modelConfig model configuration containing model name and version
     * @throws InferenceException if initialization fails or endpoint is invalid
     * @throws IllegalArgumentException if endpoint is not configured
     */
    @Override
    public void initialize(ModelConfig modelConfig) throws InferenceException {
        LOG.info("Initializing gRPC inference client");

        try {
            Map<String, Object> options = inferenceConfig.getEngineOptions();
            String endpoint = (String) options.get("endpoint");

            if (endpoint == null) {
                throw new IllegalArgumentException("gRPC endpoint not configured");
            }

            String[] parts = endpoint.split(":");
            String host = parts[0];
            int port = parts.length > 1 ? Integer.parseInt(parts[1]) : 8500;

            this.channel = ManagedChannelBuilder
                    .forAddress(host, port)
                    .usePlaintext() // 🔐 use TLS in production
                    .build();

            this.metadata = new ModelMetadata(
                    modelConfig.getModelName(),
                    modelConfig.getModelVersion(),
                    ModelFormat.REMOTE_GRPC,
                    Map.of(), // input schema (optional)
                    Map.of(), // output schema (optional)
                    -1L,      // model size unknown
                    System.currentTimeMillis()
            );

            this.initialized = true;
            LOG.info("gRPC inference client initialized");

        } catch (Exception e) {
            throw new InferenceException("Failed to initialize gRPC inference client", e);
        }
    }

    // ---------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------

    /**
     * Performs single inference using gRPC (placeholder implementation).
     *
     * <p><strong>TODO:</strong> Replace with actual gRPC calls after generating stubs.
     * Example for TensorFlow Serving:
     * <pre>{@code
     * // Generate stubs from tensorflow_serving/apis/*.proto
     * PredictionServiceGrpc.PredictionServiceBlockingStub stub =
     *     PredictionServiceGrpc.newBlockingStub(channel)
     *         .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS);
     *
     * PredictRequest request = buildPredictRequest(inputs);
     * PredictResponse response = stub.predict(request);
     * Map<String, Object> outputs = extractOutputs(response);
     * }</pre>
     *
     * @param inputs map of input names to values
     * @return placeholder inference result
     * @throws InferenceException always throws (placeholder implementation)
     */
    @Override
    public InferenceResult infer(Map<String, Object> inputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            /*
             * TODO: Replace this placeholder with real gRPC call:
             *
             * PredictRequest request = ...
             * PredictResponse response = blockingStub
             *      .withDeadlineAfter(inferenceConfig.getTimeoutMs(), TimeUnit.MILLISECONDS)
             *      .predict(request);
             */

            // Placeholder output
            Map<String, Object> output = Map.of("prediction", 0.5);

            return new InferenceResult(
                    output,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (StatusRuntimeException e) {
            throw new InferenceException(
                    "gRPC inference failed: " + e.getStatus(), e);
        }
    }

    /**
     * Performs batch inference using sequential processing (placeholder).
     *
     * <p><strong>TODO:</strong> Implement native batch support if gRPC server supports it.
     * Some frameworks like Triton support batch predictions via gRPC.
     *
     * @param batchInputs array of input maps for batch processing
     * @return aggregated inference results
     * @throws InferenceException if batch inference fails
     */
    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            Map<String, Object> aggregated = new HashMap<>();

            for (int i = 0; i < batchInputs.length; i++) {
                // Sequential fallback; replace with real batch call if supported
                InferenceResult result = infer(batchInputs[i]);
                aggregated.put("result_" + i, result.getOutputs());
            }

            return new InferenceResult(
                    aggregated,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (Exception e) {
            throw new InferenceException("gRPC batch inference failed", e);
        }
    }

    // ---------------------------------------------------------------------
    // Engine Info
    // ---------------------------------------------------------------------

    /**
     * Checks if the gRPC client is ready for inference.
     *
     * @return true if initialized and channel is not shutdown
     */
    @Override
    public boolean isReady() {
        return initialized && channel != null && !channel.isShutdown();
    }

    /**
     * Gets metadata about the gRPC model.
     *
     * @return model metadata created during initialization
     */
    @Override
    public ModelMetadata getMetadata() {
        return metadata;
    }

    /**
     * Gets the model configuration.
     *
     * @return model configuration from inference config
     */
    @Override
    public ModelConfig getModelConfig() {
        return inferenceConfig.getModelConfig();
    }

    /**
     * Gets the engine's capabilities for gRPC inference.
     *
     * <p>gRPC capabilities:
     * <ul>
     *   <li><b>Remote:</b> Yes, communicates with gRPC server</li>
     *   <li><b>Batch:</b> Yes, depends on server implementation</li>
     *   <li><b>Max Batch Size:</b> From {@link InferenceConfig#getBatchSize()}</li>
     *   <li><b>Async:</b> Depends on stub implementation</li>
     * </ul>
     *
     * @return engine capabilities for gRPC
     */
    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(
                true,   // remote
                true,   // batch supported (engine-level)
                inferenceConfig.getBatchSize(),
                false   // async depends on stub implementation
        );
    }

    // ---------------------------------------------------------------------
    // Shutdown
    // ---------------------------------------------------------------------

    /**
     * Closes the gRPC channel and releases resources.
     *
     * <p>Gracefully shuts down the {@link ManagedChannel} with 5-second timeout.
     * If shutdown is interrupted, forces immediate shutdown.
     *
     * <p>Always call this method when finished to release connections and threads.
     */
    @Override
    public void close() {
        LOG.info("Closing gRPC inference client");

        if (channel != null) {
            try {
                channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                LOG.warn("Channel shutdown interrupted", e);
                channel.shutdownNow();
            }
        }
        initialized = false;
    }

    // ---------------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------------

    /**
     * Ensures the client is initialized before use.
     *
     * @throws InferenceException if client is not initialized
     */
    private void ensureInitialized() throws InferenceException {
        if (!initialized || channel == null) {
            throw new InferenceException("gRPC inference client not initialized");
        }
    }
}