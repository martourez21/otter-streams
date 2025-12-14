package com.codedstream.otterstream.remote.vertex;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.cloud.aiplatform.v1.EndpointName;
import com.google.cloud.aiplatform.v1.PredictRequest;
import com.google.cloud.aiplatform.v1.PredictResponse;
import com.google.cloud.aiplatform.v1.PredictionServiceClient;
import com.google.cloud.aiplatform.v1.PredictionServiceSettings;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Google Vertex AI remote inference client for Google Cloud ML models.
 *
 * <p>This engine provides integration with Google Cloud Vertex AI endpoints for
 * inference on models hosted in Google Cloud. It uses the Google Cloud Vertex AI
 * Java client library to communicate with the PredictionService API.
 *
 * <h2>Supported Features:</h2>
 * <ul>
 *   <li><b>Vertex AI Endpoints:</b> Integration with deployed Vertex AI model endpoints</li>
 *   <li><b>Google Cloud Authentication:</b> Automatic credentials via Application Default Credentials</li>
 *   <li><b>Protobuf Payloads:</b> Automatic conversion between Java Maps and protobuf Values</li>
 *   <li><b>Batch Inference:</b> Native support for batch predictions</li>
 *   <li><b>Region Configuration:</b> Configurable Google Cloud regions</li>
 * </ul>
 *
 * <h2>Configuration Example:</h2>
 * <pre>{@code
 * InferenceConfig inferenceConfig = InferenceConfig.builder()
 *     .modelConfig(ModelConfig.builder()
 *         .modelName("vertex-model")
 *         .modelVersion("v1")
 *         .build())
 *     .engineOption("endpoint", "projects/my-project/locations/us-central1/endpoints/my-endpoint")
 *     .engineOption("project_id", "my-project")
 *     .engineOption("location", "us-central1")
 *     .build();
 *
 * VertexAIInferenceClient client = new VertexAIInferenceClient(inferenceConfig);
 * client.initialize(inferenceConfig.getModelConfig());
 * }</pre>
 *
 * <h2>Authentication:</h2>
 * <p>Uses Google Cloud Application Default Credentials (ADC) which automatically
 * searches for credentials in this order:
 * <ol>
 *   <li>GOOGLE_APPLICATION_CREDENTIALS environment variable</li>
 *   <li>Google Cloud SDK default credentials</li>
 *   <li>Google App Engine credentials</li>
 *   <li>Google Cloud Shell credentials</li>
 *   <li>Google Compute Engine credentials</li>
 * </ol>
 *
 * <h2>Endpoint Name Formats:</h2>
 * <ul>
 *   <li><b>Full Resource Name:</b> projects/{project}/locations/{location}/endpoints/{endpoint}</li>
 *   <li><b>Separate Components:</b> Provide project_id, location, and endpoint separately</li>
 * </ul>
 *
 * <h2>Vertex AI Request Format:</h2>
 * <pre>
 * {
 *   "instances": [
 *     {
 *       "feature1": value1,
 *       "feature2": value2,
 *       ...
 *     }
 *   ]
 * }
 * </pre>
 *
 * <h2>Performance Features:</h2>
 * <ul>
 *   <li><b>gRPC-based:</b> Uses high-performance gRPC protocol</li>
 *   <li><b>Connection Pooling:</b> Managed by Google Cloud client library</li>
 *   <li><b>Request Batching:</b> Native batch support for throughput optimization</li>
 *   <li><b>Automatic Retry:</b> Built-in retry with exponential backoff</li>
 * </ul>
 *
 * <h2>Error Handling:</h2>
 * <ul>
 *   <li>Vertex AI API errors throw {@link InferenceException}</li>
 *   <li>Authentication failures throw {@link InferenceException}</li>
 *   <li>Network/timeout errors are wrapped in {@link InferenceException}</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>{@link PredictionServiceClient} is thread-safe when created with default settings.
 * The client manages its own connection pooling and request lifecycle.
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see InferenceEngine
 * @see PredictionServiceClient
 * @see <a href="https://cloud.google.com/vertex-ai/docs/predictions/get-predictions">Vertex AI Predictions Documentation</a>
 */
public class VertexAIInferenceClient
        implements InferenceEngine<PredictionServiceClient> {

    private static final Logger LOG =
            LoggerFactory.getLogger(VertexAIInferenceClient.class);

    private final InferenceConfig inferenceConfig;

    private PredictionServiceClient predictionClient;
    private ObjectMapper objectMapper;
    private ModelMetadata metadata;
    private EndpointName endpointName;
    private boolean initialized = false;

    /**
     * Constructs a new Vertex AI inference client with the provided configuration.
     *
     * @param inferenceConfig inference configuration containing Vertex AI options
     */
    public VertexAIInferenceClient(InferenceConfig inferenceConfig) {
        this.inferenceConfig = inferenceConfig;
    }

    // ---------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------

    /**
     * Initializes the Vertex AI inference client with Google Cloud configuration.
     *
     * <p>Initialization process:
     * <ol>
     *   <li>Parses endpoint configuration from engine options</li>
     *   <li>Creates {@link EndpointName} from resource name or components</li>
     *   <li>Builds {@link PredictionServiceSettings} with automatic region endpoint</li>
     *   <li>Creates {@link PredictionServiceClient} with Application Default Credentials</li>
     *   <li>Initializes {@link ModelMetadata} with Vertex AI format</li>
     * </ol>
     *
     * <h2>Required Engine Options:</h2>
     * <ul>
     *   <li><b>endpoint:</b> Full resource name OR endpoint name</li>
     *   <li><b>project_id:</b> Google Cloud project ID (if not in endpoint)</li>
     *   <li><b>location:</b> Google Cloud region (default: us-central1)</li>
     * </ul>
     *
     * @param modelConfig model configuration containing model name and version
     * @throws InferenceException if initialization fails or required options are missing
     * @throws IllegalArgumentException if endpoint format is invalid
     */
    @Override
    public void initialize(ModelConfig modelConfig) throws InferenceException {
        LOG.info("Initializing Vertex AI inference engine");

        try {
            Map<String, Object> options = inferenceConfig.getEngineOptions();
            String endpoint = (String) options.get("endpoint");

            if (endpoint == null) {
                throw new IllegalArgumentException("Vertex AI endpoint not configured");
            }

            if (endpoint.startsWith("projects/")) {
                String[] parts = endpoint.split("/");
                if (parts.length < 6) {
                    throw new IllegalArgumentException(
                            "Invalid Vertex AI endpoint format: " + endpoint);
                }
                endpointName = EndpointName.of(parts[1], parts[3], parts[5]);
            } else {
                String project = (String) options.get("project_id");
                String location = (String) options.getOrDefault("location", "us-central1");
                endpointName = EndpointName.of(project, location, endpoint);
            }

            PredictionServiceSettings settings =
                    PredictionServiceSettings.newBuilder()
                            .setEndpoint(
                                    endpointName.getLocation()
                                            + "-aiplatform.googleapis.com:443")
                            .build();

            this.predictionClient = PredictionServiceClient.create(settings);
            this.objectMapper = new ObjectMapper();

            this.metadata = new ModelMetadata(
                    modelConfig.getModelName(),
                    modelConfig.getModelVersion(),
                    ModelFormat.VERTEX_AI,
                    Map.of(), // input schema (optional)
                    Map.of(), // output schema (optional)
                    -1L,      // model size unknown
                    System.currentTimeMillis()
            );

            this.initialized = true;
            LOG.info("Vertex AI inference engine initialized");

        } catch (Exception e) {
            throw new InferenceException("Failed to initialize Vertex AI engine", e);
        }
    }

    // ---------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------

    /**
     * Performs single inference using Vertex AI PredictionService.
     *
     * <p>Request flow:
     * <ol>
     *   <li>Converts input Map to protobuf {@link Value} using JSON serialization</li>
     *   <li>Creates {@link PredictRequest} with single instance</li>
     *   <li>Calls {@link PredictionServiceClient#predict}</li>
     *   <li>Extracts first prediction from {@link PredictResponse}</li>
     *   <li>Converts protobuf Value back to Java Map</li>
     *   <li>Returns {@link InferenceResult} with timing information</li>
     * </ol>
     *
     * <h2>Vertex AI Response Format:</h2>
     * <p>Vertex AI returns predictions in the format configured during model deployment.
     * Common formats include:
     * <ul>
     *   <li>Classification: {"classes": ["class1", "class2"], "scores": [0.8, 0.2]}</li>
     *   <li>Regression: {"value": 42.5}</li>
     *   <li>Custom outputs based on model signature</li>
     * </ul>
     *
     * @param inputs map of input names to values (must be JSON-serializable)
     * @return inference result containing Vertex AI predictions
     * @throws InferenceException if Vertex AI API call fails or no predictions returned
     */
    @Override
    public InferenceResult infer(Map<String, Object> inputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            PredictRequest request = PredictRequest.newBuilder()
                    .setEndpoint(endpointName.toString())
                    .addInstances(toValue(inputs))
                    .build();

            PredictResponse response = predictionClient.predict(request);

            if (response.getPredictionsCount() == 0) {
                throw new InferenceException("No predictions returned from Vertex AI");
            }

            Map<String, Object> output =
                    fromValue(response.getPredictions(0));

            return new InferenceResult(
                    output,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (Exception e) {
            throw new InferenceException("Vertex AI inference failed", e);
        }
    }

    /**
     * Performs batch inference using Vertex AI native batch support.
     *
     * <p>Vertex AI natively supports batch predictions, which is more efficient than
     * sequential single inferences. This method:
     * <ol>
     *   <li>Converts all batch inputs to protobuf Values</li>
     *   <li>Sends single batch request with all instances</li>
     *   <li>Receives batch response with all predictions</li>
     *   <li>Aggregates results with index prefixes ("result_0", "result_1", etc.)</li>
     * </ol>
     *
     * <h2>Batch Size Considerations:</h2>
     * <p>Vertex AI has limits on batch size (varies by model and endpoint configuration).
     * Check Vertex AI documentation for current limits. The batch size from
     * {@link InferenceConfig} is used but may be limited by Vertex AI quotas.
     *
     * @param batchInputs array of input maps for batch processing
     * @return aggregated inference result containing all batch predictions
     * @throws InferenceException if batch inference fails or batch size exceeds limits
     */
    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            List<Value> instances = new ArrayList<>();
            for (Map<String, Object> input : batchInputs) {
                instances.add(toValue(input));
            }

            PredictRequest request = PredictRequest.newBuilder()
                    .setEndpoint(endpointName.toString())
                    .addAllInstances(instances)
                    .build();

            PredictResponse response = predictionClient.predict(request);

            Map<String, Object> aggregated = new HashMap<>();
            int i = 0;
            for (Value prediction : response.getPredictionsList()) {
                aggregated.put("result_" + i++, fromValue(prediction));
            }

            return new InferenceResult(
                    aggregated,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (Exception e) {
            throw new InferenceException("Vertex AI batch inference failed", e);
        }
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    /**
     * Converts Java Map to protobuf Value using JSON serialization.
     *
     * @param input Java Map containing inference inputs
     * @return protobuf Value for Vertex AI API
     * @throws Exception if JSON serialization fails
     */
    private Value toValue(Map<String, Object> input) throws Exception {
        String json = objectMapper.writeValueAsString(input);
        Value.Builder builder = Value.newBuilder();
        JsonFormat.parser().merge(json, builder);
        return builder.build();
    }

    /**
     * Converts protobuf Value back to Java Map using JSON deserialization.
     *
     * @param value protobuf Value from Vertex AI response
     * @return Java Map containing inference outputs
     * @throws Exception if JSON deserialization fails
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> fromValue(Value value) throws Exception {
        String json = JsonFormat.printer().print(value);
        return objectMapper.readValue(json, Map.class);
    }

    /**
     * Ensures the engine is initialized before use.
     *
     * @throws InferenceException if engine is not initialized
     */
    private void ensureInitialized() throws InferenceException {
        if (!initialized || predictionClient == null) {
            throw new InferenceException("Vertex AI engine not initialized");
        }
    }

    // ---------------------------------------------------------------------
    // Engine Info
    // ---------------------------------------------------------------------

    /**
     * Checks if the engine is ready for inference.
     *
     * @return true if initialized and prediction client is available
     */
    @Override
    public boolean isReady() {
        return initialized;
    }

    /**
     * Gets metadata about the Vertex AI model.
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
     * Gets the engine's capabilities for Vertex AI inference.
     *
     * <p>Vertex AI capabilities:
     * <ul>
     *   <li><b>Remote:</b> Yes, communicates with cloud service</li>
     *   <li><b>Batch:</b> Yes, native batch support</li>
     *   <li><b>Max Batch Size:</b> From {@link InferenceConfig#getBatchSize()}</li>
     *   <li><b>Async:</b> Yes, platform supports async operations</li>
     * </ul>
     *
     * @return engine capabilities for Vertex AI
     */
    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(
                true,   // remote
                true,   // batch
                inferenceConfig.getBatchSize(),
                true    // async supported by platform
        );
    }

    /**
     * Closes the Vertex AI client and releases Google Cloud resources.
     *
     * <p>Closes the {@link PredictionServiceClient} which releases gRPC channels
     * and connection pools. Always call this method when finished to prevent
     * resource leaks.
     */
    @Override
    public void close() {
        if (predictionClient != null) {
            predictionClient.close();
        }
        initialized = false;
    }
}