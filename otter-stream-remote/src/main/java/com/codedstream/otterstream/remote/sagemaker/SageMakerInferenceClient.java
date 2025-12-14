package com.codedstream.otterstream.remote.sagemaker;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import com.codedstream.otterstream.remote.RemoteInferenceEngine;
import com.fasterxml.jackson.databind.ObjectMapper;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient;
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClientBuilder;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest;

import java.util.Map;

/**
 * AWS SageMaker remote inference client for hosted ML models.
 *
 * <p>This engine provides integration with AWS SageMaker endpoints for inference
 * on models hosted in the AWS cloud. It uses the AWS SDK for Java v2 to communicate
 * with SageMaker Runtime API, supporting both static credentials and AWS IAM roles.
 *
 * <h2>Supported Features:</h2>
 * <ul>
 *   <li><b>SageMaker Endpoints:</b> Integration with deployed SageMaker model endpoints</li>
 *   <li><b>AWS Authentication:</b> Static credentials or IAM role-based authentication</li>
 *   <li><b>Region Configuration:</b> Configurable AWS regions (default: us-east-1)</li>
 *   <li><b>JSON Payloads:</b> Automatic serialization of inputs to SageMaker format</li>
 *   <li><b>Connection Validation:</b> Test inference with ping request</li>
 * </ul>
 *
 * <h2>Configuration Example:</h2>
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelId("sagemaker-model")
 *     .endpointUrl("my-sagemaker-endpoint") // SageMaker endpoint name
 *     .authConfig(AuthConfig.builder()
 *         .apiKey("ACCESS_KEY:SECRET_KEY") // Optional static credentials
 *         .build())
 *     .build();
 *
 * SageMakerInferenceClient client = new SageMakerInferenceClient();
 * client.initialize(config);
 * }</pre>
 *
 * <h2>Authentication Options:</h2>
 * <ol>
 *   <li><b>Static Credentials:</b> Provide ACCESS_KEY:SECRET_KEY in authConfig.apiKey</li>
 *   <li><b>IAM Role:</b> Omit credentials to use AWS IAM role (EC2, ECS, Lambda)</li>
 *   <li><b>Profile:</b> Use AWS profile from ~/.aws/credentials</li>
 * </ol>
 *
 * <h2>SageMaker Request Format:</h2>
 * <pre>
 * POST /endpoints/{endpoint-name}/invocations
 * Content-Type: application/json
 *
 * {
 *   "feature1": value1,
 *   "feature2": value2,
 *   ...
 * }
 * </pre>
 *
 * <h2>Error Handling:</h2>
 * <ul>
 *   <li>SageMaker API errors throw {@link InferenceException}</li>
 *   <li>Authentication failures throw {@link InferenceException}</li>
 *   <li>Network/timeout errors are wrapped in {@link InferenceException}</li>
 * </ul>
 *
 * <h2>AWS SDK Integration:</h2>
 * <p>Uses AWS SDK for Java v2 with automatic retry logic, request compression,
 * and connection pooling. The SDK handles:
 * <ul>
 *   <li>Request signing with AWS Signature Version 4</li>
 *   <li>Automatic retry with exponential backoff</li>
 *   <li>Connection management and pooling</li>
 *   <li>Request/response logging (when configured)</li>
 * </ul>
 *
 * <h2>Cost Considerations:</h2>
 * <ul>
 *   <li>SageMaker charges per inference hour + data transfer</li>
 *   <li>Consider batch inference to reduce cost per prediction</li>
 *   <li>Use appropriate instance types for cost-performance balance</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>{@link SageMakerRuntimeClient} is thread-safe and can be shared across threads.
 * The client uses connection pooling and automatic request retry.
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see RemoteInferenceEngine
 * @see SageMakerRuntimeClient
 * @see <a href="https://docs.aws.amazon.com/sagemaker/latest/dg/API_runtime_InvokeEndpoint.html">SageMaker InvokeEndpoint API</a>
 */
public class SageMakerInferenceClient extends RemoteInferenceEngine {

    private SageMakerRuntimeClient sageMakerClient;
    private ObjectMapper objectMapper;
    private InferenceConfig inferenceConfig;
    private ModelMetadata metadata;

    /**
     * Initializes the SageMaker inference client with AWS configuration.
     *
     * <p>Initialization process:
     * <ol>
     *   <li>Creates {@link SageMakerRuntimeClient} with configured region</li>
     *   <li>Sets up static credentials if provided in authConfig.apiKey</li>
     *   <li>Initializes {@link ObjectMapper} for JSON serialization</li>
     *   <li>Creates basic {@link ModelMetadata} from configuration</li>
     * </ol>
     *
     * <h2>Region Configuration:</h2>
     * <p>Currently defaults to us-east-1. Extend to support region configuration
     * via model options if needed.
     *
     * <h2>Credential Parsing:</h2>
     * <p>If authConfig.apiKey is provided, it should be in format "ACCESS_KEY:SECRET_KEY".
     * The colon separates access key from secret key.
     *
     * @param config model configuration containing SageMaker endpoint name
     * @throws InferenceException if client initialization fails or credentials are invalid
     */
    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        super.initialize(config);

        this.inferenceConfig = InferenceConfig.builder()
                .modelConfig(config)
                .build();

        try {
            SageMakerRuntimeClientBuilder clientBuilder =
                    SageMakerRuntimeClient.builder()
                            .region(Region.US_EAST_1); // make configurable if needed

            // Optional static credentials
            if (config.getAuthConfig() != null && config.getAuthConfig().getApiKey() != null) {
                String[] credentials = config.getAuthConfig().getApiKey().split(":");
                if (credentials.length == 2) {
                    AwsBasicCredentials awsCreds =
                            AwsBasicCredentials.create(credentials[0], credentials[1]);
                    clientBuilder.credentialsProvider(
                            StaticCredentialsProvider.create(awsCreds)
                    );
                }
            }

            this.sageMakerClient = clientBuilder.build();
            this.objectMapper = new ObjectMapper();

            this.metadata = new ModelMetadata(
                    config.getModelId(),
                    config.getModelVersion(),
                    config.getFormat(),
                    Map.of(),
                    Map.of(),
                    0,
                    System.currentTimeMillis()
            );

        } catch (Exception e) {
            throw new InferenceException("Failed to initialize SageMaker client", e);
        }
    }

    /**
     * Invokes SageMaker endpoint for inference.
     *
     * <p>Request flow:
     * <ol>
     *   <li>Serialize inputs to JSON using {@link ObjectMapper}</li>
     *   <li>Create {@link InvokeEndpointRequest} with endpoint name and JSON body</li>
     *   <li>Execute request via {@link SageMakerRuntimeClient#invokeEndpoint}</li>
     *   <li>Parse response JSON back to Map</li>
     *   <li>Return {@link InferenceResult} with timing information</li>
     * </ol>
     *
     * <h2>SageMaker Response Format:</h2>
     * <p>SageMaker returns the raw model output as JSON. The structure depends on
     * the model's output configuration. Common formats include:
     * <ul>
     *   <li>Single value: {"prediction": 0.75}</li>
     *   <li>Array: {"predictions": [0.1, 0.2, 0.7]}</li>
     *   <li>Multiple outputs: {"class": "cat", "confidence": 0.92}</li>
     * </ul>
     *
     * @param inputs map of input names to values (must be JSON-serializable)
     * @return inference result containing SageMaker model outputs
     * @throws InferenceException if SageMaker API call fails or response parsing fails
     */
    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try {
            long start = System.currentTimeMillis();

            String json = objectMapper.writeValueAsString(inputs);
            SdkBytes body = SdkBytes.fromUtf8String(json);

            InvokeEndpointRequest request = InvokeEndpointRequest.builder()
                    .endpointName(endpointUrl)
                    .contentType("application/json")
                    .body(body)
                    .build();

            var response = sageMakerClient.invokeEndpoint(request);

            @SuppressWarnings("unchecked")
            Map<String, Object> outputs =
                    objectMapper.readValue(response.body().asUtf8String(), Map.class);

            return new InferenceResult(
                    outputs,
                    System.currentTimeMillis() - start,
                    inferenceConfig.getModelConfig().getModelId()
            );

        } catch (Exception e) {
            throw new InferenceException("SageMaker inference failed", e);
        }
    }

    /**
     * Validates connection to SageMaker endpoint by sending a test inference.
     *
     * <p>Sends a simple "ping" inference request to verify:
     * <ul>
     *   <li>Endpoint exists and is accessible</li>
     *   <li>Authentication works</li>
     *   <li>Endpoint responds to inference requests</li>
     * </ul>
     *
     * <p><strong>Note:</strong> This may incur SageMaker charges for the test inference.
     * Consider implementing a lighter validation if cost is a concern.
     *
     * @return true if test inference succeeds
     */
    @Override
    public boolean validateConnection() {
        try {
            infer(Map.of("ping", "ok"));
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Closes the SageMaker client and releases AWS resources.
     *
     * <p>Closes the {@link SageMakerRuntimeClient} which releases HTTP connections
     * and thread pools. Always call this method when finished to prevent resource leaks.
     *
     * @throws InferenceException if resource cleanup fails
     */
    @Override
    public void close() throws InferenceException {
        if (sageMakerClient != null) {
            sageMakerClient.close();
        }
        super.close();
    }

    /**
     * Gets metadata about the SageMaker model.
     *
     * @return model metadata extracted during initialization
     */
    @Override
    public ModelMetadata getMetadata() {
        return metadata;
    }

    /**
     * Gets the model configuration.
     *
     * @return the model configuration from inference config
     */
    @Override
    public ModelConfig getModelConfig() {
        return inferenceConfig.getModelConfig();
    }
}