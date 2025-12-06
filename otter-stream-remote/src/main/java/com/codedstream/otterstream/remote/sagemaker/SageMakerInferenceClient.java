package com.codedstream.otterstream.remote.sagemaker;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest;
import software.amazon.awssdk.core.SdkBytes;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;

public class SageMakerInferenceClient extends RemoteInferenceEngine {

    private SageMakerRuntimeClient sageMakerClient;
    private ObjectMapper objectMapper;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        super.initialize(config);

        try {
            SageMakerRuntimeClient.Builder clientBuilder = SageMakerRuntimeClient.builder()
                    .region(Region.US_EAST_1); // Default region, should be configurable

            // Configure credentials if provided
            if (config.getAuthConfig() != null && config.getAuthConfig().getApiKey() != null) {
                // Assuming API key contains access key and secret key separated by colon
                String[] credentials = config.getAuthConfig().getApiKey().split(":");
                if (credentials.length == 2) {
                    AwsBasicCredentials awsCreds = AwsBasicCredentials.create(credentials[0], credentials[1]);
                    clientBuilder.credentialsProvider(StaticCredentialsProvider.create(awsCreds));
                }
            }

            this.sageMakerClient = clientBuilder.build();
            this.objectMapper = new ObjectMapper();
        } catch (Exception e) {
            throw new InferenceException("Failed to initialize SageMaker client", e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try {
            long startTime = System.currentTimeMillis();

            String jsonBody = objectMapper.writeValueAsString(inputs);
            SdkBytes body = SdkBytes.fromUtf8String(jsonBody);

            InvokeEndpointRequest request = InvokeEndpointRequest.builder()
                    .endpointName(endpointUrl) // endpointUrl contains the endpoint name
                    .contentType("application/json")
                    .body(body)
                    .build();

            var response = sageMakerClient.invokeEndpoint(request);
            String responseBody = response.body().asUtf8String();

            @SuppressWarnings("unchecked")
            Map<String, Object> outputs = objectMapper.readValue(responseBody, Map.class);

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("SageMaker inference failed", e);
        }
    }

    @Override
    public boolean validateConnection() throws InferenceException {
        // SageMaker doesn't have a direct connection validation endpoint
        // We can try a simple inference with dummy data
        try {
            Map<String, Object> dummyInput = Map.of("validate", "connection");
            infer(dummyInput);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public void close() throws InferenceException {
        if (sageMakerClient != null) {
            sageMakerClient.close();
        }
        super.close();
    }
}
