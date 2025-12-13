package com.codedstream.otterstream.remote.sagemaker;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import com.codedstream.otterstream.remote.RemoteInferenceEngine;
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
    private InferenceConfig inferenceConfig;
    private ModelMetadata metadata;

    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        super.initialize(config);
        this.inferenceConfig = new InferenceConfig.Builder()
                .modelConfig(config)
                .build(); // Wrap ModelConfig into InferenceConfig if needed

        try {
            SageMakerRuntimeClient.Builder clientBuilder = SageMakerRuntimeClient.builder()
                    .region(Region.US_EAST_1); // default, make configurable if needed

            // Configure credentials if provided
            if (config.getAuthConfig() != null && config.getAuthConfig().getApiKey() != null) {
                String[] credentials = config.getAuthConfig().getApiKey().split(":");
                if (credentials.length == 2) {
                    AwsBasicCredentials awsCreds = AwsBasicCredentials.create(credentials[0], credentials[1]);
                    clientBuilder.credentialsProvider(StaticCredentialsProvider.create(awsCreds));
                }
            }

            this.sageMakerClient = clientBuilder.build();
            this.objectMapper = new ObjectMapper();

            // Initialize metadata
            this.metadata = new ModelMetadata(
                    config.getModelId(),
                    config.getModelVersion(),
                    config.getFormat(),
                    Map.of(), // empty input schema placeholder
                    Map.of(), // empty output schema placeholder
                    0,
                    System.currentTimeMillis()
            );
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
                    .endpointName(endpointUrl) // make sure endpointUrl is set from config
                    .contentType("application/json")
                    .body(body)
                    .build();

            var response = sageMakerClient.invokeEndpoint(request);
            String responseBody = response.body().asUtf8String();

            @SuppressWarnings("unchecked")
            Map<String, Object> outputs = objectMapper.readValue(responseBody, Map.class);

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, inferenceConfig.getModelConfig().getModelId());
        } catch (Exception e) {
            throw new InferenceException("SageMaker inference failed", e);
        }
    }

    @Override
    public boolean validateConnection() throws InferenceException {
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

    @Override
    public ModelMetadata getMetadata() {
        return metadata;
    }

    @Override
    public ModelConfig getModelConfig() {
        return inferenceConfig.getModelConfig();
    }
}
