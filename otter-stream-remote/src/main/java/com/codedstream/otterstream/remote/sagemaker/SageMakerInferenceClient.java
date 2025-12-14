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

public class SageMakerInferenceClient extends RemoteInferenceEngine {

    private SageMakerRuntimeClient sageMakerClient;
    private ObjectMapper objectMapper;
    private InferenceConfig inferenceConfig;
    private ModelMetadata metadata;

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

    @Override
    public boolean validateConnection() {
        try {
            infer(Map.of("ping", "ok"));
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
