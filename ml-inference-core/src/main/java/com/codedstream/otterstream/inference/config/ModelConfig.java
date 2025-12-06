package com.codedstream.otterstream.inference.config;

import com.codedstream.otterstream.inference.model.ModelFormat;

import java.util.Map;
import java.util.Objects;

public class ModelConfig {
    private final String modelId;
    private final String modelPath;
    private final ModelFormat format;
    private final String modelName;
    private final String modelVersion;
    private final Map<String, Object> modelOptions;
    private final String endpointUrl;
    private final AuthConfig authConfig;

    public ModelConfig(String modelId, String modelPath, ModelFormat format,
                       String modelName, String modelVersion, Map<String, Object> modelOptions,
                       String endpointUrl, AuthConfig authConfig) {
        this.modelId = Objects.requireNonNull(modelId);
        this.modelPath = modelPath;
        this.format = Objects.requireNonNull(format);
        this.modelName = modelName;
        this.modelVersion = modelVersion;
        this.modelOptions = Map.copyOf(modelOptions);
        this.endpointUrl = endpointUrl;
        this.authConfig = authConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModelId() { return modelId; }
    public String getModelPath() { return modelPath; }
    public ModelFormat getFormat() { return format; }
    public String getModelName() { return modelName; }
    public String getModelVersion() { return modelVersion; }
    public Map<String, Object> getModelOptions() { return modelOptions; }
    public String getEndpointUrl() { return endpointUrl; }
    public AuthConfig getAuthConfig() { return authConfig; }
    public boolean isRemote() { return endpointUrl != null && !endpointUrl.isEmpty(); }

    public static class Builder {
        private String modelId;
        private String modelPath;
        private ModelFormat format;
        private String modelName = "default";
        private String modelVersion = "1.0";
        private Map<String, Object> modelOptions = Map.of();
        private String endpointUrl;
        private AuthConfig authConfig;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder modelPath(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder modelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        public Builder modelOptions(Map<String, Object> modelOptions) {
            this.modelOptions = Map.copyOf(modelOptions);
            return this;
        }

        public Builder endpointUrl(String endpointUrl) {
            this.endpointUrl = endpointUrl;
            return this;
        }

        public Builder authConfig(AuthConfig authConfig) {
            this.authConfig = authConfig;
            return this;
        }

        public ModelConfig build() {
            return new ModelConfig(modelId, modelPath, format, modelName, modelVersion,
                    modelOptions, endpointUrl, authConfig);
        }
    }
}
