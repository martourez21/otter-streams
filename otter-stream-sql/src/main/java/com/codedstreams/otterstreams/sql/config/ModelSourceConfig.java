package com.codedstreams.otterstreams.sql.config;

import com.codedstream.otterstream.inference.model.ModelFormat;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration for model source locations and loading strategies.
 *
 * <p>Supports loading models from:
 * <ul>
 *   <li>Local filesystem: {@code file:///path/to/model}</li>
 *   <li>HDFS: {@code hdfs://namenode:port/path/to/model}</li>
 *   <li>AWS S3: {@code s3://bucket/prefix/model}</li>
 *   <li>MinIO: {@code minio://endpoint/bucket/model}</li>
 *   <li>HTTP/HTTPS: {@code https://model-server/models/my-model}</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // S3 model source
 * ModelSourceConfig s3Source = ModelSourceConfig.builder()
 *     .modelPath("s3://my-bucket/models/fraud-detector/")
 *     .modelFormat(ModelFormat.TENSORFLOW_SAVEDMODEL)
 *     .credentials("ACCESS_KEY", "SECRET_KEY")
 *     .region("us-east-1")
 *     .build();
 *
 * // Local filesystem
 * ModelSourceConfig localSource = ModelSourceConfig.builder()
 *     .modelPath("file:///opt/models/sentiment/")
 *     .modelFormat(ModelFormat.TENSORFLOW_GRAPHDEF)
 *     .build();
 *
 * // HTTP endpoint
 * ModelSourceConfig httpSource = ModelSourceConfig.builder()
 *     .modelPath("https://model-registry.example.com/models/v2/")
 *     .modelFormat(ModelFormat.ONNX)
 *     .authToken("Bearer xyz123...")
 *     .build();
 * }</pre>
 *
 * @author Nestor Martourez
 * @since 1.0.0
 */
public class ModelSourceConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    /** Model path/URI */
    private final String modelPath;

    /** Model format */
    private final ModelFormat modelFormat;

    /** Source type (derived from path) */
    private final SourceType sourceType;

    /** AWS/S3 credentials */
    private final String accessKey;
    private final String secretKey;
    private final String region;

    /** MinIO configuration */
    private final String minioEndpoint;

    /** HTTP authentication */
    private final String authToken;

    /** Custom headers for HTTP requests */
    private final Map<String, String> headers;

    /** Connection timeout for remote sources */
    private final long connectionTimeoutMs;

    /** Read timeout for remote sources */
    private final long readTimeoutMs;

    private ModelSourceConfig(Builder builder) {
        this.modelPath = Objects.requireNonNull(builder.modelPath, "modelPath is required");
        this.modelFormat = Objects.requireNonNull(builder.modelFormat, "modelFormat is required");
        this.sourceType = determineSourceType(modelPath);
        this.accessKey = builder.accessKey;
        this.secretKey = builder.secretKey;
        this.region = builder.region;
        this.minioEndpoint = builder.minioEndpoint;
        this.authToken = builder.authToken;
        this.headers = Map.copyOf(builder.headers);
        this.connectionTimeoutMs = builder.connectionTimeoutMs;
        this.readTimeoutMs = builder.readTimeoutMs;
    }

    /**
     * Determines source type from model path URI scheme.
     */
    private SourceType determineSourceType(String path) {
        if (path.startsWith("s3://")) {
            return SourceType.S3;
        } else if (path.startsWith("minio://")) {
            return SourceType.MINIO;
        } else if (path.startsWith("http://") || path.startsWith("https://")) {
            return SourceType.HTTP;
        } else if (path.startsWith("hdfs://")) {
            return SourceType.HDFS;
        } else if (path.startsWith("file://")) {
            return SourceType.LOCAL;
        } else {
            // Default to local if no scheme specified
            return SourceType.LOCAL;
        }
    }

    // Getters
    public String getModelPath() { return modelPath; }
    public ModelFormat getModelFormat() { return modelFormat; }
    public SourceType getSourceType() { return sourceType; }
    public String getAccessKey() { return accessKey; }
    public String getSecretKey() { return secretKey; }
    public String getRegion() { return region; }
    public String getMinioEndpoint() { return minioEndpoint; }
    public String getAuthToken() { return authToken; }
    public Map<String, String> getHeaders() { return headers; }
    public long getConnectionTimeoutMs() { return connectionTimeoutMs; }
    public long getReadTimeoutMs() { return readTimeoutMs; }

    /**
     * Checks if authentication is configured.
     */
    public boolean hasAuthentication() {
        return (accessKey != null && secretKey != null) || authToken != null;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelPath;
        private ModelFormat modelFormat = ModelFormat.TENSORFLOW_SAVEDMODEL;
        private String accessKey;
        private String secretKey;
        private String region = "us-east-1";
        private String minioEndpoint;
        private String authToken;
        private Map<String, String> headers = new HashMap<>();
        private long connectionTimeoutMs = 30000;
        private long readTimeoutMs = 60000;

        public Builder modelPath(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder modelFormat(ModelFormat modelFormat) {
            this.modelFormat = modelFormat;
            return this;
        }

        public Builder credentials(String accessKey, String secretKey) {
            this.accessKey = accessKey;
            this.secretKey = secretKey;
            return this;
        }

        public Builder region(String region) {
            this.region = region;
            return this;
        }

        public Builder minioEndpoint(String endpoint) {
            this.minioEndpoint = endpoint;
            return this;
        }

        public Builder authToken(String token) {
            this.authToken = token;
            return this;
        }

        public Builder headers(Map<String, String> headers) {
            this.headers = new HashMap<>(headers);
            return this;
        }

        public Builder addHeader(String key, String value) {
            this.headers.put(key, value);
            return this;
        }

        public Builder connectionTimeoutMs(long timeoutMs) {
            this.connectionTimeoutMs = timeoutMs;
            return this;
        }

        public Builder readTimeoutMs(long timeoutMs) {
            this.readTimeoutMs = timeoutMs;
            return this;
        }

        public ModelSourceConfig build() {
            return new ModelSourceConfig(this);
        }
    }

    /**
     * Enumeration of supported source types.
     */
    public enum SourceType {
        LOCAL,
        HDFS,
        S3,
        MINIO,
        HTTP
    }

    @Override
    public String toString() {
        return "ModelSourceConfig{" +
                "modelPath='" + modelPath + '\'' +
                ", modelFormat=" + modelFormat +
                ", sourceType=" + sourceType +
                ", region='" + region + '\'' +
                ", hasAuth=" + hasAuthentication() +
                '}';
    }
}
