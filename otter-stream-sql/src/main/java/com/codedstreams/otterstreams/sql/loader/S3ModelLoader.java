package com.codedstreams.otterstreams.sql.loader;

import com.codedstreams.otterstreams.sql.config.ModelSourceConfig;
import com.codedstreams.otterstreams.sql.config.ModelSourceConfig.SourceType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3ClientBuilder;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;

import java.io.InputStream;
import java.net.URI;

/**
 * Loads ML models from AWS S3 or S3-compatible storage (e.g. MinIO).
 */
public class S3ModelLoader implements ModelLoader {

    private static final Logger LOG = LoggerFactory.getLogger(S3ModelLoader.class);

    private final ModelSourceConfig config;
    private final S3Client s3Client;

    public S3ModelLoader(ModelSourceConfig config) {
        if (config.getSourceType() != SourceType.S3 && config.getSourceType() != SourceType.MINIO) {
            throw new IllegalArgumentException("S3ModelLoader supports only S3 or MinIO source types");
        }
        this.config = config;
        this.s3Client = createS3Client();
    }

    private S3Client createS3Client() {
        S3ClientBuilder builder = S3Client.builder();

        // Credentials (optional for IAM / IRSA / instance profiles)
        if (config.hasAuthentication() && config.getAccessKey() != null && config.getSecretKey() != null) {
            AwsBasicCredentials credentials = AwsBasicCredentials.create(
                    config.getAccessKey(),
                    config.getSecretKey()
            );
            builder.credentialsProvider(StaticCredentialsProvider.create(credentials));
        }

        // Region is required by AWS SDK even for MinIO
        builder.region(Region.of(config.getRegion() != null ? config.getRegion() : "us-east-1"));

        // Optional custom endpoint (MinIO, LocalStack, etc.)
        if (config.getSourceType() == SourceType.MINIO && config.getMinioEndpoint() != null) {
            builder.endpointOverride(URI.create(config.getMinioEndpoint()));
        }

        return builder.build();
    }

    @Override
    public InputStream loadModel() {
        String path = config.getModelPath();

        // Validate path
        if (!(path.startsWith("s3://") || path.startsWith("minio://"))) {
            throw new IllegalArgumentException(
                    "Invalid S3/MinIO path: " + path + " (expected s3://bucket/key or minio://bucket/key)"
            );
        }

        // Strip protocol
        String strippedPath = path.contains("://") ? path.split("://", 2)[1] : path;
        String[] parts = strippedPath.split("/", 2);
        String bucket = parts[0];
        String key = parts.length > 1 ? parts[1] : "";

        LOG.info("Loading model from {}: bucket={}, key={}", config.getSourceType(), bucket, key);

        GetObjectRequest request = GetObjectRequest.builder()
                .bucket(bucket)
                .key(key)
                .build();

        return s3Client.getObject(request);
    }

    @Override
    public String getModelPath() {
        return config.getModelPath();
    }
}
