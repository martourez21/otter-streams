package com.codedstreams.otterstreams.sql.loader;

import com.codedstreams.otterstreams.sql.config.ModelSourceConfig;
import io.minio.GetObjectArgs;
import io.minio.MinioClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;


/**
 * Loads models from MinIO object storage.
 */
public class MinioModelLoader implements ModelLoader {
    private static final Logger LOG = LoggerFactory.getLogger(MinioModelLoader.class);
    private final ModelSourceConfig config;
    private final MinioClient minioClient;

    public MinioModelLoader(ModelSourceConfig config) {
        this.config = config;
        this.minioClient = createMinioClient();
    }

    private MinioClient createMinioClient() {
        String endpoint = config.getMinioEndpoint();
        if (endpoint == null) {
            // Extract from path: minio://endpoint/bucket/key
            String path = config.getModelPath();
            endpoint = path.substring(8).split("/")[0];
        }

        MinioClient.Builder builder = MinioClient.builder()
                .endpoint(endpoint);

        if (config.hasAuthentication()) {
            builder.credentials(config.getAccessKey(), config.getSecretKey());
        }

        return builder.build();
    }

    @Override
    public InputStream loadModel() throws Exception {
        String path = config.getModelPath();
        // Extract bucket and object from minio://endpoint/bucket/object
        String[] parts = path.substring(path.indexOf('/', 9) + 1).split("/", 2);
        String bucket = parts[0];
        String object = parts.length > 1 ? parts[1] : "";

        LOG.info("Loading model from MinIO: bucket={}, object={}", bucket, object);

        return minioClient.getObject(
                GetObjectArgs.builder()
                        .bucket(bucket)
                        .object(object)
                        .build()
        );
    }

    @Override
    public String getModelPath() {
        return config.getModelPath();
    }
}
