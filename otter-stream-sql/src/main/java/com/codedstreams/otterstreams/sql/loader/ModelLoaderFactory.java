package com.codedstreams.otterstreams.sql.loader;

import com.codedstreams.otterstreams.sql.config.ModelSourceConfig;

/**
 * Factory for creating appropriate model loaders based on source type.
 */
public class ModelLoaderFactory {

    public static ModelLoader create(ModelSourceConfig config) {
        switch (config.getSourceType()) {
            case S3:
            case MINIO:
                return new S3ModelLoader(config);
            case HTTP:
                return new HttpModelLoader(config);
            case HDFS:
            case LOCAL:
            default:
                return new LocalModelLoader(config);
        }
    }
}
