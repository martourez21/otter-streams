package com.codedstreams.otterstreams.sql.loader;

import okhttp3.ConnectionPool;
import okhttp3.OkHttpClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Manages HTTP connection pooling for remote model servers.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ModelServerConnectionPool {
    private static final Logger LOG = LoggerFactory.getLogger(ModelServerConnectionPool.class);
    private static final ModelServerConnectionPool INSTANCE = new ModelServerConnectionPool();

    private final OkHttpClient httpClient;

    private ModelServerConnectionPool() {
        ConnectionPool connectionPool = new ConnectionPool(
                50,  // max idle connections
                5,   // keep alive duration
                TimeUnit.MINUTES
        );

        this.httpClient = new OkHttpClient.Builder()
                .connectionPool(connectionPool)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .retryOnConnectionFailure(true)
                .build();

        LOG.info("HTTP connection pool initialized");
    }

    public static ModelServerConnectionPool getInstance() {
        return INSTANCE;
    }

    public OkHttpClient getHttpClient() {
        return httpClient;
    }

    public void shutdown() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
        LOG.info("HTTP connection pool shutdown");
    }
}
