package com.codedstreams.otterstreams.sql.loader;

import com.codedstreams.otterstreams.sql.config.ModelSourceConfig;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Loads models from HTTP/HTTPS endpoints.
 */
public class HttpModelLoader implements ModelLoader {
    private static final Logger LOG = LoggerFactory.getLogger(HttpModelLoader.class);
    private final ModelSourceConfig config;
    private final OkHttpClient httpClient;

    public HttpModelLoader(ModelSourceConfig config) {
        this.config = config;
        this.httpClient = createHttpClient();
    }

    private OkHttpClient createHttpClient() {
        return new OkHttpClient.Builder()
                .connectTimeout(config.getConnectionTimeoutMs(), TimeUnit.MILLISECONDS)
                .readTimeout(config.getReadTimeoutMs(), TimeUnit.MILLISECONDS)
                .build();
    }

    @Override
    public InputStream loadModel() throws Exception {
        String url = config.getModelPath();
        LOG.info("Loading model from HTTP: {}", url);

        Request.Builder builder = new Request.Builder().url(url);

        // Add authentication
        if (config.getAuthToken() != null) {
            builder.addHeader("Authorization", config.getAuthToken());
        }

        // Add custom headers
        for (Map.Entry<String, String> header : config.getHeaders().entrySet()) {
            builder.addHeader(header.getKey(), header.getValue());
        }

        Request request = builder.build();
        Response response = httpClient.newCall(request).execute();

        if (!response.isSuccessful()) {
            throw new RuntimeException("HTTP error: " + response.code());
        }

        return response.body().byteStream();
    }

    @Override
    public String getModelPath() {
        return config.getModelPath();
    }
}
