package com.codedstreams.otterstreams.sql.async;

import org.apache.flink.streaming.api.functions.async.ResultFuture;
import java.util.Collections;

/**
 * Handles async inference results.
 *
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class AsyncResultHandler<T> {

    public void handleSuccess(ResultFuture<T> resultFuture, T result) {
        resultFuture.complete(Collections.singleton(result));
    }

    public void handleFailure(ResultFuture<T> resultFuture, Throwable error) {
        resultFuture.completeExceptionally(error);
    }

    public void handleTimeout(ResultFuture<T> resultFuture) {
        resultFuture.completeExceptionally(
                new RuntimeException("Inference timeout"));
    }
}
