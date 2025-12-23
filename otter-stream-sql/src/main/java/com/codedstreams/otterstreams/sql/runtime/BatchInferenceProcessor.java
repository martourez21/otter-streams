package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Batches inference requests for improved throughput.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class BatchInferenceProcessor {
    private static final Logger LOG = LoggerFactory.getLogger(BatchInferenceProcessor.class);

    private final InferenceEngine<?> engine;
    private final int batchSize;
    private final long batchTimeoutMs;
    private final BlockingQueue<InferenceRequest> requestQueue;
    private final Thread processingThread;
    private volatile boolean running = false;

    public BatchInferenceProcessor(InferenceEngine<?> engine, int batchSize, long batchTimeoutMs) {
        this.engine = engine;
        this.batchSize = batchSize;
        this.batchTimeoutMs = batchTimeoutMs;
        this.requestQueue = new ArrayBlockingQueue<>(batchSize * 10);
        this.processingThread = new Thread(this::processLoop, "batch-processor");
    }

    public void start() {
        if (!running) {
            running = true;
            processingThread.start();
            LOG.info("Batch processor started: batchSize={}, timeout={}ms", batchSize, batchTimeoutMs);
        }
    }

    public void stop() {
        running = false;
        processingThread.interrupt();
        LOG.info("Batch processor stopped");
    }

    /**
     * Submits an inference request and waits for result.
     */
    public InferenceResult submitAndWait(Map<String, Object> features) throws InterruptedException {
        InferenceRequest request = new InferenceRequest(features);
        requestQueue.put(request);
        return request.waitForResult();
    }

    private void processLoop() {
        List<InferenceRequest> batch = new ArrayList<>(batchSize);
        long lastBatchTime = System.currentTimeMillis();

        while (running) {
            try {
                // Collect batch
                InferenceRequest request = requestQueue.poll(batchTimeoutMs, TimeUnit.MILLISECONDS);

                if (request != null) {
                    batch.add(request);
                }

                // Process batch if full or timeout reached
                long now = System.currentTimeMillis();
                boolean timeoutReached = (now - lastBatchTime) >= batchTimeoutMs;
                boolean batchFull = batch.size() >= batchSize;

                if (!batch.isEmpty() && (batchFull || timeoutReached)) {
                    processBatch(batch);
                    batch.clear();
                    lastBatchTime = now;
                }
            } catch (InterruptedException e) {
                if (!running) break;
            } catch (Exception e) {
                LOG.error("Error in batch processing loop", e);
            }
        }

        // Process remaining requests
        if (!batch.isEmpty()) {
            processBatch(batch);
        }
    }

    @SuppressWarnings("unchecked")
    private void processBatch(List<InferenceRequest> batch) {
        try {
            LOG.debug("Processing batch of {} requests", batch.size());

            // Convert to array
            Map<String, Object>[] inputArray = batch.stream()
                    .map(InferenceRequest::getFeatures)
                    .toArray(Map[]::new);

            // Batch inference
            InferenceResult batchResult = engine.inferBatch(inputArray);

            // Distribute results (simplified - assumes each request gets same result)
            for (InferenceRequest request : batch) {
                request.setResult(batchResult);
            }
        } catch (Exception e) {
            LOG.error("Batch inference failed", e);
            for (InferenceRequest request : batch) {
                request.setError(e);
            }
        }
    }

    /**
     * Represents a single inference request in the batch queue.
     */
    private static class InferenceRequest {
        private final Map<String, Object> features;
        private volatile InferenceResult result;
        private volatile Exception error;
        private volatile boolean complete = false;

        InferenceRequest(Map<String, Object> features) {
            this.features = features;
        }

        Map<String, Object> getFeatures() {
            return features;
        }

        void setResult(InferenceResult result) {
            this.result = result;
            this.complete = true;
            synchronized (this) {
                notifyAll();
            }
        }

        void setError(Exception error) {
            this.error = error;
            this.complete = true;
            synchronized (this) {
                notifyAll();
            }
        }

        InferenceResult waitForResult() throws InterruptedException {
            synchronized (this) {
                while (!complete) {
                    wait();
                }
            }
            if (error != null) {
                throw new RuntimeException("Inference failed", error);
            }
            return result;
        }
    }
}
