package com.codedstreams.otterstream.examples;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.function.AsyncModelInferenceFunction;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.onnx.OnnxInferenceEngine;
import org.apache.flink.streaming.api.datastream.AsyncDataStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;

import java.time.Duration;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.Random;

/**
 * Example demonstrating real-time fraud detection using OtterStream's ML inference capabilities.
 *
 * <p>This example shows how to integrate machine learning models into Apache Flink streaming
 * pipelines for real-time fraud detection on financial transactions. It demonstrates:
 *
 * <ul>
 *   <li>Integration of ONNX models with Flink's async I/O</li>
 *   <li>Real-time inference on streaming transaction data</li>
 *   <li>Risk classification based on model predictions</li>
 *   <li>Configuration of inference parameters for production use</li>
 * </ul>
 *
 * <h2>Pipeline Architecture:</h2>
 * <pre>
 * Transaction Source → Async Inference → Risk Classification → Output
 *      ↓                     ↓                  ↓
 *   Synthetic    Fraud Detection     🚨/⚠️/✅ Labels
 *   Transactions    Model
 * </pre>
 *
 * <h2>Key Features Demonstrated:</h2>
 * <ol>
 *   <li><b>Async Model Inference:</b> Non-blocking ML inference using {@link AsyncModelInferenceFunction}</li>
 *   <li><b>ONNX Integration:</b> Loading and running ONNX models via {@link OnnxInferenceEngine}</li>
 *   <li><b>Stream Processing:</b> Real-time processing with configurable timeout and retry logic</li>
 *   <li><b>Risk Categorization:</b> Three-tier risk classification (HIGH/MEDIUM/LOW)</li>
 * </ol>
 *
 * <h2>Transaction Features:</h2>
 * <p>The synthetic transaction generator creates realistic transaction data with features including:
 * <ul>
 *   <li>Transaction amount (0-1000)</li>
 *   <li>Time of day (0-23 hours)</li>
 *   <li>Geographic location (10 regions)</li>
 *   <li>Merchant category (20 categories)</li>
 *   <li>Customer history (chargebacks, account age)</li>
 *   <li>Device type (5 device categories)</li>
 * </ul>
 *
 * <h2>Running the Example:</h2>
 * <pre>{@code
 * // 1. Place your ONNX model at: models/fraud_detection.onnx
 * // 2. Run the example:
 * mvn exec:java -Dexec.mainClass="com.codedstreams.otterstream.examples.FraudDetectionExample"
 * }</pre>
 *
 * <h2>Expected Output:</h2>
 * <pre>
 * 🚨 HIGH RISK - Transaction: txn_42 - Probability: 0.95
 * ✅ LOW RISK - Transaction: txn_43 - Probability: 0.23
 * ⚠️ MEDIUM RISK - Transaction: txn_44 - Probability: 0.78
 * </pre>
 *
 * <h2>Performance Considerations:</h2>
 * <ul>
 *   <li><b>Async I/O:</b> Uses Flink's async operators to prevent blocking on model inference</li>
 *   <li><b>Batch Size:</b> Configurable batch processing (default: 32) for throughput optimization</li>
 *   <li><b>Timeout:</b> 5-second inference timeout prevents pipeline stalls</li>
 *   <li><b>Retries:</b> Automatic retry (3 attempts) for transient failures</li>
 * </ul>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see AsyncModelInferenceFunction
 * @see OnnxInferenceEngine
 * @see InferenceConfig
 */
public class FraudDetectionExample {

    /**
     * Main entry point for the fraud detection pipeline.
     * <p>Sets up and executes the complete streaming pipeline with ML inference.
     *
     * @param args command line arguments (not used)
     * @throws Exception if pipeline execution fails
     */
    public static void main(String[] args) throws Exception {
        // Initialize Flink streaming environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Generate sample transaction data
        DataStream<Map<String, Object>> transactions = env.addSource(new TransactionSource());

        // Configure ML inference for fraud detection
        InferenceConfig inferenceConfig = InferenceConfig.builder()
                .modelConfig(ModelConfig.builder()
                        .modelId("fraud-detection")
                        .modelPath("models/fraud_detection.onnx")
                        .format(ModelFormat.ONNX)
                        .modelName("fraud_model")
                        .modelVersion("1.0")
                        .build())
                .batchSize(32)
                .timeout(Duration.ofMillis(5000))
                .maxRetries(3)
                .enableMetrics(true)
                .build();

        // Create async inference function with ONNX engine
        AsyncModelInferenceFunction<Map<String, Object>, InferenceResult> inferenceFunction =
                new AsyncModelInferenceFunction<>(
                        inferenceConfig,
                        cfg -> (InferenceEngine<?>) new OnnxInferenceEngine()
                );

        // Apply async inference with unordered processing for maximum throughput
        DataStream<InferenceResult> predictions = AsyncDataStream.unorderedWait(
                transactions,
                inferenceFunction,
                5000,                    // timeout in milliseconds
                TimeUnit.MILLISECONDS,   // timeout unit
                100                      // max concurrent async requests
        );

        // Process and classify inference results
        predictions
                .filter(result -> result.isSuccess())
                .map(result -> {
                    // Extract fraud probability from model output
                    float fraudProbability = result.getOutput("fraud_probability");
                    String transactionId = (String) result.getOutput("transaction_id");

                    // Classify risk based on probability thresholds
                    if (fraudProbability > 0.9) {
                        return "🚨 HIGH RISK - Transaction: " + transactionId + " - Probability: " + fraudProbability;
                    } else if (fraudProbability > 0.7) {
                        return "⚠️ MEDIUM RISK - Transaction: " + transactionId + " - Probability: " + fraudProbability;
                    } else {
                        return "✅ LOW RISK - Transaction: " + transactionId + " - Probability: " + fraudProbability;
                    }
                })
                .print();  // In production, replace with sink to database/Kafka/alerting system

        // Execute the pipeline
        env.execute("Fraud Detection Pipeline");
    }

    /**
     * Synthetic transaction data source for demonstration purposes.
     * <p>Generates realistic transaction data at a rate of 10 transactions per second.
     * In production, this would be replaced with a real data source (Kafka, Kinesis, etc.).
     *
     * <h2>Generated Features:</h2>
     * <ul>
     *   <li><b>transaction_id:</b> Unique identifier for each transaction</li>
     *   <li><b>amount:</b> Transaction amount (0-1000)</li>
     *   <li><b>time_of_day:</b> Hour of day (0-23)</li>
     *   <li><b>location:</b> Geographic region (0-9)</li>
     *   <li><b>merchant_category:</b> Merchant type (0-19)</li>
     *   <li><b>previous_chargebacks:</b> Customer chargeback history (0-2)</li>
     *   <li><b>account_age_days:</b> Customer account age (0-999 days)</li>
     *   <li><b>device_type:</b> Device used for transaction (0-4)</li>
     * </ul>
     *
     * @see SourceFunction
     */
    private static class TransactionSource implements SourceFunction<Map<String, Object>> {
        private volatile boolean running = true;
        private Random random = new Random();
        private int transactionId = 0;

        /**
         * Continuously generates synthetic transaction data.
         *
         * @param ctx source context for emitting transactions
         * @throws Exception if data generation fails
         */
        @Override
        public void run(SourceContext<Map<String, Object>> ctx) throws Exception {
            while (running) {
                Map<String, Object> transaction = new HashMap<>();
                transaction.put("transaction_id", "txn_" + (++transactionId));
                transaction.put("amount", random.nextFloat() * 1000);
                transaction.put("time_of_day", random.nextInt(24));
                transaction.put("location", random.nextInt(10));
                transaction.put("merchant_category", random.nextInt(20));
                transaction.put("previous_chargebacks", random.nextInt(3));
                transaction.put("account_age_days", random.nextInt(1000));
                transaction.put("device_type", random.nextInt(5));

                ctx.collect(transaction);

                // Simulate 10 transactions per second
                Thread.sleep(100);
            }
        }

        /**
         * Cancels the data generation.
         * <p>Called by Flink when the job is cancelled.
         */
        @Override
        public void cancel() {
            running = false;
        }
    }
}