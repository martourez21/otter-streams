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

public class FraudDetectionExample {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Generate sample transaction data
        DataStream<Map<String, Object>> transactions = env.addSource(new TransactionSource());

        // Configure ML inference
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

        // Create async inference function
        AsyncModelInferenceFunction<Map<String, Object>, InferenceResult> inferenceFunction =
                new AsyncModelInferenceFunction<>(
                        inferenceConfig,
                        cfg -> (InferenceEngine<?>) new OnnxInferenceEngine()
                );

        // Apply async inference
        DataStream<InferenceResult> predictions = AsyncDataStream.unorderedWait(
                transactions,
                inferenceFunction,
                5000,
                TimeUnit.MILLISECONDS,
                100
        );

        // Process results
        predictions
                .filter(result -> result.isSuccess())
                .map(result -> {
                    float fraudProbability = result.getOutput("fraud_probability");
                    String transactionId = (String) result.getOutput("transaction_id");

                    if (fraudProbability > 0.9) {
                        return "🚨 HIGH RISK - Transaction: " + transactionId + " - Probability: " + fraudProbability;
                    } else if (fraudProbability > 0.7) {
                        return "⚠️ MEDIUM RISK - Transaction: " + transactionId + " - Probability: " + fraudProbability;
                    } else {
                        return "✅ LOW RISK - Transaction: " + transactionId + " - Probability: " + fraudProbability;
                    }
                })
                .print();

        env.execute("Fraud Detection Pipeline");
    }

    private static class TransactionSource implements SourceFunction<Map<String, Object>> {
        private volatile boolean running = true;
        private Random random = new Random();
        private int transactionId = 0;

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

                Thread.sleep(100); // Generate 10 transactions per second
            }
        }

        @Override
        public void cancel() {
            running = false;
        }
    }
}
