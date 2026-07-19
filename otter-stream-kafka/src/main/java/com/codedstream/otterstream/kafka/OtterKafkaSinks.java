package com.codedstream.otterstream.kafka;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.model.Decision;
import java.util.Properties;
import org.apache.flink.connector.base.DeliveryGuarantee;
import org.apache.flink.connector.kafka.sink.KafkaSink;

/**
 * One-liner builders for publishing {@link InferenceResult}s or {@link Decision}s to Kafka,
 * on top of Flink's own {@code KafkaSink} — not a replacement for it. If you need more control
 * (custom partitioning, transactional delivery tuning, etc.), use
 * {@link InferenceResultKafkaSerializationSchema}/{@link DecisionKafkaSerializationSchema}
 * directly with {@code KafkaSink.builder()} yourself; these helpers just remove the boilerplate
 * for the common case.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * KafkaSink<InferenceResult> sink = OtterKafkaSinks.inferenceResultSink(
 *         "broker1:9092,broker2:9092", "fraud-inference-results");
 * resultStream.sinkTo(sink);
 *
 * KafkaSink<Decision> decisionSink = OtterKafkaSinks.decisionSink(
 *         "broker1:9092,broker2:9092", "fraud-decisions");
 * decisionStream.sinkTo(decisionSink);
 * }</pre>
 *
 * @since 0.1.0
 */
public final class OtterKafkaSinks {

    private OtterKafkaSinks() {
    }

    public static KafkaSink<InferenceResult> inferenceResultSink(String bootstrapServers, String topic) {
        return inferenceResultSink(bootstrapServers, topic, DeliveryGuarantee.AT_LEAST_ONCE, new Properties());
    }

    public static KafkaSink<InferenceResult> inferenceResultSink(
            String bootstrapServers, String topic, DeliveryGuarantee deliveryGuarantee, Properties producerConfig) {
        return KafkaSink.<InferenceResult>builder()
                .setBootstrapServers(bootstrapServers)
                .setKafkaProducerConfig(producerConfig)
                .setRecordSerializer(new InferenceResultKafkaSerializationSchema(topic))
                .setDeliveryGuarantee(deliveryGuarantee)
                .build();
    }

    public static KafkaSink<Decision> decisionSink(String bootstrapServers, String topic) {
        return decisionSink(bootstrapServers, topic, DeliveryGuarantee.AT_LEAST_ONCE, new Properties());
    }

    public static KafkaSink<Decision> decisionSink(
            String bootstrapServers, String topic, DeliveryGuarantee deliveryGuarantee, Properties producerConfig) {
        return KafkaSink.<Decision>builder()
                .setBootstrapServers(bootstrapServers)
                .setKafkaProducerConfig(producerConfig)
                .setRecordSerializer(new DecisionKafkaSerializationSchema(topic))
                .setDeliveryGuarantee(deliveryGuarantee)
                .build();
    }
}
