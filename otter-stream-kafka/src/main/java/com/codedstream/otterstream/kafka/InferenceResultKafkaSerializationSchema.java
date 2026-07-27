package com.codedstream.otterstream.kafka;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.header.Headers;

/**
 * Serializes {@link InferenceResult} to JSON for a Flink {@code KafkaSink}. Use directly with
 * Flink's own {@code KafkaSink.builder()}, or via {@link OtterKafkaSinks#inferenceResultSink}
 * for a one-liner.
 *
 * <p>Wire format:
 * <pre>{@code
 * {
 *   "modelId": "fraud-detector",
 *   "success": true,
 *   "inferenceTimeMs": 3,
 *   "outputs": { "risk_score": 0.92, "confidence": 0.88 },
 *   "errorMessage": null
 * }
 * }</pre>
 *
 * @since 0.1.0
 */
public class InferenceResultKafkaSerializationSchema implements KafkaRecordSerializationSchema<InferenceResult> {

    private final String topic;
    private final transient ObjectMapper mapper = new ObjectMapper();

    public InferenceResultKafkaSerializationSchema(String topic) {
        this.topic = topic;
    }

    @Override
    public ProducerRecord<byte[], byte[]> serialize(InferenceResult result, KafkaSinkContext context, Long timestamp) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("modelId", result.getModelId());
        payload.put("success", result.isSuccess());
        payload.put("inferenceTimeMs", result.getInferenceTimeMs());
        payload.put("outputs", result.getOutputs());
        payload.put("errorMessage", result.getErrorMessage());

        byte[] key = result.getModelId() != null ? result.getModelId().getBytes(StandardCharsets.UTF_8) : null;
        byte[] value = toJsonBytes(payload);
        return new ProducerRecord<>(topic, null, timestamp, key, value, (Headers) null);
    }

    private byte[] toJsonBytes(Map<String, Object> payload) {
        try {
            return mapper.writeValueAsBytes(payload);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to serialize InferenceResult to JSON", e);
        }
    }
}
