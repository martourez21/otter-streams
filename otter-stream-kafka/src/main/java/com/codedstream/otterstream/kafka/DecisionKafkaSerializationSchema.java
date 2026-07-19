package com.codedstream.otterstream.kafka;

import com.codedstream.otterstream.rules.model.Decision;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.header.Headers;

/**
 * Serializes rule-engine {@link Decision}s to JSON for a Flink {@code KafkaSink} — the natural
 * pairing with {@link InferenceResultKafkaSerializationSchema}: publish the raw inference result
 * to one topic and the post-rule-engine decision to another (or the same topic, differently
 * keyed), so downstream consumers (case management, alerting, audit log) don't need to embed
 * rule logic themselves.
 *
 * <p>Wire format:
 * <pre>{@code
 * {
 *   "flag": "FRAUD",
 *   "category": "HIGH_RISK",
 *   "confidence": 0.92,
 *   "matchedRuleIds": ["high-risk-score"],
 *   "timestampMillis": 1737033600000
 * }
 * }</pre>
 *
 * @since 0.1.0
 */
public class DecisionKafkaSerializationSchema implements KafkaRecordSerializationSchema<Decision> {

    private final String topic;
    private final transient ObjectMapper mapper = new ObjectMapper();

    public DecisionKafkaSerializationSchema(String topic) {
        this.topic = topic;
    }

    @Override
    public ProducerRecord<byte[], byte[]> serialize(Decision decision, KafkaSinkContext context, Long timestamp) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("flag", decision.flag());
        payload.put("category", decision.category());
        payload.put("confidence", decision.confidence());
        payload.put("matchedRuleIds", decision.matchedRuleIds());
        payload.put("timestampMillis", decision.timestampMillis());

        byte[] key = decision.flag() != null ? decision.flag().getBytes(StandardCharsets.UTF_8) : null;
        byte[] value = toJsonBytes(payload);
        return new ProducerRecord<>(topic, null, timestamp, key, value, (Headers) null);
    }

    private byte[] toJsonBytes(Map<String, Object> payload) {
        try {
            return mapper.writeValueAsBytes(payload);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to serialize Decision to JSON", e);
        }
    }
}
