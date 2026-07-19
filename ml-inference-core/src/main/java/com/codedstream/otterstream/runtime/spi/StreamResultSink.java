package com.codedstream.otterstream.runtime.spi;

/**
 * Extension point for publishing inference results (or rule-engine decisions) to a configured
 * downstream stream/messaging system. {@code otter-stream-kafka}'s
 * {@code InferenceResultKafkaSerializationSchema}/{@code OtterKafkaSinks} cover Kafka via
 * Flink's native {@code KafkaSink}; this interface is the generic shape for anything else
 * (Kinesis, Pulsar, a webhook, an internal event bus) a project wants to wire up the same way.
 *
 * <p>Deliberately minimal — a single {@code send} method — since the actual delivery mechanics
 * (batching, retries, partitioning, delivery guarantees) are provider-specific and better
 * expressed through each target system's own idiomatic client/Flink connector (as
 * {@code otter-stream-kafka} does by wrapping Flink's real {@code KafkaSink} rather than
 * reimplementing delivery semantics). This interface exists so a project can depend on one
 * stable type when its downstream target is pluggable/configurable, not to replace
 * system-specific sink implementations.
 *
 * @param <T> the type being published, typically {@code InferenceResult} or a rule-engine
 *            {@code Decision}
 * @since 0.1.0
 */
public interface StreamResultSink<T> {

    /** A stable identifier, e.g. {@code "kafka"}, {@code "kinesis"}, {@code "webhook"}. */
    String getSinkId();

    /**
     * Publishes one value. Implementations decide their own delivery/consistency guarantees;
     * callers should treat this as best-effort unless a specific implementation documents
     * stronger guarantees.
     *
     * @param key   a routing/partitioning key, may be null if the target system doesn't use one
     * @param value the value to publish
     * @throws Exception if publishing fails
     */
    void send(String key, T value) throws Exception;

    /** Releases any resources (connections, producers) held by this sink. */
    default void close() throws Exception {
    }
}
