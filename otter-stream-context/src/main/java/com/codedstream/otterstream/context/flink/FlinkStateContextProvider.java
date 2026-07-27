package com.codedstream.otterstream.context.flink;

import com.codedstream.otterstream.context.spi.ContextProvider;
import java.util.Map;
import org.apache.flink.api.common.state.ValueState;

/**
 * {@link ContextProvider} backed by Flink's own keyed state — continuously-maintained context
 * (a rolling risk score, a customer profile kept current by a side stream) available at
 * inference time with no external lookup at all, since Flink already has it in local state.
 * This is the "no expensive joins at inference time" pattern: something else keeps this state
 * current as updates stream in, and inference just reads what's already there.
 *
 * <h2>This is architecturally different from every other provider in this module — read this first</h2>
 * Every other {@code ContextProvider} (Redis, Memcached, Pinecone, conversation memory) is a
 * free-standing object you can construct once and call {@code fetch(key, ...)} on for any key,
 * from any thread, any time. <b>This one is not.</b> Flink's {@link ValueState} is scoped to
 * whatever key the enclosing keyed operator is currently processing — there is no "look up a
 * different entity's state" operation, by design, in Flink's state model. Concretely:
 *
 * <ul>
 *   <li>You can only construct this from inside a {@code RichFunction}'s {@code open()}, via
 *       {@code getRuntimeContext().getState(descriptor)} — never as a standalone object.</li>
 *   <li>You can only call {@link #fetch}/{@link #update} from inside that same function's
 *       per-record processing (e.g. {@code processElement}, {@code asyncInvoke}) — never from
 *       another thread or outside a record's processing scope.</li>
 *   <li>The {@code key} parameter to {@link #fetch} is <b>advisory only</b> — Flink's state
 *       always reflects whatever key is implicitly current for the record being processed, not
 *       the string you pass in. If your {@link com.codedstream.otterstream.context.ContextEngine}
 *       assembly key doesn't match Flink's actual current key for that record, you will silently
 *       read the wrong entity's state (or none) — this is a correctness requirement of your
 *       operator's keying, not something this class can validate for you.</li>
 * </ul>
 *
 * <p>Given those constraints, this is normally used <em>outside</em> {@code ContextEngine}'s
 * parallel multi-provider fan-out (which assumes providers are freely callable from a pool
 * thread) — read it directly in your keyed function and merge it into the request map you pass
 * to {@code ContextEngine.assemble}, rather than registering it as one of the engine's own
 * parallel providers. It still implements {@link ContextProvider} so it composes with the same
 * {@link com.codedstream.otterstream.context.Context}/{@code flatten()} shape as everything else.
 *
 * <h2>Usage (inside your own RichFunction)</h2>
 * <pre>{@code
 * public class FraudDetectionFunction extends KeyedProcessFunction<String, Transaction, Decision> {
 *     private transient FlinkStateContextProvider stateContext;
 *
 *     public void open(Configuration parameters) {
 *         ValueState<Map<String, Object>> state = getRuntimeContext().getState(
 *                 new ValueStateDescriptor<>("customer-context", Types.MAP(Types.STRING, Types.GENERIC(Object.class))));
 *         this.stateContext = new FlinkStateContextProvider("flink-state", state);
 *     }
 *
 *     public void processElement(Transaction txn, Context ctx, Collector<Decision> out) throws Exception {
 *         Map<String, Object> customerContext = stateContext.fetch(txn.getCustomerId(), Map.of());
 *         // ... merge into the request map passed to ContextEngine.assemble, or use directly ...
 *     }
 * }
 * }</pre>
 *
 * @since 0.1.0
 */
public class FlinkStateContextProvider implements ContextProvider {

    private final String providerId;
    private final ValueState<Map<String, Object>> state;

    /**
     * @param providerId a stable identifier for this provider
     * @param state      a {@link ValueState} obtained from {@code RuntimeContext.getState(...)}
     *                   inside your function's {@code open()} — see class Javadoc
     */
    public FlinkStateContextProvider(String providerId, ValueState<Map<String, Object>> state) {
        this.providerId = providerId;
        this.state = state;
    }

    @Override
    public String getProviderId() {
        return providerId;
    }

    /**
     * Returns the current record's keyed state as context — {@code key} is advisory only, see
     * class Javadoc. Must be called from within the enclosing function's per-record processing.
     */
    @Override
    public Map<String, Object> fetch(String key, Map<String, Object> request) throws Exception {
        Map<String, Object> value = state.value();
        return value != null ? value : Map.of();
    }

    /**
     * Updates the current record's keyed state — typically called from a side stream that keeps
     * context current (e.g. a customer-profile-updates stream), not from the same path that
     * reads it. Must be called from within the enclosing function's per-record processing.
     */
    public void update(Map<String, Object> newContext) throws Exception {
        state.update(newContext);
    }
}
