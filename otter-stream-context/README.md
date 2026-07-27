# otter-stream-context

An optional Context Engine: assembles context — feature lookups, conversation memory, vector
search, cached state, continuously-maintained Flink state — from multiple providers **in
parallel** before inference. Built for RAG and other context/memory-heavy AI applications on
top of Otter Streams, and built to handle a large number of concurrent users, not just to work
correctly at low volume.

**Entirely opt-in.** `ml-inference-core` has no dependency on this module and no awareness of
it — a project that doesn't need context assembly pays zero cost for this module existing. Pull
it in only for the specific project that needs it.

## Why this exists

In most real pipelines, the model call is not what's slow — assembling the context it needs is.
A fraud-detection or RAG request might need a customer profile, recent transaction history, a
vector search over a document store, and conversation memory, all before the model even runs.
Doing that sequentially, one lookup at a time, is usually where the real latency budget goes.
`ContextEngine` fans all of that out in parallel, on a dedicated thread pool, with a per-provider
timeout — a slow or failing provider degrades that one provider's contribution, not the whole
request.

## Quick start

```java
ContextEngine engine = ContextEngine.builder()
        .provider(new FeatureProviderContextAdapter(redisFeatureProvider)) // reuses existing FeatureProvider work
        .provider(new ConversationMemoryProvider(20, 100_000, Duration.ofHours(2)))
        .provider(pineconeContextProvider)
        .cache(new ContextCache(100_000, 30))
        .parallelism(64)
        .perProviderTimeout(Duration.ofMillis(300))
        .build();

Context context = engine.assemble(userId, Map.of(
        "embedding", queryEmbedding,
        "topK", 5));

Map<String, Object> modelInput = context.flatten(); // namespaced "<providerId>.<key>" -> value
InferenceResult result = runtime.infer("rag-model", modelInput);
```

## What's built, mapped to the "Context Engine" vision

| Capability | Class | Notes |
|---|---|---|
| Generalized provider interface | `ContextProvider` | `FeatureProvider` (Redis/JDBC/Feast) adapts in via `FeatureProviderContextAdapter` — no duplicated client code |
| Parallel assembly, scale-optimized | `ContextEngine` | Dedicated executor (never the shared common pool — see class Javadoc), per-provider timeout, graceful partial-failure |
| Context cache | `ContextCache` | Caffeine-backed, bounded size + TTL |
| Redis cache | *(reuse)* | `otter-stream-feature-redis`'s `RedisFeatureProvider`, adapted |
| **Memcached cache** | `MemcachedContextProvider` | New — zero-dependency raw-socket client, the other major cache alongside Redis |
| Conversation/session memory | `ConversationMemoryProvider` | Bounded per-session *and* across total sessions (Caffeine) — safe under many concurrent users |
| Vector search | `VectorSearchProvider` (interface) + `PineconeContextProvider` | One concrete implementation; implement the interface directly for Milvus/OpenSearch |
| Flink-native streaming state | `FlinkStateContextProvider` | **Read this class's Javadoc before using it** — architecturally different from every other provider here, see below |

## The one provider that works differently: Flink state

Every provider above is a free-standing object — construct it once, call `fetch()` for any key,
from any thread. `FlinkStateContextProvider` is not: Flink's `ValueState` is scoped to whatever
key the enclosing keyed operator currently has, so this one can only be constructed inside a
`RichFunction.open()` and only called from that function's own per-record processing. Read its
class Javadoc in full before reaching for it — using it wrong fails silently (you get the wrong
entity's state, not an exception).

## Built for concurrency and scale — specifics, not a marketing claim

- **Dedicated executor per `ContextEngine`, never `ForkJoinPool.commonPool()`.** This is the
  exact bug fixed in `AsyncModelInferenceFunction` (see `PERFORMANCE.md`) — a blocking provider
  call on the shared pool can starve unrelated work in the same JVM. It matters more here than
  it did there: every inference request now fans out to N provider calls, not one.
- **Per-provider timeout, not a global one.** One slow provider produces one failed
  `ContextResult`, not a stalled request.
- **Every bounded-memory structure is actually bounded.** `ContextCache` and
  `ConversationMemoryProvider` are both Caffeine-backed with hard size caps — a workload with
  high entity/session cardinality (many concurrent users, by definition) can't grow either into
  an OOM the way an unbounded `Map` would.
- **Per-session locking in `ConversationMemoryProvider` is per-key, not global** — concurrent
  turns for *different* sessions never contend with each other.

## What this doesn't do

- **Doesn't make `OtterRuntime` own context assembly.** `ContextEngine` is a separate,
  standalone object you call before `runtime.infer(...)` in your own pipeline code — the same
  composition pattern `otter-stream-rules` and `otter-stream-experiments` use. `OtterRuntime`
  has no `.context(...)` builder method and isn't getting one; that would require the Runtime to
  own entity-id extraction, per-context error policy, and merge order, all of which are
  currently (and deliberately) your pipeline's job, not the Runtime's.
- **`MemcachedContextProvider` is a minimal single-node client** — no connection pooling, no
  cluster-aware consistent hashing, no auth. Fine for "point at one instance," not a replacement
  for a full client library if you need cluster routing.
- **`VectorSearchProvider` ships one concrete implementation (Pinecone).** Milvus and
  OpenSearch aren't bespoke-implemented here — implement the interface directly against their
  APIs, the same "one generic pattern, implement for your specific backend" approach used for
  `otter-stream-rules`'s `DecisionEngineConnector`.
- **Feature versioning/monitoring decorators from `ml-inference-core`'s `runtime.feature`
  package are not automatically wired in here** — compose them yourself if you want a
  `ContextProvider` that's also version-stamped or latency-monitored (wrap the underlying
  `FeatureProvider` with `VersioningFeatureProvider`/`MonitoredFeatureProvider` *before* adapting
  it with `FeatureProviderContextAdapter`).

## Verification status

Not compiled in this environment — same limitation as every other Java module built without
Maven Central access. Reviewed by hand against actual current APIs (`FeatureProvider`,
`ValueState`, Caffeine's builder shape matching `ModelCache`'s established usage). The Memcached
client's text-protocol parsing is the piece most worth testing against a real instance before
trusting it — line-based protocol parsing mixed with raw byte-count payload reads is exactly the
kind of code that looks right and has an off-by-one until you run it.
