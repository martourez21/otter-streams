# Performance & Concurrency Review

Scope: a targeted review of the hot inference path against two goals — sub-5ms best-case
inference latency, and handling large numbers of concurrent requests — not an exhaustive
audit of every module. Findings below are real, specific, and either fixed or explicitly
flagged as follow-ups; nothing here is a generic "looks fine" pass.

## Fixed in this pass

### 1. `AsyncModelInferenceFunction` was using the shared JVM-wide `ForkJoinPool.commonPool()`

**Before:** `CompletableFuture.supplyAsync(() -> inferenceEngine.infer(...))` with no explicit
executor. Java silently runs this on `ForkJoinPool.commonPool()` — sized to
`Runtime.availableProcessors() - 1`, shared by *everything else in the JVM* using unqualified
`CompletableFuture`/parallel streams. Since `infer()` blocks the calling thread for the model's
full inference duration, one hot model under load could starve unrelated work elsewhere in the
same TaskManager (and vice versa) — a classic hard-to-diagnose latency-spike source under
concurrency, directly at odds with a sub-5ms target.

**Fixed:** a dedicated, named, bounded `ExecutorService` per function instance, sized via a new
constructor parameter (default 32, tune to roughly match your `AsyncDataStream.unorderedWait(...,
capacity)` value), created in `open()` and shut down in `close()`.

### 2. Lazy engine initialization inside `asyncInvoke` raced under concurrent calls

**Before:** `if (inferenceEngine == null || !inferenceEngine.isReady()) initializeEngine();`
executed on every `asyncInvoke` call. Flink's async operator allows multiple in-flight
`asyncInvoke` calls concurrently per subtask (that's what the `capacity` parameter is for) —
so the very first burst of concurrent calls could all observe `inferenceEngine == null`
simultaneously and each call `initializeEngine()`, redundantly (and non-deterministically)
racing to construct/initialize the engine.

**Fixed:** initialization moved into `open(Configuration)`, which Flink guarantees runs exactly
once per parallel subtask, strictly before any `asyncInvoke` call. This removes the race by
construction rather than adding a lock around it — no synchronization needed, and no per-call
`isReady()` check overhead either.

### 3. `TensorFlowGraphDefEngine.infer()` was returning a fabricated constant

Not a performance bug, but found during this review's pass over the hot path: `infer()`
returned `{"output": 0.5}` unconditionally, ignoring the actual input and loaded graph — a
silent-fabrication correctness issue in the same family as one already fixed elsewhere in this
codebase (`GrpcInferenceClient`, which correctly refuses to fabricate and throws instead).
Replaced with a real `Session.Runner` implementation. See the class Javadoc for the
`inputTensorName`/`outputTensorNames` configuration it now requires (GraphDef carries no
signature metadata, unlike SavedModel).

## Reviewed and already sound (no change needed)

- **`ManagedModel.infer()`** (Milestone 5/6 hot path): per-call cost is an `AtomicReference.get()`
  for canary routing, an `AtomicInteger` increment/decrement for in-flight tracking, and — only
  if a shadow is configured — a cheap sample-rate check before an async, fire-and-forget
  `executor.submit()`. No locks, no allocation beyond what the engine call itself needs. This
  was correct when written (Milestones 5/6) and remains so.
- **`LifecycleManager`'s drain-on-retire loop** (10ms polling until in-flight count hits zero or
  a timeout): intentionally *not* on the request hot path — it runs once per deployment/swap,
  not per inference call, so its coarse polling granularity is an acceptable, deliberate
  trade-off against added complexity (condition variables/callbacks), not an oversight.
- **`RuleEngine`'s metrics** (`otter-stream-rules`): `LongAdder` rather than `AtomicLong` for
  hit counters specifically because they're write-heavy (every evaluation) and read-light
  (dashboard polling) — see `DefaultRuleEngine`'s class Javadoc for the reasoning. Condition
  expressions are compiled once at rule-set load time, never re-parsed per evaluation.

## Explicitly not covered by this pass

This was a targeted review of the paths most directly on the inference hot path, not a
line-by-line audit of every module. In particular, not reviewed here:

- The individual provider engines' internal tensor construction/extraction code
  (`otter-stream-onnx`, `-tensorflow`, `-pytorch`, `-xgboost`, `-pmml`) for allocation
  efficiency — a reasonable next pass, but a larger one (six modules, each with its own
  native-library-specific considerations).
- JVM-level GC tuning guidance (heap sizing, GC algorithm choice for low-pause inference
  workloads) — deployment-specific, not something this codebase can prescribe generically.
- Actual load-test numbers. Everything above is a code-level review; none of it has been
  benchmarked in this environment (no running Flink cluster or GPU hardware available here — see
  `otter-control-plane/ARCHITECTURE.md`'s own verification caveats for the broader pattern of
  what could and couldn't be executed in this sandbox). Treat the fixes above as removing known
  latency risks, not as a guarantee of the sub-5ms target in any specific deployment.

## Java 17 usage in this pass

The Rule Engine module (`otter-stream-rules`) leans on Java 17 features where they genuinely
fit: `record` types for all immutable data (`Decision`, `Rule`, `RuleSet`, `Node` AST variants),
a `sealed interface` for the expression AST, and pattern-matching `instanceof`
(`if (!(current instanceof Map<?,?> map))`) in place of manual casts. Pattern-matching `switch`
(which would let `Node`'s AST dispatch read even more directly) is preview-only under this
project's `release=17` compiler setting — flagged in `Node.java`'s class Javadoc as a mechanical
follow-up once/if the baseline moves to 21, rather than reached for now via
`--enable-preview` (which would require every downstream consumer to opt into preview features
too — not a reasonable thing to force on library users).
