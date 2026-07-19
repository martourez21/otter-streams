# otter-stream-rules

Post-inference decision/rule engine for Otter Stream: turn an `InferenceResult` into a `Decision`
(a flag like `FRAUD`, `APPROVE`, `REVIEW`, plus which rule(s) fired) using rules you configure —
YAML by default, `.properties` as an alternative, or a hand-written class for fully programmatic
rules. This is deliberately **not** a general-purpose scripting engine (see
[`ExpressionEvaluator`](src/main/java/com/codedstream/otterstream/rules/expr/ExpressionEvaluator.java)'s
Javadoc for why) — just enough boolean logic to express real decision rules safely and fast.

## Install

```xml
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-rules</artifactId>
    <version>0.0.4</version>
</dependency>
```

Pulls in only `ml-inference-core` + `snakeyaml` — no heavy rule-engine runtime. If you need to
embed Drools directly, add `otter-stream-rules-drools` as well (kept separate specifically so
its KIE dependency tree never lands on your classpath unless you ask for it).

## Quick start (default: YAML)

```java
RuleEngine engine = new DefaultRuleEngine(
        YamlRuleSetSource.fromClasspath("rules-example.yaml"));

InferenceResult result = runtime.infer("fraud-detector", inputs);
Decision decision = engine.evaluate(result, Map.of());

if (decision.isFlagged()) {
    System.out.println(decision.flag() + " via rule(s) " + decision.matchedRuleIds());
}
```

See [`rules-example.yaml`](src/main/resources/rules-example.yaml) for the full rule format.

## The other two configuration paths

**Properties file** (functionally identical, flat-file instead of nested YAML):

```java
RuleEngine engine = new DefaultRuleEngine(
        PropertiesRuleSetSource.fromClasspath("rules.properties"));
```

**A hand-written class** — implement `RuleSetSource` directly and build the `RuleSet` in code;
there's no separate "programmatic" SPI beyond this one:

```java
RuleSetSource source = () -> new RuleSet(
        "fraud-rules", "1", RuleEvaluationMode.SINGLE,
        List.of(new Rule("high-risk", "High Risk", 100,
                "output.risk_score > 0.85", "FRAUD", "HIGH_RISK", true, "#c0392b", null)));
RuleEngine engine = new DefaultRuleEngine(source);
```

Or skip `DefaultRuleEngine` entirely and implement the `RuleEngine` interface yourself if you
need logic `DefaultRuleEngine` doesn't offer — it's an interface precisely so this is possible.

## Single, multiple, or batch flagging

- **Single** (`RuleEvaluationMode.SINGLE` on the `RuleSet`): evaluation stops at the first
  (highest-priority) matching rule. `Decision.matchedRuleIds()` has at most one entry.
- **Multiple** (`RuleEvaluationMode.MULTIPLE`): every enabled rule is evaluated;
  `matchedRuleIds()` can have several entries (e.g. flagged as both `HIGH_RISK` and
  `NEW_ACCOUNT`), with `flag()` set from the highest-priority match.
- **Batch**: call `evaluateBatch(List<InferenceResult>, context)` instead of `evaluate(...)` in
  a loop. Below 64 items it evaluates sequentially; above that it parallelizes automatically
  (see `DefaultRuleEngine`'s class Javadoc for why 64, and why that's safe).

## Metrics (for the rule dashboard)

```java
RuleMetricsSnapshot snapshot = engine.getMetrics();
snapshot.hitsByRuleId();  // Map<String, Long> — per-rule hit counts
snapshot.hitsByFlag();    // Map<String, Long> — per-flag counts (FRAUD, REVIEW, APPROVE, ...)
```

This is what the Rule Dashboard (Otter Control Plane) polls/receives to render per-rule and
per-flag charts, using each rule's configured `color` for consistent, professional-looking
charts rather than arbitrary auto-assigned colors.

## Connecting to an external enterprise decision engine

Instead of evaluating in-process, delegate to KIE Server (Red Hat Decision Manager), Camunda
DMN, IBM ODM, or any in-house REST decision service:

```java
DecisionEngineConnector connector = new RestDecisionEngineConnector(
        "kie-server",
        URI.create("https://decisions.mycompany.com/kie-server/decision"),
        System.getenv("KIE_SERVER_TOKEN"));

Decision decision = connector.evaluate(result, Map.of());
```

One connector, configuration-driven, works against any of them — see
[`DecisionEngineConnector`](src/main/java/com/codedstream/otterstream/rules/spi/DecisionEngineConnector.java)'s
Javadoc for why this project doesn't ship a bespoke SDK integration per vendor. For embedding
Drools directly (not over REST), see `otter-stream-rules-drools`.

## Composing with `OtterRuntime`

This module has no dependency on (or hook into) `OtterRuntime` — it just consumes
`InferenceResult`, which is already a stable, public type. Compose them in your own pipeline
code:

```java
InferenceResult result = runtime.infer(modelId, inputs);
Decision decision = ruleEngine.evaluate(result, Map.of("region", "US"));
// ... route `decision` downstream (Kafka sink, database, alert, etc.)
```

This is a deliberate choice, not an oversight: keeping `ml-inference-core` free of a dependency
on the rule engine (and vice versa) means each can evolve independently, and a project that
doesn't need rules pays zero cost for this module existing.
