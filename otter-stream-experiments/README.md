# otter-stream-experiments

A/B testing on top of `OtterRuntime`'s existing canary mechanism (Milestone 6) — this module
does not route traffic itself, it names, tracks, and statistically evaluates experiments that
use the canary routing `LifecycleManager` already does.

## Install

```xml
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-experiments</artifactId>
    <version>0.0.4</version>
</dependency>
```

Depends only on `ml-inference-core`. No math library — `StatisticalTest` implements Welch's
t-test and a two-proportion z-test itself (see its Javadoc for the honesty note on the p-value
approximation used).

## Quick start

```java
ExperimentManager experiments = new ExperimentManager(runtime);

Experiment exp = experiments.startExperiment(
        "fraud-model-v3.3-rollout", variantConfig, 10); // 10% of traffic to the variant

// In your pipeline, after each inference + rule decision:
experiments.recordOutcome(exp.experimentId(), group, decision.confidence());

// Later — check significance and decide:
var comparison = experiments.compareContinuous(exp.experimentId());
if (comparison.significantAt95() && comparison.variantMean() > comparison.controlMean()) {
    experiments.concludePromote(exp.experimentId());
} else {
    experiments.concludeRollback(exp.experimentId());
}
```

## Two comparison modes

- **`compareContinuous`** — Welch's t-test, for real-valued metrics (confidence, latency,
  any measurement).
- **`compareBinary`** — two-proportion z-test, for conversion/flag-rate style metrics. Pick a
  threshold; every recorded value at or above it counts as "positive."

## What this doesn't do

- **Doesn't tell you which group served a request.** `ManagedModel`'s canary routing decision
  happens inside `infer()` and isn't exposed per-call today — you need your own way to know
  which group produced a given result (tag it yourself, or compare the served model version
  against `Experiment.controlVersion()`/`variantVersion()`). See `ExperimentManager`'s class
  Javadoc for the full reasoning — this was a deliberate choice to avoid changing
  `ManagedModel`'s public contract as a side effect of adding this module.
- **Doesn't compute exact t-distribution p-values.** Uses a normal approximation, accurate for
  moderate-to-large samples (~30+ per group). See `StatisticalTest`'s Javadoc.
- **One experiment per model at a time.** Starting a second experiment for a model that already
  has one running throws `ExperimentException` rather than silently stacking canaries.

## Verification status

Not compiled in this environment (no Maven Central access here — same limitation as every other
Java module in this project). Reviewed carefully by hand: method signatures against
`OtterRuntime`/`ManagedModel`'s actual current API, checked-exception propagation
(`DeploymentException`), and the statistics math worked through by hand against known test
cases (e.g. `welchTTest` on two identical samples returns `t=0, p=1`).
