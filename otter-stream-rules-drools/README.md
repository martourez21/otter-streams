# otter-stream-rules-drools

Embedded Drools `DecisionEngineConnector` for `otter-stream-rules` — see that module's README
first for the overall Rule Engine design.

**Only add this dependency if you specifically need in-process Drools.** If your organization
runs Drools/Red Hat Decision Manager behind a KIE Server REST endpoint (the common enterprise
deployment shape), use `RestDecisionEngineConnector` in `otter-stream-rules` instead — it needs
no Drools dependency at all. This module exists specifically for the embedded case, and is kept
separate so its dependency tree (`drools-core`, `drools-compiler`, `kie-api`) never lands on a
project that doesn't need it.

## Usage

```java
DecisionEngineConnector connector = new DroolsDecisionEngineConnector(
        "fraud-drools", List.of("rules/fraud-detection.drl"));

Decision decision = connector.evaluate(inferenceResult, Map.of());
```

Your DRL rules operate on an `InferenceFact` (wrapping `modelId`/`outputs`/`context`) and should
`insert()` a `RuleDecision` as their conclusion — see
[`DroolsDecisionEngineConnector`](src/main/java/com/codedstream/otterstream/rules/drools/DroolsDecisionEngineConnector.java)'s
Javadoc for a complete example rule and the full integration pattern.

## A note on verification

This connector is written against the standard, well-documented KIE API
(`KieServices`/`KieFileSystem`/`KieBuilder`/`KieContainer`/`KieSession`) but — like the rest of
this project's Maven-dependent code in this environment — has not been compiled against real
Drools jars here (no Maven Central network access in the authoring environment). Validate it
against your actual Drools version (`drools.version` in this module's `pom.xml`) before relying
on it in production; treat it as a solid, idiomatic starting point rather than pre-verified code.
