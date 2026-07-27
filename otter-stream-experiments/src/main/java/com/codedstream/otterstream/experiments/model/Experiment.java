package com.codedstream.otterstream.experiments.model;

import java.util.Objects;

/**
 * A named A/B test between a model's current primary ("control") and a candidate version
 * ("variant"), running via {@code OtterRuntime}'s existing canary mechanism
 * ({@code LifecycleManager.deployCanary}, Milestone 6) — this class adds naming, lifecycle
 * tracking, and (via {@link com.codedstream.otterstream.experiments.ExperimentManager}) outcome
 * recording and statistical comparison on top of that existing traffic-split machinery. It does
 * not reimplement traffic routing.
 *
 * @param experimentId        stable unique id
 * @param name                human-readable name, e.g. {@code "fraud-model-v3.3-rollout"}
 * @param modelId              the model this experiment runs against
 * @param controlVersion       the version serving as primary when the experiment started
 * @param variantVersion       the candidate version deployed as canary
 * @param variantTrafficPercent percentage of traffic routed to the variant (mirrors the canary's trafficPercent)
 * @param status               current lifecycle status
 * @param startedAtMillis      when the experiment was started
 * @param endedAtMillis        when the experiment was concluded, or -1 if still running
 * @since 0.1.0
 */
public record Experiment(
        String experimentId,
        String name,
        String modelId,
        String controlVersion,
        String variantVersion,
        int variantTrafficPercent,
        ExperimentStatus status,
        long startedAtMillis,
        long endedAtMillis) {

    public Experiment {
        Objects.requireNonNull(experimentId, "experimentId cannot be null");
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(modelId, "modelId cannot be null");
        Objects.requireNonNull(controlVersion, "controlVersion cannot be null");
        Objects.requireNonNull(variantVersion, "variantVersion cannot be null");
        Objects.requireNonNull(status, "status cannot be null");
        if (variantTrafficPercent < 0 || variantTrafficPercent > 100) {
            throw new IllegalArgumentException("variantTrafficPercent must be 0-100, was " + variantTrafficPercent);
        }
    }

    public Experiment withStatus(ExperimentStatus newStatus, long endedAtMillis) {
        return new Experiment(experimentId, name, modelId, controlVersion, variantVersion,
                variantTrafficPercent, newStatus, startedAtMillis, endedAtMillis);
    }

    public Experiment withTrafficPercent(int newTrafficPercent) {
        return new Experiment(experimentId, name, modelId, controlVersion, variantVersion,
                newTrafficPercent, status, startedAtMillis, endedAtMillis);
    }
}
