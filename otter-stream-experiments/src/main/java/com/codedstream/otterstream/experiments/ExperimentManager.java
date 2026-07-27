package com.codedstream.otterstream.experiments;

import com.codedstream.otterstream.experiments.model.Experiment;
import com.codedstream.otterstream.experiments.model.ExperimentGroup;
import com.codedstream.otterstream.experiments.model.ExperimentOutcome;
import com.codedstream.otterstream.experiments.model.ExperimentStatus;
import com.codedstream.otterstream.experiments.stats.StatisticalTest;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.DeploymentException;
import com.codedstream.otterstream.runtime.OtterRuntime;
import com.codedstream.otterstream.runtime.lifecycle.ManagedModel;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * A/B testing on top of {@link OtterRuntime}'s existing canary mechanism (Milestone 6,
 * {@code LifecycleManager.deployCanary}). This class does not route any traffic itself — that
 * remains {@code ManagedModel}'s job — it adds three things canary deployment alone doesn't
 * have: a stable experiment identity independent of the underlying model version strings,
 * outcome recording, and statistical comparison between control and variant.
 *
 * <p>One running experiment per {@code modelId} at a time — starting a second experiment for a
 * model that already has one running fails fast rather than silently deploying a second canary
 * on top of the first, which {@code LifecycleManager.deployCanary} itself would technically
 * allow but which would make control/variant attribution ambiguous for this class's purposes.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ExperimentManager experiments = new ExperimentManager(runtime);
 *
 * Experiment exp = experiments.startExperiment(
 *         "fraud-model-v3.3-rollout", variantConfig, 10); // 10% of traffic to the variant
 *
 * // In your pipeline, after each inference + rule decision:
 * ExperimentGroup group = result.wasVariant() ? ExperimentGroup.VARIANT : ExperimentGroup.CONTROL;
 * experiments.recordOutcome(exp.experimentId(), group, decision.confidence());
 *
 * // Later, check significance:
 * StatisticalTest.ComparisonResult comparison = experiments.compareContinuous(exp.experimentId());
 * if (comparison.significantAt95() && comparison.variantMean() > comparison.controlMean()) {
 *     experiments.concludePromote(exp.experimentId());
 * } else {
 *     experiments.concludeRollback(exp.experimentId());
 * }
 * }</pre>
 *
 * <p><b>Attributing which group served a given request:</b> this class deliberately does not
 * try to infer that for you — {@code ManagedModel}'s canary routing decision (primary vs.
 * canary) happens inside {@code infer()} and isn't exposed per-call today. Callers currently
 * need their own way to know which group a given result came from (e.g. tagging the
 * {@code InferenceResult}'s metadata, or comparing the served model version against
 * {@link Experiment#controlVersion()}/{@link Experiment#variantVersion()}). Exposing that
 * routing decision directly from {@code ManagedModel.infer()} would remove the need for this
 * workaround — tracked as a natural follow-up, not implemented here to avoid changing
 * {@code ManagedModel}'s public contract as a side effect of this module.
 *
 * @since 0.1.0
 */
public class ExperimentManager {

    private final OtterRuntime runtime;
    private final ConcurrentHashMap<String, Experiment> experimentsById = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> runningExperimentIdByModelId = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, List<ExperimentOutcome>> outcomesByExperimentId = new ConcurrentHashMap<>();

    public ExperimentManager(OtterRuntime runtime) {
        this.runtime = Objects.requireNonNull(runtime, "runtime cannot be null");
    }

    /**
     * Starts a new experiment: deploys {@code variantConfig} as a canary at
     * {@code variantTrafficPercent}, and begins tracking it under {@code name}.
     *
     * @param name                  human-readable experiment name
     * @param variantConfig         the candidate model configuration
     * @param variantTrafficPercent percentage of traffic to route to the variant (0-100)
     * @return the started experiment
     * @throws ExperimentException if an experiment is already running for this model
     * @throws DeploymentException if the canary deployment itself fails validation/warmup
     */
    public Experiment startExperiment(String name, ModelConfig variantConfig, int variantTrafficPercent)
            throws DeploymentException {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(variantConfig, "variantConfig cannot be null");
        String modelId = variantConfig.getModelId();

        String existing = runningExperimentIdByModelId.putIfAbsent(modelId, "PENDING");
        if (existing != null) {
            throw new ExperimentException(
                    "An experiment is already running for model '" + modelId + "': " + existing);
        }

        try {
            String controlVersion = currentActiveVersion(modelId);
            runtime.deployCanary(variantConfig, variantTrafficPercent);
            String variantVersion = variantConfig.getModelVersion() != null
                    ? variantConfig.getModelVersion()
                    : "unversioned";

            String experimentId = UUID.randomUUID().toString();
            Experiment experiment = new Experiment(
                    experimentId, name, modelId, controlVersion, variantVersion,
                    variantTrafficPercent, ExperimentStatus.RUNNING,
                    System.currentTimeMillis(), -1);

            experimentsById.put(experimentId, experiment);
            outcomesByExperimentId.put(experimentId, new CopyOnWriteArrayList<>());
            runningExperimentIdByModelId.put(modelId, experimentId);
            return experiment;
        } catch (DeploymentException | RuntimeException e) {
            runningExperimentIdByModelId.remove(modelId, "PENDING");
            throw e;
        }
    }

    /**
     * Records one outcome observation for an experiment.
     *
     * @param experimentId the experiment this observation belongs to
     * @param group        which arm served the request
     * @param metricValue  the observed metric value — meaning is caller-defined, see class Javadoc
     * @throws ExperimentException if the experiment id is unknown
     */
    public void recordOutcome(String experimentId, ExperimentGroup group, double metricValue) {
        List<ExperimentOutcome> outcomes = requireOutcomeList(experimentId);
        outcomes.add(new ExperimentOutcome(experimentId, group, metricValue, System.currentTimeMillis()));
    }

    /**
     * Compares control vs. variant as continuous metrics (Welch's t-test) — use for confidence
     * scores, latency, or any real-valued measurement.
     *
     * @throws ExperimentException     if the experiment id is unknown
     * @throws IllegalArgumentException if either group has fewer than 2 recorded observations
     */
    public StatisticalTest.ComparisonResult compareContinuous(String experimentId) {
        List<ExperimentOutcome> outcomes = requireOutcomeList(experimentId);
        List<Double> control = outcomes.stream()
                .filter(o -> o.group() == ExperimentGroup.CONTROL)
                .map(ExperimentOutcome::metricValue)
                .toList();
        List<Double> variant = outcomes.stream()
                .filter(o -> o.group() == ExperimentGroup.VARIANT)
                .map(ExperimentOutcome::metricValue)
                .toList();
        return StatisticalTest.welchTTest(control, variant);
    }

    /**
     * Compares control vs. variant as a binary/proportion metric (two-proportion z-test) —
     * treats every recorded {@code metricValue >= positiveThreshold} as a "positive" outcome
     * (e.g. "flagged as fraud"), everything else as negative, then compares the two rates.
     *
     * @param experimentId      the experiment to compare
     * @param positiveThreshold threshold at/above which a recorded value counts as positive
     * @throws ExperimentException if the experiment id is unknown
     */
    public StatisticalTest.ComparisonResult compareBinary(String experimentId, double positiveThreshold) {
        List<ExperimentOutcome> outcomes = requireOutcomeList(experimentId);
        int controlTotal = 0;
        int controlSuccess = 0;
        int variantTotal = 0;
        int variantSuccess = 0;
        for (ExperimentOutcome outcome : outcomes) {
            boolean positive = outcome.metricValue() >= positiveThreshold;
            if (outcome.group() == ExperimentGroup.CONTROL) {
                controlTotal++;
                if (positive) controlSuccess++;
            } else {
                variantTotal++;
                if (positive) variantSuccess++;
            }
        }
        return StatisticalTest.twoProportionZTest(controlSuccess, controlTotal, variantSuccess, variantTotal);
    }

    /**
     * Concludes the experiment by promoting the variant to primary
     * ({@code OtterRuntime.promoteCanary}) — the variant won.
     */
    public Experiment concludePromote(String experimentId) {
        Experiment experiment = requireExperiment(experimentId);
        runtime.promoteCanary(experiment.modelId());
        return conclude(experiment, ExperimentStatus.CONCLUDED_PROMOTED);
    }

    /**
     * Concludes the experiment by discarding the variant ({@code OtterRuntime.rollbackCanary})
     * — the control stayed primary.
     */
    public Experiment concludeRollback(String experimentId) {
        Experiment experiment = requireExperiment(experimentId);
        runtime.rollbackCanary(experiment.modelId());
        return conclude(experiment, ExperimentStatus.CONCLUDED_ROLLED_BACK);
    }

    public Experiment getExperiment(String experimentId) {
        return requireExperiment(experimentId);
    }

    public List<ExperimentOutcome> getOutcomes(String experimentId) {
        return List.copyOf(requireOutcomeList(experimentId));
    }

    public Optional<String> getRunningExperimentId(String modelId) {
        String id = runningExperimentIdByModelId.get(modelId);
        return (id == null || id.equals("PENDING")) ? Optional.empty() : Optional.of(id);
    }

    private Experiment conclude(Experiment experiment, ExperimentStatus status) {
        Experiment updated = experiment.withStatus(status, System.currentTimeMillis());
        experimentsById.put(experiment.experimentId(), updated);
        runningExperimentIdByModelId.remove(experiment.modelId());
        return updated;
    }

    private String currentActiveVersion(String modelId) {
        if (!runtime.isDeployed(modelId)) {
            return "none";
        }
        ManagedModel managed = runtime.getManagedModel(modelId);
        var activeVersion = managed.getActiveVersion();
        return activeVersion != null ? activeVersion.getVersion() : "none";
    }

    private Experiment requireExperiment(String experimentId) {
        Experiment experiment = experimentsById.get(experimentId);
        if (experiment == null) {
            throw new ExperimentException("Unknown experiment id: " + experimentId);
        }
        return experiment;
    }

    private List<ExperimentOutcome> requireOutcomeList(String experimentId) {
        List<ExperimentOutcome> outcomes = outcomesByExperimentId.get(experimentId);
        if (outcomes == null) {
            throw new ExperimentException("Unknown experiment id: " + experimentId);
        }
        return outcomes;
    }
}
