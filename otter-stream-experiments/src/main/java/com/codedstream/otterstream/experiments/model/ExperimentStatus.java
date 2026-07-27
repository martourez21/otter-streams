package com.codedstream.otterstream.experiments.model;

/**
 * Lifecycle status of an {@link Experiment}.
 *
 * @since 0.1.0
 */
public enum ExperimentStatus {
    /** Canary deployed, traffic split active, outcomes being recorded. */
    RUNNING,
    /** Concluded by promoting the variant to primary — the variant won. */
    CONCLUDED_PROMOTED,
    /** Concluded by discarding the variant — the control stayed primary. */
    CONCLUDED_ROLLED_BACK
}
