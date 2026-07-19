package com.codedstream.otterstream.runtime.lifecycle;

import com.codedstream.otterstream.inference.config.ModelConfig;
import java.util.Objects;

/**
 * Tracks a single deployment attempt of a model version as it moves through the
 * lifecycle: validate → warm → active → retired (or failed at any prior stage).
 *
 * <p>Mutable by design (the {@link LifecycleManager} advances {@link #getStatus()} as the
 * deployment progresses) but every mutation is confined to the lifecycle package.
 *
 * <p>Carries the {@link ModelConfig} it was deployed from so a retired version can be
 * re-deployed later — this is what powers {@link LifecycleManager#rollback(String)}.
 *
 * @since 0.1.0
 * @see LifecycleManager
 */
public final class ModelVersion {

    /**
     * The stage a {@link ModelVersion} is currently in.
     */
    public enum Status {
        /** Engine is initializing / model is being loaded and checked. */
        VALIDATING,
        /** Model loaded successfully; an optional warmup probe is being run. */
        WARMING,
        /** This version is live and serving traffic (as primary or canary). */
        ACTIVE,
        /** This version was swapped out in favor of a newer one and its engine has been closed. */
        RETIRED,
        /** Validation or warmup failed; this version never went live. */
        FAILED
    }

    private final String version;
    private final ModelConfig config;
    private final long createdAt;
    private volatile Status status;

    public ModelVersion(String version, ModelConfig config) {
        this.version = Objects.requireNonNull(version, "version cannot be null");
        this.config = config;
        this.createdAt = System.currentTimeMillis();
        this.status = Status.VALIDATING;
    }

    public String getVersion() {
        return version;
    }

    /**
     * @return the configuration this version was deployed from; may be null for versions
     *         constructed without one (not expected in normal use through {@link LifecycleManager})
     */
    public ModelConfig getConfig() {
        return config;
    }

    /**
     * @return epoch millis when this ModelVersion instance was created (deployment start time)
     */
    public long getCreatedAt() {
        return createdAt;
    }

    public Status getStatus() {
        return status;
    }

    void setStatus(Status status) {
        this.status = Objects.requireNonNull(status);
    }

    @Override
    public String toString() {
        return "ModelVersion{version='" + version + "', status=" + status + ", createdAt=" + createdAt + "}";
    }
}
