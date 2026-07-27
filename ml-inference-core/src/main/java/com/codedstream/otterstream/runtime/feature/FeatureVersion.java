package com.codedstream.otterstream.runtime.feature;

import java.util.Objects;

/**
 * A version tag stamped onto a feature fetch, produced by {@link VersioningFeatureProvider} —
 * the "feature versioning" piece of the Feature Store Integration roadmap item.
 *
 * <p><b>Scope, stated plainly:</b> this is fetch-time version stamping — "these values were
 * read from provider X, version tag Y, at time Z" — not point-in-time correctness / time-travel
 * queries ("give me what the features were as of last Tuesday"). True point-in-time correctness
 * needs the underlying store to support historical reads (Feast's offline store does; a plain
 * Redis hash or JDBC table generally doesn't, since they only hold current state). Building that
 * properly would mean either a versioned storage backend this project doesn't control, or a
 * separate feature history store — a materially bigger undertaking than a stamping decorator,
 * and not what's built here.
 *
 * @param versionTag     caller-supplied version identifier (e.g. a feature-set version, a
 *                       git commit, a training-pipeline run id — whatever "version" means for
 *                       your feature pipeline)
 * @param computedAtMillis when this fetch happened
 * @since 0.1.0
 */
public record FeatureVersion(String versionTag, long computedAtMillis) {

    public FeatureVersion {
        Objects.requireNonNull(versionTag, "versionTag cannot be null");
    }
}
