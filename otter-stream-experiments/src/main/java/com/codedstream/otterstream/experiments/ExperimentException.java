package com.codedstream.otterstream.experiments;

/**
 * Thrown for experiment lifecycle errors — starting an experiment for a model that already has
 * one running, concluding/recording against an unknown experiment id, etc.
 *
 * @since 0.1.0
 */
public class ExperimentException extends RuntimeException {

    public ExperimentException(String message) {
        super(message);
    }

    public ExperimentException(String message, Throwable cause) {
        super(message, cause);
    }
}
