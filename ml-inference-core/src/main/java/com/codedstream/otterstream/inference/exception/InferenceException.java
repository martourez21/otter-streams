package com.codedstream.otterstream.inference.exception;

public class InferenceException extends Exception {
    public InferenceException(String message) {
        super(message);
    }

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }
}

