package com.codedstream.otterstream.inference.exception;

public class ModelLoadException extends Exception {
    public ModelLoadException(String message) {
        super(message);
    }

    public ModelLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}
