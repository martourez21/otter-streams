package com.codedstreams.otterstreams.sql.metadata;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Schema definition for model inputs and outputs.
 */
public class InputOutputSchema implements Serializable {
    private static final long serialVersionUID = 1L;

    private final List<FieldSchema> inputs;
    private final List<FieldSchema> outputs;

    public InputOutputSchema(List<FieldSchema> inputs, List<FieldSchema> outputs) {
        this.inputs = List.copyOf(inputs);
        this.outputs = List.copyOf(outputs);
    }

    public List<FieldSchema> getInputs() { return inputs; }
    public List<FieldSchema> getOutputs() { return outputs; }

    public static class FieldSchema implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String name;
        private final String type;
        private final List<Integer> shape;

        public FieldSchema(String name, String type, List<Integer> shape) {
            this.name = name;
            this.type = type;
            this.shape = shape != null ? List.copyOf(shape) : List.of();
        }

        public String getName() { return name; }
        public String getType() { return type; }
        public List<Integer> getShape() { return shape; }
    }
}

