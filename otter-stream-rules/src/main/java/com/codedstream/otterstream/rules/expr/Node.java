package com.codedstream.otterstream.rules.expr;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Package-private AST for the rule condition grammar. Deliberately a plain sealed interface
 * hierarchy rather than pattern-matching switch (which is preview-only under this project's
 * Java 17 baseline) — each node type implements {@link #eval(Map)} directly. Migrating to a
 * {@code switch} expression is a mechanical follow-up once the baseline moves to 21.
 */
sealed interface Node permits Node.Or, Node.And, Node.Not, Node.Comparison, Node.FieldRef, Node.Literal {

    Object eval(Map<String, Object> context);

    default boolean evalBool(Map<String, Object> context) {
        Object result = eval(context);
        if (result instanceof Boolean b) {
            return b;
        }
        throw new IllegalStateException("Expression did not evaluate to a boolean: " + result);
    }

    record Or(Node left, Node right) implements Node {
        public Object eval(Map<String, Object> context) {
            return left.evalBool(context) || right.evalBool(context);
        }
    }

    record And(Node left, Node right) implements Node {
        public Object eval(Map<String, Object> context) {
            return left.evalBool(context) && right.evalBool(context);
        }
    }

    record Not(Node operand) implements Node {
        public Object eval(Map<String, Object> context) {
            return !operand.evalBool(context);
        }
    }

    enum Op { EQ, NEQ, GT, GTE, LT, LTE }

    record Comparison(Node left, Op op, Node right) implements Node {
        public Object eval(Map<String, Object> context) {
            Object l = left.eval(context);
            Object r = right.eval(context);
            return switch (op) {
                case EQ -> Objects.equals(l, r);
                case NEQ -> !Objects.equals(l, r);
                case GT -> compareNumeric(l, r) > 0;
                case GTE -> compareNumeric(l, r) >= 0;
                case LT -> compareNumeric(l, r) < 0;
                case LTE -> compareNumeric(l, r) <= 0;
            };
        }

        private static int compareNumeric(Object l, Object r) {
            if (!(l instanceof Number ln) || !(r instanceof Number rn)) {
                throw new IllegalStateException(
                        "Ordering comparison requires two numbers, got: " + describe(l) + " and " + describe(r));
            }
            return Double.compare(ln.doubleValue(), rn.doubleValue());
        }

        private static String describe(Object o) {
            return o == null ? "null" : o.getClass().getSimpleName() + "(" + o + ")";
        }
    }

    /** A dotted field path (e.g. {@code output.risk_score}) resolved against nested maps only — no reflection, by design (safety + speed). */
    record FieldRef(List<String> path) implements Node {
        @SuppressWarnings("unchecked")
        public Object eval(Map<String, Object> context) {
            Object current = context;
            for (String segment : path) {
                if (!(current instanceof Map<?, ?> map)) {
                    return null;
                }
                current = ((Map<String, Object>) map).get(segment);
            }
            return current;
        }
    }

    record Literal(Object value) implements Node {
        public Object eval(Map<String, Object> context) {
            return value;
        }
    }
}
