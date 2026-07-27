package com.codedstream.otterstream.rules.expr;

import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Compiles a small, safe boolean expression language into a {@link CompiledCondition}.
 *
 * <p>Deliberately <em>not</em> a general-purpose scripting language — no arithmetic, no
 * function calls, no method invocation, no reflection. This is a considered trade-off, not a
 * missing feature: rule conditions come from configuration (YAML/properties), often authored or
 * reviewed by people who aren't Java engineers, and a rule engine that can only express
 * comparisons and boolean logic can't be turned into an arbitrary-code-execution vector by a
 * malformed or malicious rule file. If you need real computation, do it upstream (as another
 * inference/feature-lookup stage) and expose the result as a field the condition can compare.
 *
 * <h2>Grammar</h2>
 * <pre>
 * expr       := orExpr
 * orExpr     := andExpr ( '||' andExpr )*
 * andExpr    := notExpr ( '&&' notExpr )*
 * notExpr    := '!' notExpr | comparison
 * comparison := operand ( ('==' | '!=' | '>' | '>=' | '<' | '<=') operand )?
 * operand    := NUMBER | STRING | 'true' | 'false' | 'null' | fieldPath | '(' expr ')'
 * fieldPath  := IDENT ('.' IDENT)*
 * </pre>
 *
 * <h2>Example conditions</h2>
 * <pre>{@code
 * output.risk_score > 0.85
 * output.risk_score > 0.85 && output.country == 'US'
 * !(output.decision == 'APPROVE') || confidence < 0.5
 * }</pre>
 *
 * @since 0.1.0
 */
public final class ExpressionEvaluator {

    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "\\s*(?:(&&|\\|\\||==|!=|>=|<=|[()!><.,])|('(?:[^'\\\\]|\\\\.)*'|\"(?:[^\"\\\\]|\\\\.)*\")"
                    + "|(-?\\d+\\.\\d+|-?\\d+)|(true|false|null)|([A-Za-z_][A-Za-z0-9_]*))");

    private ExpressionEvaluator() {
    }

    /**
     * Parses and compiles a condition string once. The returned {@link CompiledCondition} is
     * safe to cache and evaluate repeatedly — evaluation never re-parses.
     *
     * @param expression the condition string
     * @return a compiled, reusable condition
     * @throws RuleEvaluationException if the expression is syntactically invalid
     */
    public static CompiledCondition compile(String expression) throws RuleEvaluationException {
        List<String> tokens = tokenize(expression);
        Parser parser = new Parser(tokens, expression);
        Node node = parser.parseExpr();
        parser.expectEnd();
        return node::evalBool;
    }

    private static List<String> tokenize(String expression) throws RuleEvaluationException {
        List<String> tokens = new ArrayList<>();
        Matcher matcher = TOKEN_PATTERN.matcher(expression);
        int pos = 0;
        while (pos < expression.length()) {
            matcher.region(pos, expression.length());
            if (!matcher.lookingAt()) {
                throw new RuleEvaluationException(
                        "Unable to parse expression near position " + pos + " in: " + expression);
            }
            String whole = matcher.group();
            if (!whole.isBlank()) {
                tokens.add(whole.trim());
            }
            pos = matcher.end();
        }
        return tokens;
    }

    /** Simple recursive-descent parser producing a {@link Node} AST. */
    private static final class Parser {
        private final List<String> tokens;
        private final String source;
        private int index = 0;

        Parser(List<String> tokens, String source) {
            this.tokens = tokens;
            this.source = source;
        }

        Node parseExpr() throws RuleEvaluationException {
            return parseOr();
        }

        private Node parseOr() throws RuleEvaluationException {
            Node left = parseAnd();
            while (peekIs("||")) {
                advance();
                left = new Node.Or(left, parseAnd());
            }
            return left;
        }

        private Node parseAnd() throws RuleEvaluationException {
            Node left = parseNot();
            while (peekIs("&&")) {
                advance();
                left = new Node.And(left, parseNot());
            }
            return left;
        }

        private Node parseNot() throws RuleEvaluationException {
            if (peekIs("!")) {
                advance();
                return new Node.Not(parseNot());
            }
            return parseComparison();
        }

        private Node parseComparison() throws RuleEvaluationException {
            Node left = parseOperand();
            String op = peek();
            Node.Op mapped = mapOp(op);
            if (mapped != null) {
                advance();
                Node right = parseOperand();
                return new Node.Comparison(left, mapped, right);
            }
            return left;
        }

        private Node.Op mapOp(String token) {
            return switch (token == null ? "" : token) {
                case "==" -> Node.Op.EQ;
                case "!=" -> Node.Op.NEQ;
                case ">" -> Node.Op.GT;
                case ">=" -> Node.Op.GTE;
                case "<" -> Node.Op.LT;
                case "<=" -> Node.Op.LTE;
                default -> null;
            };
        }

        private Node parseOperand() throws RuleEvaluationException {
            String token = peek();
            if (token == null) {
                throw error("Unexpected end of expression");
            }
            if (token.equals("(")) {
                advance();
                Node inner = parseExpr();
                expect(")");
                return inner;
            }
            if (token.equals("true")) {
                advance();
                return new Node.Literal(Boolean.TRUE);
            }
            if (token.equals("false")) {
                advance();
                return new Node.Literal(Boolean.FALSE);
            }
            if (token.equals("null")) {
                advance();
                return new Node.Literal(null);
            }
            if (token.startsWith("'") || token.startsWith("\"")) {
                advance();
                String unquoted = token.substring(1, token.length() - 1)
                        .replace("\\'", "'")
                        .replace("\\\"", "\"")
                        .replace("\\\\", "\\");
                return new Node.Literal(unquoted);
            }
            if (token.matches("-?\\d+\\.\\d+")) {
                advance();
                return new Node.Literal(Double.parseDouble(token));
            }
            if (token.matches("-?\\d+")) {
                advance();
                return new Node.Literal(Long.parseLong(token));
            }
            if (token.matches("[A-Za-z_][A-Za-z0-9_]*")) {
                List<String> path = new ArrayList<>();
                path.add(token);
                advance();
                while (peekIs(".")) {
                    advance();
                    String next = peek();
                    if (next == null || !next.matches("[A-Za-z_][A-Za-z0-9_]*")) {
                        throw error("Expected field name after '.'");
                    }
                    path.add(next);
                    advance();
                }
                return new Node.FieldRef(path);
            }
            throw error("Unexpected token: " + token);
        }

        private String peek() {
            return index < tokens.size() ? tokens.get(index) : null;
        }

        private boolean peekIs(String expected) {
            return expected.equals(peek());
        }

        private void advance() {
            index++;
        }

        private void expect(String expected) throws RuleEvaluationException {
            if (!peekIs(expected)) {
                throw error("Expected '" + expected + "'");
            }
            advance();
        }

        void expectEnd() throws RuleEvaluationException {
            if (index != tokens.size()) {
                throw error("Unexpected trailing tokens starting at: " + peek());
            }
        }

        private RuleEvaluationException error(String message) {
            return new RuleEvaluationException(message + " (expression: \"" + source + "\")");
        }
    }
}
