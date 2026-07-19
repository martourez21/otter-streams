package com.codedstream.otterstream.rules.drools;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.model.Decision;
import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import com.codedstream.otterstream.rules.spi.DecisionEngineConnector;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.kie.api.KieServices;
import org.kie.api.builder.KieBuilder;
import org.kie.api.builder.KieFileSystem;
import org.kie.api.builder.Message;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;

/**
 * {@link DecisionEngineConnector} backed by an embedded Drools KIE session — for teams running
 * Drools in-process rather than fronting it with a KIE Server REST endpoint (use
 * {@code RestDecisionEngineConnector} in {@code otter-stream-rules} for that case instead, with
 * no Drools dependency at all).
 *
 * <p><b>Integration pattern:</b> this connector inserts the inference result as an
 * {@link InferenceFact} (a simple bean wrapping {@code modelId}/{@code outputs}/{@code context})
 * into a fresh {@link KieSession}, fires all rules, then collects every {@link RuleDecision}
 * fact your DRL rules chose to {@code insert()} as their conclusion. This mirrors the standard
 * Drools "rules produce facts, not side effects" idiom — your DRL is responsible for inserting
 * a {@code RuleDecision} when a rule fires, e.g.:
 *
 * <pre>{@code
 * rule "High Risk Score"
 *     salience 100
 * when
 *     $fact : InferenceFact(risk() > 0.85)
 * then
 *     insert(new RuleDecision("high-risk-score", "FRAUD", "HIGH_RISK"));
 * end
 * }</pre>
 *
 * <p><b>Verification note:</b> this class is written against the standard, well-documented KIE
 * API shape ({@code KieServices}/{@code KieFileSystem}/{@code KieBuilder}/{@code KieContainer}/
 * {@code KieSession}) but has not been compiled against real Drools jars in the environment this
 * was authored in (no Maven Central network access there). Treat this as a solid starting point
 * to validate against your actual Drools version, not as pre-verified working code — the same
 * caveat noted for the rest of this project's Maven-dependent modules.
 *
 * @since 0.1.0
 */
public final class DroolsDecisionEngineConnector implements DecisionEngineConnector {

    private final String connectorId;
    private final KieContainer kieContainer;

    /**
     * @param connectorId  a stable identifier, e.g. {@code "drools-fraud-rules"}
     * @param drlResources one or more classpath paths to {@code .drl} rule files
     */
    public DroolsDecisionEngineConnector(String connectorId, List<String> drlResources) throws RuleEvaluationException {
        this.connectorId = Objects.requireNonNull(connectorId, "connectorId cannot be null");
        this.kieContainer = buildContainer(Objects.requireNonNull(drlResources, "drlResources cannot be null"));
    }

    private KieContainer buildContainer(List<String> drlResources) throws RuleEvaluationException {
        KieServices kieServices = KieServices.Factory.get();
        KieFileSystem kfs = kieServices.newKieFileSystem();

        for (String resourcePath : drlResources) {
            try (InputStream stream = getClass().getClassLoader().getResourceAsStream(resourcePath)) {
                if (stream == null) {
                    throw new RuleEvaluationException("DRL classpath resource not found: " + resourcePath);
                }
                String content = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
                kfs.write("src/main/resources/" + resourcePath, content);
            } catch (IOException e) {
                throw new RuleEvaluationException("Failed to read DRL resource: " + resourcePath, e);
            }
        }

        KieBuilder kieBuilder = kieServices.newKieBuilder(kfs);
        kieBuilder.buildAll();
        List<Message> errors = kieBuilder.getResults().getMessages(Message.Level.ERROR);
        if (!errors.isEmpty()) {
            throw new RuleEvaluationException("Drools build errors: " + errors);
        }

        return kieServices.newKieContainer(kieServices.getRepository().getDefaultReleaseId());
    }

    @Override
    public String getConnectorId() {
        return connectorId;
    }

    @Override
    public Decision evaluate(InferenceResult inferenceResult, Map<String, Object> context) throws RuleEvaluationException {
        KieSession session = kieContainer.newKieSession();
        try {
            InferenceFact fact = new InferenceFact(
                    inferenceResult.getModelId(),
                    inferenceResult.getOutputs(),
                    context != null ? context : Map.of());
            session.insert(fact);
            session.fireAllRules();

            List<RuleDecision> decisions = new ArrayList<>();
            for (Object o : session.getObjects()) {
                if (o instanceof RuleDecision ruleDecision) {
                    decisions.add(ruleDecision);
                }
            }

            if (decisions.isEmpty()) {
                return Decision.unflagged(System.currentTimeMillis());
            }
            RuleDecision top = decisions.get(0);
            List<String> matchedIds = new ArrayList<>();
            for (RuleDecision d : decisions) {
                matchedIds.add(d.ruleId());
            }
            return new Decision(
                    top.flag(), top.category(), 0.0, matchedIds, System.currentTimeMillis(),
                    Map.of("connectorId", connectorId));
        } catch (RuntimeException e) {
            throw new RuleEvaluationException("Drools evaluation failed for connector '" + connectorId + "'", e);
        } finally {
            session.dispose();
        }
    }

    /** Fact inserted into the KIE session — the input side of the rules. */
    public record InferenceFact(String modelId, Map<String, Object> outputs, Map<String, Object> context) {
        public InferenceFact {
            outputs = outputs == null ? Map.of() : new LinkedHashMap<>(outputs);
            context = context == null ? Map.of() : new LinkedHashMap<>(context);
        }

        /** Convenience accessor for the common case of a numeric "risk_score"-style output field, for use in DRL `when` clauses. */
        public double risk() {
            Object value = outputs.get("risk_score");
            return value instanceof Number number ? number.doubleValue() : 0.0;
        }
    }

    /** Fact your DRL rules insert as their conclusion — the output side. */
    public record RuleDecision(String ruleId, String flag, String category) {
    }
}
