package com.codedstreams.otterstreams.sql.udf;

import com.codedstreams.otterstreams.sql.rules.DecisionEngine;
import com.codedstreams.otterstreams.sql.rules.DroolsDecisionEngine;
import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.StatelessKieSession;

/**
 * Provides a rule engine instance for Flink SQL and DataStream pipelines.
 *
 * <p>Uses lazy initialization to ensure:
 * <ul>
 *   <li>One engine per task</li>
 *   <li>No serialization of Drools internals</li>
 *   <li>Fast startup and reuse</li>
 * </ul>
 *
 * <h3>Used By</h3>
 * <ul>
 *   <li>Flink SQL UDFs</li>
 *   <li>DataStream operators</li>
 *   <li>CEP conditions</li>
 * </ul>
 */
public final class RuleEngineProvider {

    private static transient DecisionEngine<Object> engine;

    private RuleEngineProvider() {
    }

    /**
     * Returns a lazily initialized decision engine.
     */
    public static DecisionEngine<Object> get() {
        if (engine == null) {
            engine = createDroolsEngine();
        }
        return engine;
    }

    /**
     * Builds a Drools stateless session.
     *
     * <p>Rules are loaded via:
     * <ul>
     *   <li>META-INF/kmodule.xml</li>
     *   <li>classpath DRL files</li>
     * </ul>
     */
    private static DecisionEngine<Object> createDroolsEngine() {
        KieServices kieServices = KieServices.Factory.get();
        KieContainer container = kieServices.getKieClasspathContainer();
        StatelessKieSession session = container.newStatelessKieSession();
        return new DroolsDecisionEngine(session);
    }
}
