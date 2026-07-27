package com.codedstream.otterstream.context.memory;

import com.codedstream.otterstream.context.spi.ContextProvider;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Conversation/session memory for RAG applications — the "session/user context" and "memory"
 * piece a RAG pipeline needs: what has this session/user already said, so the model has
 * conversational context rather than seeing each message in isolation.
 *
 * <p><b>Bounded in both dimensions, specifically for "large concurrent users":</b>
 * <ul>
 *   <li><b>Per-session:</b> each session keeps at most {@code maxTurnsPerSession} turns (oldest
 *       dropped first) — a runaway single conversation can't grow without limit.</li>
 *   <li><b>Across all sessions:</b> backed by a Caffeine cache with a hard
 *       {@code maxSessions} bound and a TTL, so a workload with many concurrent/short-lived
 *       sessions (exactly the "large concurrent users" case) can't grow this provider's memory
 *       unboundedly the way a plain {@code ConcurrentHashMap} would — the same safety property
 *       {@link com.codedstream.otterstream.context.ContextCache} has, applied here for the same
 *       reason.</li>
 * </ul>
 *
 * <p>Per-session appends are synchronized on that session's own deque (a per-key lock, not a
 * single global lock) — concurrent turns for <em>different</em> sessions never contend with each
 * other, which is the property that actually matters for scaling to many concurrent users; only
 * concurrent turns for the *same* session serialize, which is inherently sequential anyway (a
 * conversation only has one "next turn" at a time).
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ConversationMemoryProvider memory = new ConversationMemoryProvider(
 *         20,                  // keep last 20 turns per session
 *         100_000,             // track at most 100k concurrent sessions
 *         Duration.ofHours(2)); // evict a session after 2h of inactivity
 *
 * memory.appendTurn(sessionId, "user", userMessage);
 * // ... after getting a model response:
 * memory.appendTurn(sessionId, "assistant", modelResponse);
 *
 * // As a ContextProvider, fetch() returns the turn history for the next request:
 * Map<String, Object> context = memory.fetch(sessionId, Map.of());
 * List<ConversationTurn> turns = (List<ConversationTurn>) context.get("turns");
 * }</pre>
 *
 * @since 0.1.0
 */
public class ConversationMemoryProvider implements ContextProvider {

    private final String providerId;
    private final int maxTurnsPerSession;
    private final Cache<String, SessionMemory> sessionsByKey;

    private record SessionMemory(Deque<ConversationTurn> turns, ReentrantLock lock) {
        static SessionMemory create() {
            return new SessionMemory(new ArrayDeque<>(), new ReentrantLock());
        }
    }

    public ConversationMemoryProvider(int maxTurnsPerSession, long maxSessions, java.time.Duration sessionTtl) {
        this("conversation-memory", maxTurnsPerSession, maxSessions, sessionTtl);
    }

    public ConversationMemoryProvider(String providerId, int maxTurnsPerSession, long maxSessions, java.time.Duration sessionTtl) {
        this.providerId = providerId;
        this.maxTurnsPerSession = maxTurnsPerSession;
        this.sessionsByKey = Caffeine.newBuilder()
                .maximumSize(maxSessions)
                .expireAfterAccess(sessionTtl.toSeconds(), TimeUnit.SECONDS)
                .build();
    }

    @Override
    public String getProviderId() {
        return providerId;
    }

    /** Records one turn for a session, evicting the oldest turn if {@code maxTurnsPerSession} is exceeded. */
    public void appendTurn(String sessionId, String role, String content) {
        SessionMemory session = sessionsByKey.get(sessionId, id -> SessionMemory.create());
        session.lock().lock();
        try {
            session.turns().addLast(new ConversationTurn(role, content, System.currentTimeMillis()));
            while (session.turns().size() > maxTurnsPerSession) {
                session.turns().removeFirst();
            }
        } finally {
            session.lock().unlock();
        }
    }

    /** Clears a session's history — call when a conversation explicitly ends. */
    public void clearSession(String sessionId) {
        sessionsByKey.invalidate(sessionId);
    }

    public int getTrackedSessionCount() {
        return (int) sessionsByKey.estimatedSize();
    }

    /**
     * @return {@code {"turns": List<ConversationTurn>}} — empty list if the session has no
     *         recorded history (or has expired/never existed)
     */
    @Override
    public Map<String, Object> fetch(String sessionId, Map<String, Object> request) {
        SessionMemory session = sessionsByKey.getIfPresent(sessionId);
        if (session == null) {
            return Map.of("turns", List.<ConversationTurn>of());
        }
        session.lock().lock();
        try {
            return Map.of("turns", List.copyOf(session.turns()));
        } finally {
            session.lock().unlock();
        }
    }
}
