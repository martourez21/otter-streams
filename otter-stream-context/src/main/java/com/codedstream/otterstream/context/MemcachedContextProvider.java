package com.codedstream.otterstream.context;

import com.codedstream.otterstream.context.spi.ContextProvider;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Objects;

/**
 * {@link ContextProvider} backed by Memcached — the other widely-deployed cache alongside
 * Redis, and specifically requested as a complementary cache provider option. Implements just
 * enough of Memcached's classic text protocol (get/set) over a raw {@link Socket} to be useful
 * as a context cache — no client library dependency, consistent with this module's
 * zero-new-dependency approach elsewhere (see {@code RestDecisionEngineConnector} in
 * {@code otter-stream-rules} for the same reasoning applied to HTTP).
 *
 * <p><b>Scope, stated plainly:</b> this is a minimal single-node client — one TCP connection per
 * call, no connection pooling, no consistent hashing across a Memcached cluster, no SASL auth.
 * It's the right tool for "point this at one Memcached instance (or a client-side load balancer
 * in front of a cluster) and cache context maps" — it is not a replacement for a full-featured
 * client (e.g. `XMemcached`, `spymemcached`) if you need cluster-aware routing or auth. Values
 * are serialized as JSON (via the `jackson-databind` this module already gets transitively
 * through `ml-inference-core`), not Java serialization — avoids both the cross-version
 * fragility and the deserialization-of-untrusted-data concerns Java serialization carries,
 * even though in this case the data was written by this same class.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * MemcachedContextProvider cache = new MemcachedContextProvider(
 *         "memcached-session-cache", "localhost", 11211, "ctx:", 300); // 300s default TTL
 *
 * cache.store(sessionId, assembledContextMap);
 * // ...later, from another request, possibly another process:
 * Map<String, Object> cached = cache.fetch(sessionId, Map.of());
 * }</pre>
 *
 * @since 0.1.0
 */
public class MemcachedContextProvider implements ContextProvider, Closeable {

    private final String providerId;
    private final String host;
    private final int port;
    private final String keyPrefix;
    private final int defaultTtlSeconds;
    private final int connectTimeoutMillis;
    private final int socketTimeoutMillis;
    private final ObjectMapper mapper = new ObjectMapper();

    public MemcachedContextProvider(String providerId, String host, int port, String keyPrefix, int defaultTtlSeconds) {
        this(providerId, host, port, keyPrefix, defaultTtlSeconds, 2000, 2000);
    }

    public MemcachedContextProvider(
            String providerId, String host, int port, String keyPrefix, int defaultTtlSeconds,
            int connectTimeoutMillis, int socketTimeoutMillis) {
        this.providerId = Objects.requireNonNull(providerId, "providerId cannot be null");
        this.host = Objects.requireNonNull(host, "host cannot be null");
        this.port = port;
        this.keyPrefix = keyPrefix != null ? keyPrefix : "";
        this.defaultTtlSeconds = defaultTtlSeconds;
        this.connectTimeoutMillis = connectTimeoutMillis;
        this.socketTimeoutMillis = socketTimeoutMillis;
    }

    @Override
    public String getProviderId() {
        return providerId;
    }

    /**
     * Fetches a previously-{@link #store}d context map. Returns an empty map on a cache miss —
     * matching {@link ContextProvider}'s "no context available" convention, not an exception.
     */
    @Override
    public Map<String, Object> fetch(String key, Map<String, Object> request) throws Exception {
        String fullKey = keyPrefix + key;
        try (Socket socket = openSocket()) {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            writeLine(out, "get " + fullKey);
            String header = readLine(in);
            if (header == null || header.startsWith("END")) {
                return Map.of();
            }
            // "VALUE <key> <flags> <bytes>"
            String[] parts = header.split(" ");
            if (parts.length < 4 || !parts[0].equals("VALUE")) {
                return Map.of();
            }
            int byteCount = Integer.parseInt(parts[3]);
            byte[] payload = readExactly(in, byteCount);
            readLine(in); // trailing \r\n after the payload
            readLine(in); // "END"

            @SuppressWarnings("unchecked")
            Map<String, Object> result = mapper.readValue(payload, Map.class);
            return result;
        }
    }

    /** Stores {@code value} under {@code key} for this provider's default TTL. */
    public void store(String key, Map<String, Object> value) throws Exception {
        store(key, value, defaultTtlSeconds);
    }

    /** Stores {@code value} under {@code key} for {@code ttlSeconds} (0 = never expire, per Memcached's own convention). */
    public void store(String key, Map<String, Object> value, int ttlSeconds) throws Exception {
        String fullKey = keyPrefix + key;
        byte[] payload = mapper.writeValueAsBytes(value);

        try (Socket socket = openSocket()) {
            InputStream in = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            writeLine(out, "set " + fullKey + " 0 " + ttlSeconds + " " + payload.length);
            out.write(payload);
            out.write('\r');
            out.write('\n');
            out.flush();

            String response = readLine(in);
            if (response == null || !response.startsWith("STORED")) {
                throw new IOException("Memcached SET failed for key '" + fullKey + "': " + response);
            }
        }
    }

    public void delete(String key) throws Exception {
        String fullKey = keyPrefix + key;
        try (Socket socket = openSocket()) {
            writeLine(socket.getOutputStream(), "delete " + fullKey);
            readLine(socket.getInputStream()); // "DELETED" or "NOT_FOUND" — either is a fine outcome for delete
        }
    }

    private Socket openSocket() throws IOException {
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress(host, port), connectTimeoutMillis);
        socket.setSoTimeout(socketTimeoutMillis);
        return socket;
    }

    private static void writeLine(OutputStream out, String line) throws IOException {
        out.write(line.getBytes(StandardCharsets.US_ASCII));
        out.write('\r');
        out.write('\n');
        out.flush();
    }

    /** Reads one CRLF-terminated protocol line as ASCII — deliberately not a {@link java.io.BufferedReader}, whose internal buffering would swallow bytes belonging to a following binary payload. */
    private static String readLine(InputStream in) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int prev = -1;
        int b;
        while ((b = in.read()) != -1) {
            if (prev == '\r' && b == '\n') {
                byte[] bytes = buffer.toByteArray();
                return new String(bytes, 0, bytes.length - 1, StandardCharsets.US_ASCII);
            }
            buffer.write(b);
            prev = b;
        }
        return buffer.size() == 0 ? null : buffer.toString(StandardCharsets.US_ASCII);
    }

    private static byte[] readExactly(InputStream in, int count) throws IOException {
        byte[] result = new byte[count];
        int read = 0;
        while (read < count) {
            int n = in.read(result, read, count - read);
            if (n == -1) {
                throw new IOException("Memcached connection closed mid-payload (expected " + count + " bytes, got " + read + ")");
            }
            read += n;
        }
        return result;
    }

    @Override
    public void close() {
        // Stateless (one socket per call) — nothing to release.
    }
}
