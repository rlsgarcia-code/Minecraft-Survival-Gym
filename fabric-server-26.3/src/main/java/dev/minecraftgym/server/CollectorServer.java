package dev.minecraftgym.server;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.UUID;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class CollectorServer implements AutoCloseable {
    static final int PROTOCOL_VERSION = 2;
    static final int DEFAULT_PORT = 25571;
    private static final int MAX_MESSAGE_BYTES = 16 * 1024 * 1024;
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private final AuthoritativeStateCollector collector;
    private final int port;
    private volatile boolean running;
    private ServerSocket serverSocket;

    CollectorServer(AuthoritativeStateCollector collector, int port) {
        this.collector = collector;
        this.port = port;
    }

    void start() {
        running = true;
        Thread.ofPlatform().name("minecraft-gym-authoritative-collector").daemon(true).start(this::run);
    }

    private void run() {
        try (ServerSocket socket = new ServerSocket(port, 16, InetAddress.getLoopbackAddress())) {
            serverSocket = socket;
            while (running) {
                Socket client = socket.accept();
                Thread.ofVirtual().name("minecraft-gym-collector-client").start(() -> serveSafely(client));
            }
        } catch (IOException error) {
            if (running) {
                MinecraftGymServerCollector.LOGGER.error("Collector failed on 127.0.0.1:{}", port, error);
            }
        }
    }

    private void serveSafely(Socket client) {
        try (client) {
            client.setTcpNoDelay(true);
            client.setSoTimeout(30_000);
            serve(client);
        } catch (EOFException ignored) {
            // Normal client disconnect.
        } catch (Exception error) {
            if (running) {
                MinecraftGymServerCollector.LOGGER.warn("Collector client disconnected after error", error);
            }
        }
    }

    private void serve(Socket client) throws Exception {
        DataInputStream input = new DataInputStream(client.getInputStream());
        DataOutputStream output = new DataOutputStream(client.getOutputStream());
        String sessionId = "collector-session:" + UUID.randomUUID();
        while (running && !client.isClosed()) {
            int length = input.readInt();
            if (length <= 0 || length > MAX_MESSAGE_BYTES) {
                throw new IOException("Invalid collector message length: " + length);
            }
            byte[] body = input.readNBytes(length);
            if (body.length != length) {
                throw new EOFException("Truncated collector request");
            }
            JsonObject wireRequest = JsonParser.parseString(new String(body, StandardCharsets.UTF_8))
                    .getAsJsonObject();
            JsonObject response;
            try {
                JsonObject request = wireRequest.has("payload_json")
                        ? unwrapSignedRequest(wireRequest)
                        : wireRequest;
                String type = string(request, "type");
                validateProtocol(request);
                if ("hello".equals(type)) {
                    response = hello(sessionId);
                } else {
                    if (!wireRequest.has("payload_json")) {
                        throw new IllegalArgumentException("Collector request has no signed payload envelope");
                    }
                    validateSession(request, sessionId);
                    response = switch (type) {
                        case "snapshot" -> snapshot(request, sessionId);
                        case "ack" -> acknowledgement(request, sessionId);
                        default -> throw new IllegalArgumentException("Unknown collector request: " + type);
                    };
                }
            } catch (Exception error) {
                response = error(error);
            }
            byte[] encoded = GSON.toJson(signedEnvelope(response)).getBytes(StandardCharsets.UTF_8);
            output.writeInt(encoded.length);
            output.write(encoded);
            output.flush();
        }
    }

    private JsonObject snapshot(JsonObject request, String sessionId) {
        long after = request.has("after_sequence") ? request.get("after_sequence").getAsLong() : -1;
        int limit = request.has("limit") ? request.get("limit").getAsInt() : 200;
        AuthoritativeStateCollector.Batch batch = collector.after(after, limit);
        JsonObject response = base(sessionId);
        response.addProperty("earliest_sequence", batch.earliestSequence());
        response.addProperty("latest_sequence", batch.latestSequence());
        response.addProperty("gap", batch.gap());
        JsonArray records = new JsonArray();
        batch.records().forEach(records::add);
        response.add("records", records);
        return response;
    }

    private static JsonObject acknowledgement(JsonObject request, String sessionId) {
        JsonObject response = base(sessionId);
        response.addProperty("acked_sequence", request.get("sequence").getAsLong());
        return response;
    }

    private static JsonObject hello(String sessionId) {
        JsonObject response = base(sessionId);
        response.addProperty("collector_version", "0.1.0");
        response.addProperty("minecraft_version", "26.3");
        response.addProperty("tick_rate_hz", 20);
        response.add("capabilities", GSON.toJsonTree(new String[] {
                "authoritative_tick", "pseudonymous_actors", "position", "rotation",
                "velocity", "vitals", "inventory", "effects", "inventory_delta", "death",
                "block_break", "block_interaction", "block_change",
                "resume_after_sequence", "gap_detection", "response_checksums"
        }));
        return response;
    }

    private static JsonObject base(String sessionId) {
        JsonObject response = new JsonObject();
        response.addProperty("ok", true);
        response.addProperty("protocol_version", PROTOCOL_VERSION);
        response.addProperty("session_id", sessionId);
        return response;
    }

    private static JsonObject error(Exception error) {
        JsonObject response = new JsonObject();
        response.addProperty("ok", false);
        response.addProperty("protocol_version", PROTOCOL_VERSION);
        response.addProperty("error", error.getClass().getSimpleName() + ": " + error.getMessage());
        return response;
    }

    private static void validateProtocol(JsonObject request) {
        if (!request.has("protocol_version") || request.get("protocol_version").getAsInt() != PROTOCOL_VERSION) {
            throw new IllegalArgumentException("Collector protocol version must be " + PROTOCOL_VERSION);
        }
    }

    private static void validateSession(JsonObject request, String sessionId) {
        if (!request.has("session_id") || !sessionId.equals(request.get("session_id").getAsString())) {
            throw new IllegalArgumentException("Invalid collector session_id");
        }
    }

    private static JsonObject unwrapSignedRequest(JsonObject envelope) {
        if (!envelope.has("payload_json") || !envelope.has("payload_sha256")) {
            throw new IllegalArgumentException("Collector request has no signed payload envelope");
        }
        String payloadJson = envelope.get("payload_json").getAsString();
        String supplied = envelope.get("payload_sha256").getAsString();
        String expected = sha256(payloadJson);
        if (!MessageDigest.isEqual(
                supplied.getBytes(StandardCharsets.US_ASCII),
                expected.getBytes(StandardCharsets.US_ASCII))) {
            throw new IllegalArgumentException("Collector payload checksum mismatch");
        }
        return JsonParser.parseString(payloadJson).getAsJsonObject();
    }

    private static JsonObject signedEnvelope(JsonObject payload) {
        String payloadJson = GSON.toJson(payload);
        JsonObject envelope = new JsonObject();
        envelope.addProperty("ok", true);
        envelope.addProperty("payload_json", payloadJson);
        envelope.addProperty("payload_sha256", sha256(payloadJson));
        return envelope;
    }

    private static String sha256(String value) {
        try {
            return HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    private static String string(JsonObject object, String key) {
        return object.has(key) ? object.get(key).getAsString() : "";
    }

    @Override
    public void close() {
        running = false;
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {
            }
        }
    }
}
