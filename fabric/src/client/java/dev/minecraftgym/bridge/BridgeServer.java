package dev.minecraftgym.bridge;

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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class BridgeServer implements AutoCloseable {
    private static final int MAX_MESSAGE_BYTES = 64 * 1024 * 1024;
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private final BridgeRuntime runtime;
    private final int port;
    private volatile boolean running;
    private ServerSocket serverSocket;
    private Thread thread;

    BridgeServer(BridgeRuntime runtime, int port) {
        this.runtime = runtime;
        this.port = port;
    }

    void start() {
        running = true;
        thread = Thread.ofPlatform().name("minecraft-gym-bridge").daemon(true).start(this::run);
    }

    private void run() {
        try (ServerSocket socket = new ServerSocket(port, 1, InetAddress.getLoopbackAddress())) {
            serverSocket = socket;
            while (running) {
                try (Socket client = socket.accept()) {
                    client.setTcpNoDelay(true);
                    client.setSoTimeout(130_000);
                    serve(client);
                } catch (EOFException ignored) {
                    // A client may disconnect between episodes.
                } catch (Exception error) {
                    if (running) {
                        BridgeLog.LOGGER.error("Bridge client session failed", error);
                    }
                } finally {
                    runtime.releaseControls();
                }
            }
        } catch (IOException error) {
            if (running) {
                BridgeLog.LOGGER.error("Could not bind Minecraft Gym Bridge", error);
            }
        }
    }

    private void serve(Socket client) throws Exception {
        DataInputStream input = new DataInputStream(client.getInputStream());
        DataOutputStream output = new DataOutputStream(client.getOutputStream());
        String sessionId = "session:" + UUID.randomUUID();
        String episodeId = null;
        long nextStepId = 0;
        while (running && !client.isClosed()) {
            int length = input.readInt();
            if (length <= 0 || length > MAX_MESSAGE_BYTES) {
                throw new IOException("Invalid bridge message length: " + length);
            }
            byte[] body = input.readNBytes(length);
            if (body.length != length) {
                throw new EOFException("Truncated bridge request");
            }
            JsonObject request = JsonParser.parseString(new String(body, StandardCharsets.UTF_8))
                    .getAsJsonObject();
            JsonObject response;
            try {
                String type = string(request, "type");
                validateProtocol(request);
                if ("hello".equals(type)) {
                    response = hello(sessionId);
                } else {
                    validateSession(request, sessionId);
                    validateChecksum(request);
                    boolean transition = isTransition(type);
                    long stepId = transition ? requireStepId(request, nextStepId) : -1;
                    if ("reset".equals(type)) {
                        episodeId = "episode:" + UUID.randomUUID();
                    }
                    response = runtime.submit(request).get(120, TimeUnit.SECONDS);
                    response.addProperty("session_id", sessionId);
                    if (transition) {
                        if (episodeId == null) {
                            throw new IllegalStateException("reset must start an episode before transitions");
                        }
                        response.addProperty("episode_id", episodeId);
                        response.addProperty("step_id", stepId);
                        nextStepId++;
                    }
                }
            } catch (Exception error) {
                response = error(error);
            }
            byte[] encoded = GSON.toJson(response).getBytes(StandardCharsets.UTF_8);
            output.writeInt(encoded.length);
            output.write(encoded);
            output.flush();
        }
    }

    private static void validateProtocol(JsonObject request) {
        int version = request.has("protocol_version")
                ? request.get("protocol_version").getAsInt()
                : -1;
        if (version != BridgeRuntime.PROTOCOL_VERSION) {
            throw new IllegalArgumentException("Protocol version must be " + BridgeRuntime.PROTOCOL_VERSION);
        }
    }

    private static JsonObject hello(String sessionId) {
        JsonObject response = new JsonObject();
        response.addProperty("ok", true);
        response.addProperty("protocol_version", BridgeRuntime.PROTOCOL_VERSION);
        response.addProperty("bridge_version", "0.2.0");
        response.addProperty("minecraft_version", "1.21");
        response.addProperty("session_id", sessionId);
        response.addProperty("action_contract", "minecraft-native-control/v2");
        response.addProperty("observation_contract", "minecraft-observation/v2");
        response.addProperty("tick_rate_hz", 20);
        response.add("capabilities", GSON.toJsonTree(new String[] {
                "rgb", "human_input", "dagger", "survival_soft_reset", "frozen_tick_step",
                "semantic_events", "privileged_context", "checksummed_requests",
                "contiguous_step_ack"
        }));
        return response;
    }

    private static void validateSession(JsonObject request, String sessionId) {
        if (!request.has("session_id") || !sessionId.equals(request.get("session_id").getAsString())) {
            throw new IllegalArgumentException("request has an invalid session_id");
        }
    }

    private static long requireStepId(JsonObject request, long expected) {
        if (!request.has("step_id")) {
            throw new IllegalArgumentException("transition request has no step_id");
        }
        long actual = request.get("step_id").getAsLong();
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "step_id must be contiguous: expected " + expected + ", received " + actual);
        }
        return actual;
    }

    private static boolean isTransition(String type) {
        return "reset".equals(type) || "step".equals(type);
    }

    private static void validateChecksum(JsonObject request) {
        if (!request.has("payload_sha256")) {
            throw new IllegalArgumentException("request has no payload_sha256");
        }
        String supplied = request.get("payload_sha256").getAsString();
        JsonObject unsigned = request.deepCopy();
        unsigned.remove("payload_sha256");
        String expected = sha256(canonicalJson(unsigned));
        if (!MessageDigest.isEqual(
                supplied.getBytes(StandardCharsets.US_ASCII),
                expected.getBytes(StandardCharsets.US_ASCII))) {
            throw new IllegalArgumentException("payload_sha256 does not match the request");
        }
    }

    /** Match Python json.dumps(sort_keys=True, separators=(",", ":")). */
    private static String canonicalJson(JsonElement value) {
        if (value.isJsonArray()) {
            JsonArray result = new JsonArray();
            for (JsonElement child : value.getAsJsonArray()) {
                result.add(JsonParser.parseString(canonicalJson(child)));
            }
            return GSON.toJson(result);
        }
        if (value.isJsonObject()) {
            JsonObject source = value.getAsJsonObject();
            JsonObject result = new JsonObject();
            List<String> keys = new ArrayList<>(source.keySet());
            keys.sort(Comparator.naturalOrder());
            for (String key : keys) {
                result.add(key, JsonParser.parseString(canonicalJson(source.get(key))));
            }
            return GSON.toJson(result);
        }
        return GSON.toJson(value);
    }

    private static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    private static JsonObject error(Exception error) {
        Throwable cause = error.getCause() == null ? error : error.getCause();
        JsonObject response = new JsonObject();
        response.addProperty("ok", false);
        response.addProperty("error", cause.getClass().getSimpleName() + ": " + cause.getMessage());
        return response;
    }

    private static String string(JsonObject object, String key) {
        return object.has(key) ? object.get(key).getAsString() : "";
    }

    @Override
    public void close() {
        running = false;
        runtime.close();
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException ignored) {
            }
        }
    }
}
