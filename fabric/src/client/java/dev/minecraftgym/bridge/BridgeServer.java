package dev.minecraftgym.bridge;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

final class BridgeServer implements AutoCloseable {
    private static final int MAX_MESSAGE_BYTES = 64 * 1024 * 1024;
    private static final Gson GSON = new Gson();

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
                validateProtocol(request);
                if ("hello".equals(string(request, "type"))) {
                    response = hello();
                } else {
                    response = runtime.submit(request).get(120, TimeUnit.SECONDS);
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

    private static JsonObject hello() {
        JsonObject response = new JsonObject();
        response.addProperty("ok", true);
        response.addProperty("protocol_version", BridgeRuntime.PROTOCOL_VERSION);
        response.addProperty("minecraft_version", "1.21");
        response.add("capabilities", GSON.toJsonTree(new String[] {
                "rgb", "human_input", "dagger", "survival_soft_reset", "frozen_tick_step"
        }));
        return response;
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
